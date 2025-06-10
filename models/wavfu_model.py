import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import numpy as np
import os
os.environ['TF\_ENABLE\_ONEDNN\_OPTS'] = '0'

from archs import build_network
from losses import build_loss
from metrics import calculate_metric
from utils import get_root_logger, imwrite, tensor2img
from utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import torch.nn as nn
from utils.utils import *
import cv2

def ycrcb_to_bgr(one):
    one = one.astype('float32')
    Y, Cr, Cb = cv2.split(one)
    B = (Cb - 0.5) * 1. / 0.564 + Y
    R = (Cr - 0.5) * 1. / 0.713 + Y
    G = 1. / 0.587 * (Y - 0.299 * R - 0.114 * B)
    return cv2.merge([B, G, R])


def RGB2GRAY(rgb_image):
    """
    将RGB格式转换为GRAY格式。
    :param rgb_image: RGB格式的图像数据, 其shape为[B, C, H, W], 其中C=3。
    :return: 灰度图像
    """

    R = rgb_image[0:1]
    G = rgb_image[1:2]
    B = rgb_image[2:3]

    # 使用上述公式计算灰度值
    gray = 0.299 * R + 0.587 * G + 0.114 * B

    # 限制值在[0,1]范围内
    #gray = gray.clamp(0.0, 1.0)

    return gray

def RGB2YCrCb(input_im):
    """将RGB图像转换为YCrCb颜色空间"""
    device = input_im.device
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).to(device)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out[:, :1], out[:, 1:2], out[:, 2:3]

def merge_channels(Y, Cr, Cb):
    """Merge separate Y, Cr, Cb tensors into a single tensor."""
    # Stack the channels along the channel dimension (dimension 1)
    return torch.cat((Y, Cr, Cb), dim=1)

def YCrCb2RGB(Y, Cr, Cb):
    """将YCrCb颜色空间转换回RGB"""
    input_im = torch.cat((Y, Cr, Cb), dim=1)
    device = input_im.device
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(device)
    temp = (im_flat + bias).mm(mat)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def compute_class_distribution(seg_result):
    B, _, H, W = seg_result.shape
    total_pixels = H * W
    class_distribution = []

    for b in range(B):
        img = seg_result[b, 0]
        class_counts = torch.bincount(img.view(-1))
        class_proportions = class_counts.float() / total_pixels
        class_distribution.append(class_proportions)

    return torch.stack(class_distribution)

@MODEL_REGISTRY.register()
class WAVFUModel(BaseModel):
    """Base IF model for infrares-visible image fusion."""

    def __init__(self, opt):
        super(WAVFUModel, self).__init__(opt)
        torch.set_float32_matmul_precision('high')
        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        #self.print_network(self.net_g)
        self.clip_grad_norm_value = 0.1

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        """初始化训练设置"""
        train_opt = self.opt['train']
        self.clip_grad_norm_value = 0.01

        # EMA设置
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'使用指数移动平均，衰减率: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)
            self.net_g_ema.eval()

        self.net_g.train()

        # 定义损失函数
        self.losses = ["content_mask", "edge", "percep_vis", "percep_ir"]
        for loss_name in self.losses:
            if train_opt.get(f'{loss_name}_opt'):
                loss_criterion = build_loss(train_opt[f'{loss_name}_opt']).to(self.device)
                setattr(self, f'cri_{loss_name}', loss_criterion)
            else:
                setattr(self, f'cri_{loss_name}', None)

        # 小波损失设置
        if train_opt.get("wavelet_opt"):
            self.cri_wavelet = train_opt['wavelet_opt'].get('use_loss', False)
        else:
            self.cri_wavelet = False

        # 确保至少有一个损失函数
        if not any(getattr(self, f'cri_{loss_name}') for loss_name in self.losses):
            raise ValueError('所有损失函数都未定义')

        # 设置优化器和调度器
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        """加载数据"""
        self.VIS = data['VIS'].to(self.device)
        self.IR = data['IR'].to(self.device)
        if 'CBCR' in data:
            self.CBCR = data['CBCR'].to(self.device)
        if 'MASK' in data:
            self.MASK = data['MASK'].to(self.device)

    def optimize_parameters(self, current_iter):
        """优化参数"""
        self.optimizer_g.zero_grad()
        self.data_fusion = self.net_g(self.VIS, self.IR)

        # 计算损失
        loss_conditions = [
            ("cri_content_mask", "l_content_masx1", self.IR,self.VIS, self.data_fusion, self.MASK),
            ("cri_edge", "l_edge_sum", self.IR, self.VIS, self.data_fusion),
            ("cri_percep_vis", "l_percep_vis", self.VIS, self.data_fusion),
            ("cri_percep_ir", "l_percep_ir", self.IR, self.data_fusion),
        ]
        l_total, loss_dict = self.compute_losses(loss_conditions)

        # 小波损失
        if self.cri_wavelet:
            l_wavelet = self.net_g.get_wavelet_loss()
            l_total += l_wavelet
            loss_dict['l_wavelet'] = l_wavelet

        l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

        # 反向传播和优化
        l_total.backward()
        if self.clip_grad:
            nn.utils.clip_grad_norm_(self.net_g.parameters(), max_norm=self.clip_grad_norm_value, norm_type=2)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        """测试模型"""
        self.VIS, pads = self.pad_to(self.VIS, stride=32, scale=1)
        self.IR, pads = self.pad_to(self.IR, stride=32, scale=1)

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.data_fusion = self.net_g_ema(self.VIS, self.IR)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.data_fusion = self.net_g(self.VIS, self.IR)
            self.net_g.train()

        self.data_fusion = self.unpad(self.data_fusion, pads)
        self.VIS = self.unpad(self.VIS, pads)
        self.IR = self.unpad(self.IR, pads)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics')
        #with_metrics = (self.opt['val'].get('metrics') is not None) and (current_iter > self.phase1_iter)
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)

        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')
        compute_miou = False  # Default to True if the field is not present
        seg_metric = SegmentationMetric(9, device=self.device)
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['IR_path'][0]))[0]
            self.feed_data(val_data)
            self.test()
            visuals = self.get_current_visuals()
            #VIS_img = tensor2img([visuals['VIS']]) # 3 h w
            VIS_img = tensor2img(self.VIS.detach().cpu()) # 1 h w
            IR_img = tensor2img([visuals['IR']]) # 1 h w
            CBCR_img = visuals['CBCR']
            fusion_img = tensor2img([visuals['result']])
            fusion_img=(fusion_img-np.min(fusion_img))/(np.max(fusion_img)-np.min(fusion_img))
            fusion_img, VIS_img, IR_img = fusion_img * 255, VIS_img * 255, IR_img * 255
            metric_data['img_fusion'] = fusion_img.squeeze(0)
            metric_data['img_A'] = VIS_img.squeeze(0)
            metric_data['img_B'] = IR_img.squeeze(0)
            if 'LABEL' in val_data:
                pass
                #compute_miou = True
                #self.seg_result = torch.argmax(self.semantic_pred, dim=1, keepdim=True)
                #seg_metric.addBatch(self.seg_result, self.LABEL, [255])
            # tentative for out of GPU memory
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = self._get_save_path(dataset_name, img_name, current_iter)
                else:
                    save_img_path = self._get_save_path(dataset_name, img_name, current_iter)
                #print(fusion_img.shape)
                #print(type(fusion_img))
                #print(CBCR_img.shape)
                vi_filepath = val_data['VIS_path'][0]
                fuse_y = fusion_img.squeeze(0)
                img_vi  = cv2.imread(vi_filepath, flags=cv2.IMREAD_COLOR)
                # get cb and cr channels of the visible image
                vi_ycbcr = cv2.cvtColor(img_vi, cv2.COLOR_BGR2YCrCb)
                vi_y  = vi_ycbcr[:, :, 0]
                vi_cb = vi_ycbcr[:, :, 1]
                vi_cr = vi_ycbcr[:, :, 2]
                # get BGR-fused image
                fused_ycbcr = np.stack([fuse_y, vi_cb, vi_cr], axis=2).astype(np.uint8)
                fused_bgr = cv2.cvtColor(fused_ycbcr, cv2.COLOR_YCrCb2BGR)
                color_fusion = ycrcb_to_bgr(cv2.merge([fusion_img.transpose(1, 2, 0), CBCR_img]))
                #cv2.imwrite(save_img_path, fused_bgr)
                imwrite(save_img_path, fused_bgr)
                #imwrite(metric_data['img_fusion'], save_img_path)
                
                
            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()
        if compute_miou == True:
            self.mIoU = np.array(seg_metric.meanIntersectionOverUnion().item())
            self.Acc = np.array(seg_metric.pixelAccuracy().item())
            logger = get_root_logger()
            logger.info('mIou: {:.4f}, Acc: {:.4f}\n'.format(self.mIoU, self.Acc))
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            best_metric_value = self.best_metric_results[dataset_name][metric]["val"]
            best_metric_iter = self.best_metric_results[dataset_name][metric]["iter"]
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'
            if (value==best_metric_value) and (self.is_train==True):
                print(f'Saving best %s models and training states.' % metric)
                self.save_best(metric, best_metric_iter)

        logger = get_root_logger()
        #logger.info('mIou: {:.4f}, Acc: {:.4f}\n'.format(self.mIoU, self.Acc))
        if not self.is_train:
            print(log_str) # 由于某些原因在test阶段需要print才行
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        #out_dict['VIS'] = self.VIS.detach().cpu()
        out_dict['VIS'] = (self.VIS.detach().cpu())
        out_dict['IR'] = self.IR.detach().cpu()
        # 使用hasattr函数检查类实例中是否有这些属性，并进行赋值
        if hasattr(self, 'CBCR'):
            out_dict['CBCR'] = self.CBCR.detach().cpu().squeeze(0).numpy()
        if hasattr(self, 'data_fusion'):
            out_dict['result'] = self.data_fusion.detach().cpu()

        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

    def _get_save_path(self, dataset_name, img_name, current_iter):
        """获取保存路径"""
        if self.opt['is_train']:
            return osp.join(self.opt['path']['visualization'], 'train',
                          f"{current_iter:06d}_{img_name}.png")
        else:
            suffix = self.opt['val'].get('suffix', '')
            return osp.join(self.opt['path']['visualization'], dataset_name,
                          f"{img_name}_{suffix}.png" if suffix else f"{img_name}.png")
