from torch.utils.data.dataset import Dataset
from PIL import Image
import torchvision
import torchvision.transforms.functional as TF
import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
import torch.utils.data as Data
import sys
import os
sys.path.append(os.getcwd())

from utils.registry import DATASET_REGISTRY
from utils.utils import randrot, randfilp, RGB2YCrCb, YCbCr2RGB
from natsort import natsorted
import cv2

def min_max_normalize(tensor):
    """归一化张量到[0,1]范围"""
    min_val, max_val = tensor.min(), tensor.max()
    return (tensor - min_val) / (max_val - min_val) if max_val > min_val else tensor

def rgb2y(img):
    """RGB转Y通道"""
    return img[0:1] * 0.299 + img[1:2] * 0.587 + img[2:3] * 0.114

def bgr_to_ycrcb(path):
    """BGR图像转YCbCr颜色空间"""
    img = cv2.imread(path, 1).astype('float32')
    B, G, R = cv2.split(img)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    return Y, cv2.merge([Cr, Cb])

@DATASET_REGISTRY.register()
class WAVFUDataset(Data.Dataset):
    """红外和可见光图像融合数据集
    
    属性:
        opt (dict): 配置参数字典
        is_train (bool): 是否为训练模式
        img_size (int): 图像大小
        stride (int): 滑动窗口步长
        is_RGB (bool): 是否使用RGB模式
    """
    
    def __init__(self, opt):
        super(WAVFUDataset, self).__init__()
        self.opt = opt
        self.is_train = opt['is_train']
        self.img_size = int(opt['img_size']) if self.is_train else 128
        self.arg = opt.get('is_arg', True)
        self.stride = opt.get('stride', 128)
        self.is_RGB = opt.get('is_RGB', False)
        
        # 计算图像块数量
        h, w = 480, 640
        self.patch_per_line = (w - self.img_size) // self.stride + 1
        self.patch_per_colum = (h - self.img_size) // self.stride + 1
        self.patch_per_img = self.patch_per_line * self.patch_per_colum
        
        # 设置数据路径
        self.vis_folder = opt['dataroot_source2']
        self.ir_folder = opt['dataroot_source1']
        
        if self.is_train:
            self.mask_folder = opt.get('mask_path')
        else:
            self.ir_list = natsorted(os.listdir(self.ir_folder))
            
        self.crop = torchvision.transforms.RandomCrop(self.img_size)
        self.ir_list = natsorted(os.listdir(self.ir_folder))
        
    def _load_image(self, path, is_visible=True):
        """加载并预处理图像"""
        if is_visible:
            img = Image.open(path).convert('RGB')
        else:
            img = Image.open(path).convert('L')
        img_tensor = TF.to_tensor(img).unsqueeze(0)
        return min_max_normalize(img_tensor)
    
    def _augment(self, img):
        """数据增强"""
        # 随机旋转
        rot_times = random.randint(0, 3)
        img = torch.rot90(img, rot_times, [-2, -1])
        
        # 随机翻转
        if random.random() > 0.5:
            img = torch.flip(img, [-1])
        if random.random() > 0.5:
            img = torch.flip(img, [-2])
            
        return img
    
    def __getitem__(self, idx):
        if self.is_train:
            # 计算图像块位置
            img_idx, patch_idx = idx // self.patch_per_img, idx % self.patch_per_img
            h_idx, w_idx = patch_idx // self.patch_per_line, patch_idx % self.patch_per_line
            image_name = self.ir_list[img_idx]
            
            # 加载图像
            vis = self._load_image(os.path.join(self.vis_folder, image_name), True)
            ir = self._load_image(os.path.join(self.ir_folder, image_name), False)
            
            # 加载掩码（如果存在）
            mask = torch.zeros(1, *ir.shape[1:])
            if self.mask_folder:
                mask = self._load_image(os.path.join(self.mask_folder, image_name), False)
            
            # 组合并裁剪
            combined = torch.cat((vis, ir, mask), dim=1)
            patch = combined[:, :, 
                           h_idx * self.stride:h_idx * self.stride + self.img_size,
                           w_idx * self.stride:w_idx * self.stride + self.img_size]
            
            # 数据增强
            if self.arg:
                patch = self._augment(patch)
            
            # 分离通道
            vis, ir, mask = torch.split(patch, [3, 1, 1], dim=1)
            vis = vis.squeeze(0)
            ir, mask = ir.squeeze(0), mask.squeeze(0)
            
            # 转换为灰度图（如果需要）
            if not self.is_RGB:
                vis = rgb2y(vis)
            
            return {
                'VIS': vis,
                'IR': ir,
                'MASK': mask.long()
            }
        else:
            # 测试模式
            image_name = self.ir_list[idx]
            vis = self._load_image(os.path.join(self.vis_folder, image_name), True)
            ir = self._load_image(os.path.join(self.ir_folder, image_name), False)
            
            vis, ir = vis.squeeze(0), ir.squeeze(0)
            vis = rgb2y(vis)
            
            data = {
                'IR': ir,
                'VIS': vis,
                'IR_path': os.path.join(self.ir_folder, image_name),
                'VIS_path': os.path.join(self.vis_folder, image_name)
            }
            
            if self.is_RGB:
                _, cbcr = bgr_to_ycrcb(data['VIS_path'])
                data['CBCR'] = cbcr
                
            return data
    
    def __len__(self):
        return self.patch_per_img * len(self.ir_list) if self.is_train else len(self.ir_list) 