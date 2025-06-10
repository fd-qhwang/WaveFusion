import torch
from torch import nn as nn
from torch.nn import functional as F

from archs.vgg_arch import VGGFeatureExtractor
from utils.registry import LOSS_REGISTRY
import kornia
import kornia.losses
import torch.nn.functional as F
from .loss_util import weighted_loss
import numpy as np
import math
from math import exp

_reduction_modes = ['none', 'mean', 'sum']

@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)



class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)
        #return torch.abs(sobelx), torch.abs(sobely)

@LOSS_REGISTRY.register()
class SobelLoss1(nn.Module):
    def __init__(self, reduction='mean',loss_weight=1.0):
        super(SobelLoss1, self).__init__()
        self.sobelconv = Sobelxy()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, image_1, image_2, generate_img):
        grad_1 = self.sobelconv(image_1)
        grad_2 = self.sobelconv(image_2)
        generate_img_grad = self.sobelconv(generate_img)

        # Taking the element-wise maximum of the gradients of the two input images
        x_grad_joint = torch.max(grad_1, grad_2)

        # Computing the L1 loss between the maximum gradient and the gradient of the generated image
        loss = F.l1_loss(x_grad_joint, generate_img_grad,reduction=self.reduction)

        return loss * self.loss_weight
    
@LOSS_REGISTRY.register()
class SobelLoss2(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(SobelLoss2, self).__init__()
        self.sobel_conv = Sobelxy()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, img_vi, img_ir, img_fu):
        vi_grad_x, vi_grad_y = self.sobel_conv(img_vi)
        ir_grad_x, ir_grad_y = self.sobel_conv(img_ir)
        fu_grad_x, fu_grad_y = self.sobel_conv(img_fu)
        
        grad_joint_x = torch.max(vi_grad_x, ir_grad_x)
        grad_joint_y = torch.max(vi_grad_y, ir_grad_y)
        
        loss = F.l1_loss(grad_joint_x, fu_grad_x, reduction=self.reduction) + \
               F.l1_loss(grad_joint_y, fu_grad_y, reduction=self.reduction)
        
        return loss * self.loss_weight


        
@LOSS_REGISTRY.register()
class MaskLoss(nn.Module):
    """
    Computes the mask-based MSE loss based on the given logic.

    Args:
        window_size (int): The size of the window to compute local operations. Default: 9.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, window_size=9, reduction='mean', loss_weight=1.0):
        super(MaskLoss, self).__init__()
        self.window_size = window_size
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, img_ir, img_vis, img_fuse, mask=None):

        # 计算最大值
        max_val = torch.max(img_ir, img_vis)
        # 计算最大值损失，忽略通道维度，保持与max_mask维度一致
        max_loss = F.mse_loss(img_fuse, max_val, reduction='none')
        avg_loss = (F.mse_loss(img_fuse, img_ir, reduction='none') + F.mse_loss(img_fuse, img_vis, reduction='none')) / 2
        
        max_mask = mask
        avg_mask = 1 - mask
        
        res = max_mask * max_loss + avg_mask * avg_loss
        #print(res.shape)
        
        if self.reduction == 'mean':
            return (res.mean() * self.loss_weight)
        elif self.reduction == 'sum':
            return (res.sum() * self.loss_weight)
        else:
            return res * self.loss_weight
    