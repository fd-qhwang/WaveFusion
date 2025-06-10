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

_reduction_modes = ['none', 'mean', 'sum']



@LOSS_REGISTRY.register()
class OhemCELoss(nn.Module):
    """Online Hard Example Mining with Cross Entropy Loss.

    Args:
        thresh (float): Threshold for hard example mining. Default: 0.7.
        n_min (int): Minimum number of hard examples. Default: 16.
        ignore_lb (int): Label to ignore during loss computation. Default: 255.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        loss_weight (float): Loss weight for the output. Default: 1.0.
    """

    def __init__(self, thresh=0.75, n_min=16 * 256 * 256 // 16, ignore_lb=255, reduction='mean', loss_weight=1.0):
        super(OhemCELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: ["none", "mean", "sum"]')

        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, logits, labels):
        """
        Args:
            logits (Tensor): of shape (N, C, H, W). Raw predictions from the model.
            labels (Tensor): of shape (N, 1, H, W). True labels.
        """
        labels = labels.squeeze(1)  # Reduce the channel dimension.
        N, C, H, W = logits.size()
        #labels = labels.long()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)

        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]

        if self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'mean':
            loss = torch.mean(loss)

        return self.loss_weight * loss

@LOSS_REGISTRY.register()
class BCEWithLogitsLoss(nn.Module):
    """Binary Cross Entropy with Logits Loss.

    Args:
        loss_weight (float): Loss weight for BCE loss. Default: 1.0.
        class_weights (list or tuple): Weights for each class. Default: None.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, class_weights=None, reduction='mean'):
        super(BCEWithLogitsLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights).float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        else:
            self.class_weights = None

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, 1, H, W). Ground truth tensor.
        """
        target_one_hot = F.one_hot(target.squeeze(1).long(), num_classes=pred.size(1))
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        return self.loss_weight * F.binary_cross_entropy_with_logits(pred, target_one_hot, pos_weight=self.class_weights, reduction=self.reduction)

@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        x = x.repeat_interleave(3, dim=1)
        gt = gt.repeat_interleave(3, dim=1)
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram