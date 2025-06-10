
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import math
import numbers
# 加上这句话可以解决绝对引用的问题，但是同时导致了相对引用的问题
import sys,os
sys.path.append(os.getcwd())

from utils.registry import ARCH_REGISTRY
from einops import rearrange
import sys,os
from torchinfo import summary
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn import init
from archs.wave_vit import WaveSTLayer
from archs.odconv import ODConv2d
from archs.wavelet_block import LWaveSTLayer
def odconv3x3(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
    return ODConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                    reduction=reduction, kernel_num=kernel_num)


def odconv1x1(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
    return ODConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0,
                    reduction=reduction, kernel_num=kernel_num)

class FReLU(nn.Module):
    r""" FReLU formulation. The funnel condition has a window size of kxk. (k=3 by default)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv_frelu = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.bn_frelu = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x1 = self.conv_frelu(x)
        x1 = self.bn_frelu(x1)
        x2 = torch.stack([x, x1], dim=0)
        out, _ = torch.max(x2, dim=0)
        return out
    
##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        #self.qkv_dwconv = odconv3x3(dim*3, dim*3, reduction=0.0625, kernel_num=1)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
       
class ConcatFusion(nn.Module):
    def __init__(self, feature_dim):
        super(ConcatFusion, self).__init__()
        self.dim_reduce = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(inplace=True),
            nn.Hardswish(inplace=False),
            #nn.LeakyReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(inplace=True),
            #nn.LeakyReLU(inplace=True),
        )
                    
    def forward(self, modality1, modality2):
        concat_features = torch.cat([modality1, modality2], dim=1)
        return self.dim_reduce(concat_features)


##---------- Share Encoder -----------------------
#@ARCH_REGISTRY.register()

class Encoder(nn.Module):
    def __init__(self, input_channels=1, output_channels=64):
        super(Encoder, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(),
            #nn.Hardswish(inplace=False),
            nn.Conv2d(output_channels, output_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.encoder1(x)
        return x

        
class Decoder_WT(nn.Module):
    def __init__(self, inp_channels, out_channels, dim, input_resolution, depths, num_heads_list, window_size,
                 mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None, upsample=None):
        super(Decoder_WT, self).__init__()
        # Initial convolution-activation to enhance non-linearity
        self.fusion = ConcatFusion(inp_channels)

        self.refinement = WaveSTLayer(
            dim=dim,
            input_resolution=input_resolution,
            depths=depths,  # 假设层中有两个块
            num_heads_list=num_heads_list,
            window_size=window_size,
            drop_path=drop_path,
            downsample=None,
            upsample=None
        )
        # Convolution to produce the desired output channels
        self.decoder = nn.Sequential(
            #nn.LeakyReLU(inplace=True),
            #FReLU(inp_channels//2),
            nn.Conv2d(inp_channels, inp_channels, kernel_size=3, stride=1, padding=1),
            #nn.Hardswish(inplace=False),
            nn.LeakyReLU(),
            nn.Conv2d(inp_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(out_channels),
        )
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, inp_img=None):
        #x = self.fusion(x1, x2)
        x = self.refinement(x)
        x = self.decoder(x)

        # If inp_img is provided, add it to the output
        if inp_img is not None:
            x = x + inp_img

        return (self.tanh(x) + 1) / 2
        #return self.sigmoid(x)
        #return x
        
class Decoder_LWT(nn.Module):
    def __init__(self, inp_channels, out_channels, dim, input_resolution, depths, num_heads_list, window_size,
                 mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None, upsample=None):
        super(Decoder_LWT, self).__init__()
        # Initial convolution-activation to enhance non-linearity
        self.fusion = ConcatFusion(inp_channels)

        self.refinement = LWaveSTLayer(
            dim=dim,
            input_resolution=input_resolution,
            depths=depths,  # 假设层中有两个块
            num_heads_list=num_heads_list,
            window_size=window_size,
            drop_path=drop_path,
            downsample=None,
            upsample=None
        )
        # Convolution to produce the desired output channels
        self.decoder = nn.Sequential(
            #nn.LeakyReLU(inplace=True),
            #FReLU(inp_channels//2),
            nn.Conv2d(inp_channels, inp_channels, kernel_size=3, stride=1, padding=1),
            #nn.Hardswish(inplace=False),
            nn.LeakyReLU(),
            nn.Conv2d(inp_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(out_channels),
        )
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, inp_img=None):
        #x = self.fusion(x1, x2)
        x = self.refinement(x)
        x = self.decoder(x)

        # If inp_img is provided, add it to the output
        if inp_img is not None:
            x = x + inp_img

        return (self.tanh(x) + 1) / 2
        #return self.sigmoid(x)
        #return x

    def get_wavelet_loss(self):
        wavelet_loss = 0.0  # 初始化为0
        wavelet_loss += self.refinement.get_wavelet_loss()  # 累加 refinement 层的 wavelet_loss
        return wavelet_loss
