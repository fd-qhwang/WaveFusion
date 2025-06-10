# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
from torch import tensor
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchinfo import summary
import sys,os
from einops import rearrange
from archs.arch_utils import LayerNorm, FeedForward, Attention
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

    
class DualModalAttention(nn.Module):
    def __init__(self, dim=36, num_heads=6, bias=False):
        super(DualModalAttention, self).__init__()

        assert dim % num_heads == 0, "Dimension must be divisible by the number of heads"

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1))  # 可学习的平衡因子

        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv1 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.qkv2 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x1, x2):
        b, c, h, w = x1.shape

        qkv1 = self.qkv_dwconv1(self.qkv1(x1))
        q1, k1, v1 = qkv1.chunk(3, dim=1)
        
        qkv2 = self.qkv_dwconv2(self.qkv2(x2))
        q2, k2, v2 = qkv2.chunk(3, dim=1)

        enhanced_x1 = self.compute_attention(q1, k1, v1, k2, v2, b, h, w)
        enhanced_x2 = self.compute_attention(q2, k2, v2, k1, v1, b, h, w)

        #out_x1 = self.project_out(enhanced_x1) + x1
        #out_x2 = self.project_out(enhanced_x2) + x2
        out_x1 = self.project_out(enhanced_x1)
        out_x2 = self.project_out(enhanced_x2)

        return out_x1, out_x2

    def compute_attention(self, q, k_self, v_self, k_cross, v_cross, b, h, w):
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_self = rearrange(k_self, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_self = rearrange(v_self, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_cross = rearrange(k_cross, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_cross = rearrange(v_cross, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k_self = torch.nn.functional.normalize(k_self, dim=-1)
        k_cross = torch.nn.functional.normalize(k_cross, dim=-1)

        self_attn = (q @ k_self.transpose(-2, -1)) * self.temperature
        self_attn = self_attn.softmax(dim=-1)
        self_out = self_attn @ v_self

        cross_attn = (q @ k_cross.transpose(-2, -1)) * self.temperature
        cross_attn = (-cross_attn).softmax(dim=-1)
        cross_out = cross_attn @ v_cross

        beta = torch.sigmoid(self.beta)
        out_combined = beta * self_out + cross_out
        out = rearrange(out_combined, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        return out


class DAIMBlock(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type="WithBias"):
        super(DAIMBlock, self).__init__()

        # Layer normalization for x1 and x2
        self.norm1_x1 = LayerNorm(dim, LayerNorm_type)
        self.norm1_x2 = LayerNorm(dim, LayerNorm_type)

        # Cross attention
        self.cross_attn = DualModalAttention(dim, num_heads, bias)

        # Layer normalization after cross attention for x1 and x2
        self.norm2_x1 = LayerNorm(dim, LayerNorm_type)
        self.norm2_x2 = LayerNorm(dim, LayerNorm_type)

        # Feed-forward networks for x1 and x2
        self.ffn_x1 = FeedForward(dim, ffn_expansion_factor, bias)
        self.ffn_x2 = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x1, x2):
        # Cross attention
        enhanced_x1, enhanced_x2 = self.cross_attn(self.norm1_x1(x1), self.norm1_x2(x2))
        x1 = x1 + enhanced_x1
        x2 = x2 + enhanced_x2

        # Feed-forward networks
        x1 = x1 + self.ffn_x1(self.norm2_x1(x1))
        x2 = x2 + self.ffn_x2(self.norm2_x2(x2))

        return x1, x2
