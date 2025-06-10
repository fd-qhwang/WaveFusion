import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# 加上这句话可以解决绝对引用的问题，但是同时导致了相对引用的问题
import sys,os
sys.path.append(os.getcwd())
import math
import numpy as np
from utils.registry import ARCH_REGISTRY
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
import sys,os
from torchinfo import summary
import time
from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D, DWT_2D_tiny
from thop import profile
import torch.utils.checkpoint as checkpoint


#########################################
# Downsample Block
class Downsample(nn.Module):
    '''
    input_shape(B, C, D, H, W)
    output_shape(B, 2*C, D/2, H/2, W/2)
    作用：使用卷积进行降采样，同时扩大通道数，可以用三维池化加线性层代替
    '''
    def __init__(self, in_channels, out_channels, down_type='dwt', wavename='haar'):
        super(Downsample, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.type = down_type
        if self.type=='dwt':
            self.reduce = nn.Sequential(
                nn.Conv2d(in_channels, out_channels//4, kernel_size=1, padding=0, stride=1),
                nn.LeakyReLU(inplace=True),
            )
            self.dwt_down = DWT_2D(wavename=wavename)
            self.filter = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
                nn.LeakyReLU(inplace=True),
            )
        else:
            self.conv_down = nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)

    def forward(self, x):

        if (self.type == 'dwt'):
            x_d = self.reduce(x)
            x_LL, x_LH, x_HL, x_HH = self.dwt_down(x_d)
            x_down = torch.cat([x_LL, x_LH, x_HL, x_HH], dim=1)
            x_down = self.filter(x_down)
            return x_down
        
        elif (self.type == 'conv'):
            x_down = self.conv_down(x)
            return x_down
        else:
            return x
            
# Upsample Block
class Upsample(nn.Module):
    '''
    input_shape(B, C, D, H, W)
    output_shape(B, C/2, 2*D, 2*H, 2*W)
    作用：使用卷积进行上采样，同时减小通道数，可以用三维池化加线性层代替
    '''
    def __init__(self, in_channels, out_channels, up_type='dwt', wavename='haar'):
        super(Upsample, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.type = up_type

        self.conv_up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=True)
        self.dilate = nn.Sequential(
            nn.Conv2d(in_channels, out_channels*4, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(inplace=True),
        )
        self.dwt_up = IDWT_2D(wavename=wavename)
        self.filter = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(inplace=True),
        )

    def _Itransformer(self,out):
        C=int(out.shape[1]/4)
        # print(out.shape[0])
        y = out.reshape((out.shape[0], C, 4, out.shape[-2], out.shape[-1]))
        yll, ylh, yhl, yhh = y[:,:,0].contiguous(), y[:,:,1].contiguous(), y[:,:,2].contiguous(), y[:,:,3].contiguous()
        return yll, ylh, yhl, yhh

    def forward(self, x):
        if self.type == 'dwt':
            x = self.dilate(x)
            x_LL, x_Lh, x_HL, x_HH = self._Itransformer(x)
            x = self.dwt_up(x_LL, x_Lh, x_HL, x_HH)
            x = self.filter(x)
        elif self.type == 'conv':
            x = self.conv_up(x)

        return x
    
class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)

        # TODO recover
        # self.relative_position_params = nn.Parameter(torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size -1))
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # supporting sqaure.
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True; 
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        # sqaure validation
        # assert h_windows == w_windows

        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        # Adding learnable relative embedding
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        # Using Attn Mask to distinguish different subwindows.
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        # negative is allowed
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]

class SwinTransBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(SwinTransBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        if input_resolution <= window_size:
            self.type = 'W'

        #print("Block Initial Type: {}, drop_path_rate:{:.6f}".format(self.type, drop_path))
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )
    
    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x

class WaveTransBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(WaveTransBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        if input_resolution <= window_size:
            self.type = 'W'

        #print("Block Initial Type: {}, drop_path_rate:{:.6f}".format(self.type, drop_path))
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )
        self.dwt_down = DWT_2D(wavename='haar')
        self.conv_high_freq = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(inplace=True),
        )
        self.dwt_up = IDWT_2D(wavename='haar')
        self.conv_1x1 = nn.Conv2d(input_dim, input_dim, kernel_size=1, padding=0, stride=1)
    
    def forward(self, x):
        residual = x  # 保存原始输入作为残差连接的一部分

        x = self.ln1(x)  # 进行层归一化
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] to [B, C, H, W]
        X_LL, X_LH, X_HL, X_HH = self.dwt_down(x)
        X_LL = self.msa(X_LL.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # 处理高频分量
        X_high_freq = [X_LH, X_HL, X_HH]
        X_high_freq = [self.conv_high_freq(component) for component in X_high_freq]

        x = self.dwt_up(X_LL, *X_high_freq)
        x = self.conv_1x1(x)
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] to [B, H, W, C]

        # 应用残差连接
        x = residual + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.ln2(x)))

        return x

    
class WaveLayer(nn.Module):
    """ A basic layer of Swin Transformer consisting of alternating W and SW blocks. """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None, upsample=None):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.dim = dim
        self.input_resolution = input_resolution
        self.downsample = downsample
        self.upsample = upsample

        # downsample layer
        if downsample is not None:
            self.downsample = downsample(
                        in_channels=dim,
                        out_channels=dim,
                        down_type='dwt'
                        )
            self.input_resolution = input_resolution // 2
        else:
            self.downsample = None
            
        for i in range(depth):
            # Alternate between 'W' and 'SW' types for each block
            block_type = 'W' if i % 2 == 0 else 'SW'
            head_dim = dim // num_heads

            block = WaveTransBlock(input_dim=dim, output_dim=dim, head_dim=head_dim,
                                   window_size=window_size, drop_path=drop_path,
                                   type=block_type, input_resolution=self.input_resolution)
            
            self.blocks.append(block)

        # upsample layer
        if upsample is not None:
            self.upsample = upsample(dim, dim, up_type='dwt')
        else:
            self.upsample = None

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        x = Rearrange('b c h w -> b h w c')(x)
        for block in self.blocks:
            x = block(x)
        x = Rearrange('b h w c -> b c h w')(x)
        if self.upsample is not None:
            x = self.upsample(x)

        return x

    
class WaveSTLayer(nn.Module):
    """ 
    WaveSTLayer: A Swin Transformer layer with variable depths and head numbers,
    integrated with wavelet-based down/up-sampling for multi-scale representation.
    """
    def __init__(self, dim, input_resolution, depths, num_heads_list, window_size,
                 mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None, upsample=None):
        super().__init__()

        self.layers = nn.ModuleList()

        for i, (depth, num_heads) in enumerate(zip(depths, num_heads_list)):
            layer = WaveLayer(
                dim=dim,
                input_resolution=input_resolution,
                depth=depth,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
                norm_layer=norm_layer,
                downsample=downsample if i < len(depths) - 1 else None,
                upsample=upsample if i < len(depths) - 1 else None
            )
            self.layers.append(layer)

        # Add a 3x3 convolutional layer for feature refinement
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = x

        for layer in self.layers:
            x = layer(x)

        x = self.conv(x)
        x += residual  # Add residual connection

        return x