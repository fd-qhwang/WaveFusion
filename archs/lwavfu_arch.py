
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# 加上这句话可以解决绝对引用的问题，但是同时导致了相对引用的问题
import sys,os
sys.path.append(os.getcwd())

from utils.registry import ARCH_REGISTRY
from archs.arch_utils import Encoder, Decoder_WT, Decoder_LWT, ConcatFusion
from archs.attn_utils import DAIMBlock
from archs.wave_vit import WaveSTLayer
from archs.wavelet_block import LWaveSTLayer
from einops import rearrange
import sys,os
from torchinfo import summary
                 
@ARCH_REGISTRY.register()
class LWAVFUNet(nn.Module):
    def __init__(self, 
                 inp_channels=1, 
                 dim=64,
                 wavestlayer_configs={},
                 decoder_configs={},
                 ):
        super(LWAVFUNet, self).__init__()

        self.inp_channels = inp_channels
        self.dim = dim

        # Step 1: Embedding and CrossTransfomer Fusion
        
        self.VIS_embed = Encoder(input_channels=inp_channels, output_channels=dim)
        self.IR_embed = Encoder(input_channels=inp_channels, output_channels=dim)
        
        # Step 2: Feature Interaction
        self.DAIM = DAIMBlock(dim=dim, num_heads=8)
        self.fused = ConcatFusion(feature_dim=dim)
        
        # Step 3: Deep Feature Extraction
        self.WAF = LWaveSTLayer(**wavestlayer_configs)
        
        # Step 4: Decoding
        self.decoder = Decoder_LWT(inp_channels=dim, out_channels=inp_channels, **decoder_configs)
        

    def forward(self, modality1, modality2):
        H, W = modality1.shape[2:]
        # Embedding
        enc1 = self.VIS_embed(modality1)
        enc2 = self.IR_embed(modality2)
        # Shallow Crossattn 
        fuse1, fuse2 = self.DAIM(enc1, enc2)
        # Feature Extraction
        fused = self.fused(fuse1, fuse2)
        fused_feature = self.WAF(fused)
        # Deep Fusion

        # Decoder Fusion
        #output = self.decoder(fused_feature, modality1)
        output = self.decoder(fused_feature)
        return output
        
    def get_wavelet_loss(self):
        return self.WAF.get_wavelet_loss() + self.decoder.get_wavelet_loss()

