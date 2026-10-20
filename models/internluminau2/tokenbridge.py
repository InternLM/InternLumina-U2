import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torch_npu
except ImportError:
    torch_npu = None
from timm.layers import Mlp
from timm.models.vision_transformer import Block

def build_causal_mask(seq_length):
    """
    Create a causal attention mask for autoregressive prediction.
    
    Args:
        seq_length (int): Length of the sequence
        
    Returns:
        torch.Tensor: Causal attention mask with -inf in upper triangle
    """
    mask = torch.empty(seq_length, seq_length)
    mask.fill_(float("-inf"))
    mask.triu_(1) 
    return mask

def modulate(x, shift, scale):
    """
    Apply AdaLN modulation with shift and scale parameters.
    
    Args:
        x (torch.Tensor): Input tensor
        shift (torch.Tensor): Shift parameter
        scale (torch.Tensor): Scale parameter
        
    Returns:
        torch.Tensor: Modulated tensor
    """
    return x * (1 + scale) + shift

class CausalAttention(nn.Module):
    """
    Causal self-attention module for autoregressive prediction with KV-caching support.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, attn_drop=0., 
                 proj_drop=0., norm_layer=nn.LayerNorm):
        """
        Initialize the causal attention module.
        
        Args:
            dim (int): Input dimension
            num_heads (int): Number of attention heads
            qkv_bias (bool): Whether to use bias in QKV projection
            qk_norm (bool): Whether to apply normalization to Q and K
            attn_drop (float): Dropout rate for attention weights
            proj_drop (float): Dropout rate for projection output
            norm_layer: Normalization layer class
        """
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_cache = False
        self.k_cache = None
        self.v_cache = None

    def reset_cache(self):
        """Reset the key and value caches."""
        self.k_cache = None
        self.v_cache = None

    def forward(self, x, attn_mask=None):
        """
        Forward pass for causal attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, N, C]
            attn_mask (torch.Tensor): Optional attention mask
            
        Returns:
            torch.Tensor: Output tensor of shape [B, N, C]
        """
        if attn_mask is None:
            attn_mask = build_causal_mask(x.size(1)).to(x.device)
            
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.use_cache:
            if self.k_cache is None and self.v_cache is None:
                self.k_cache = k
                self.v_cache = v
            else:
                self.k_cache = torch.cat([self.k_cache, k], dim=2)
                self.v_cache = torch.cat([self.v_cache, v], dim=2)
                k, v = self.k_cache, self.v_cache

        if q.device.type == "npu" and attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                atten_mask_npu = torch.logical_not(attn_mask.bool())
            else:
                atten_mask_npu = attn_mask.bool()
            head_num = q.shape[1]
            dropout_p = self.attn_drop.p if self.training else 0.
            x = torch_npu.npu_fusion_attention(
                q, k, v,
                head_num,
                input_layout="BNSD",
                pse=None,
                atten_mask=atten_mask_npu,
                scale=1.0 / math.sqrt(q.shape[-1]),
                pre_tockens=2147483647,
                next_tockens=2147483647,
                keep_prob=1 - dropout_p,
            )[0]
        else:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CausalBlock(nn.Module):
    """
    Transformer block with causal attention.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_norm=False,
                 proj_drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """
        Initialize the causal block.
        
        Args:
            dim (int): Input dimension
            num_heads (int): Number of attention heads
            mlp_ratio (float): Ratio for MLP hidden dimension
            qkv_bias (bool): Whether to use bias in QKV projection
            qk_norm (bool): Whether to apply normalization to Q and K
            proj_drop (float): Dropout rate for projections
            attn_drop (float): Dropout rate for attention weights
            act_layer: Activation layer class
            norm_layer: Normalization layer class
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CausalAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

    def forward(self, x, attn_mask=None, c=None):
        """
        Forward pass for the causal block.
        
        Args:
            x (torch.Tensor): Input tensor
            attn_mask (torch.Tensor): Optional attention mask
            c (torch.Tensor): Conditioning tensor for AdaLN modulation
            
        Returns:
            torch.Tensor: Output tensor
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    Final normalization layer.
    """
    def __init__(self, dim, norm_layer):
        """
        Initialize the final layer.
        
        Args:
            dim (int): Input dimension
            norm_layer: Normalization layer class
        """
        super().__init__()
        self.norm_final = norm_layer(dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2*dim)
        )
    
    def forward(self, x, c):
        """
        Forward pass for the final layer.
        
        Args:
            x (torch.Tensor): Input tensor
            c (torch.Tensor): Conditioning tensor for AdaLN modulation
            
        Returns:
            torch.Tensor: Output tensor
        """
        scale, shift = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return x
