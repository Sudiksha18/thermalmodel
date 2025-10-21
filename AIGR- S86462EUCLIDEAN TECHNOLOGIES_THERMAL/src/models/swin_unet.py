#!/usr/bin/env python3
"""
Swin Transformer U-Net for thermal anomaly detection.
High-accuracy transformer-based segmentation model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math

try:
    from transformers import SwinModel, SwinConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

import logging
logger = logging.getLogger(__name__)


@dataclass
class SwinUNetConfig:
    """Configuration for Swin U-Net model."""
    
    # Input configuration
    input_channels: int = 1
    num_classes: int = 2
    image_size: Tuple[int, int] = (512, 512)
    
    # Swin Transformer configuration
    backbone: str = "swin_tiny"  # swin_tiny, swin_small, swin_base
    pretrained: bool = True
    patch_size: int = 4
    window_size: int = 7
    embed_dim: int = 96
    depths: List[int] = (2, 2, 6, 2)
    num_heads: List[int] = (3, 6, 12, 24)
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    
    # Decoder configuration
    decoder_channels: List[int] = (512, 256, 128, 64)
    skip_connection: bool = True
    deep_supervision: bool = False
    
    # Training configuration
    freeze_encoder: bool = False
    use_checkpointing: bool = True


class PatchEmbedding(nn.Module):
    """
    Patch embedding layer for thermal images.
    """
    
    def __init__(self, 
                 in_channels: int = 1,
                 embed_dim: int = 96,
                 patch_size: int = 4,
                 norm_layer: Optional[nn.Module] = None):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Pad if needed
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        # Project patches
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = self.norm(x)
        
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer block with window-based self-attention.
    """
    
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 window_size: int = 7,
                 shift_size: int = 0,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop: float = 0.0,
                 attn_drop: float = 0.0,
                 drop_path: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
    
    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # Window attention
        attn_windows = self.attn(x_windows)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x.view(B, H * W, C)
        
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention.
    """
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class Mlp(nn.Module):
    """Multi-layer perceptron."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


def window_partition(x, window_size):
    """Partition into non-overlapping windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reverse window partition."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinEncoder(nn.Module):
    """Swin Transformer encoder."""
    
    def __init__(self, config: SwinUNetConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            in_channels=config.input_channels,
            embed_dim=config.embed_dim,
            patch_size=config.patch_size,
            norm_layer=nn.LayerNorm
        )
        
        # Build layers
        self.layers = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        
        for i_layer in range(len(config.depths)):
            layer = SwinStage(
                dim=int(config.embed_dim * 2 ** i_layer),
                depth=config.depths[i_layer],
                num_heads=config.num_heads[i_layer],
                window_size=config.window_size,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=True,
                drop=config.drop_rate,
                attn_drop=config.attn_drop_rate,
                drop_path=dpr[sum(config.depths[:i_layer]):sum(config.depths[:i_layer + 1])],
                downsample=i_layer < len(config.depths) - 1
            )
            self.layers.append(layer)
        
        self.num_features = int(config.embed_dim * 2 ** (len(config.depths) - 1))
        self.norm = nn.LayerNorm(self.num_features)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Calculate dimensions after patch embedding
        H_patches = (H + self.config.patch_size - 1) // self.config.patch_size
        W_patches = (W + self.config.patch_size - 1) // self.config.patch_size
        
        features = []
        for layer in self.layers:
            x, H_patches, W_patches = layer(x, H_patches, W_patches)
            features.append(x.view(B, H_patches, W_patches, -1).permute(0, 3, 1, 2))
        
        return features


class SwinStage(nn.Module):
    """Swin Transformer stage."""
    
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4.0, qkv_bias=True,
                 drop=0.0, attn_drop=0.0, drop_path=0.0, downsample=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        
        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path
            )
            for i in range(depth)
        ])
        
        # Downsample layer
        if downsample:
            self.downsample = PatchMerging(dim)
        else:
            self.downsample = None
    
    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        
        return x, H, W


class PatchMerging(nn.Module):
    """Patch merging layer."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
    
    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        x = x.view(B, H, W, C)
        
        # Padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        
        x = self.norm(x)
        x = self.reduction(x)
        
        H_new = (H + 1) // 2
        W_new = (W + 1) // 2
        
        return x, H_new, W_new


class DecoderBlock(nn.Module):
    """U-Net decoder block with skip connections."""
    
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, skip=None):
        # Upsample
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Concatenate with skip connection
        if skip is not None:
            # Ensure dimensions match
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        
        # Convolutions
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        return x


class SwinUNet(nn.Module):
    """
    Swin Transformer U-Net for thermal anomaly detection.
    """
    
    def __init__(self, config: SwinUNetConfig):
        super().__init__()
        self.config = config
        
        # Encoder
        self.encoder = SwinEncoder(config)
        
        # Decoder
        encoder_channels = [int(config.embed_dim * 2 ** i) for i in range(len(config.depths))]
        decoder_channels = config.decoder_channels
        
        self.decoder_blocks = nn.ModuleList()
        
        # Build decoder blocks
        for i in range(len(decoder_channels)):
            if i == 0:
                # First decoder block (from bottleneck)
                in_channels = encoder_channels[-1]
                skip_channels = encoder_channels[-2] if len(encoder_channels) > 1 else 0
            else:
                in_channels = decoder_channels[i-1]
                skip_idx = len(encoder_channels) - 2 - i
                skip_channels = encoder_channels[skip_idx] if skip_idx >= 0 else 0
            
            out_channels = decoder_channels[i]
            
            self.decoder_blocks.append(
                DecoderBlock(in_channels, skip_channels, out_channels)
            )
        
        # Final classification head
        self.final_conv = nn.Conv2d(decoder_channels[-1], config.num_classes, 1)
        
        # Deep supervision heads (optional)
        if config.deep_supervision:
            self.deep_supervision_heads = nn.ModuleList([
                nn.Conv2d(ch, config.num_classes, 1) for ch in decoder_channels
            ])
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Encoder
        encoder_features = self.encoder(x)
        
        # Decoder
        decoder_outputs = []
        decoder_x = encoder_features[-1]
        
        for i, decoder_block in enumerate(self.decoder_blocks):
            # Skip connection
            skip_idx = len(encoder_features) - 2 - i
            skip = encoder_features[skip_idx] if skip_idx >= 0 else None
            
            decoder_x = decoder_block(decoder_x, skip)
            decoder_outputs.append(decoder_x)
        
        # Final output
        output = self.final_conv(decoder_x)
        
        # Resize to input size
        if output.shape[2:] != (H, W):
            output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
        
        result = {'logits': output}
        
        # Deep supervision
        if self.config.deep_supervision and self.training:
            deep_outputs = []
            for i, head in enumerate(self.deep_supervision_heads):
                deep_out = head(decoder_outputs[i])
                deep_out = F.interpolate(deep_out, size=(H, W), mode='bilinear', align_corners=False)
                deep_outputs.append(deep_out)
            result['deep_supervision'] = deep_outputs
        
        return result


def create_swin_unet(config_dict: Dict) -> SwinUNet:
    """
    Create Swin U-Net model from configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        SwinUNet model
    """
    model_config = config_dict.get('model', {})
    
    config = SwinUNetConfig(
        input_channels=model_config.get('input_channels', 1),
        num_classes=model_config.get('num_classes', 2),
        image_size=tuple(config_dict.get('dataset', {}).get('image_size', [512, 512])),
        backbone=model_config.get('backbone', 'swin_tiny'),
        pretrained=model_config.get('pretrained', True),
        drop_rate=model_config.get('dropout', 0.1),
        deep_supervision=model_config.get('deep_supervision', False)
    )
    
    return SwinUNet(config)


if __name__ == "__main__":
    # Test model
    config = SwinUNetConfig(
        input_channels=1,
        num_classes=2,
        image_size=(512, 512)
    )
    
    model = SwinUNet(config)
    x = torch.randn(2, 1, 512, 512)
    
    with torch.no_grad():
        output = model(x)
        print(f"Output shape: {output['logits'].shape}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("Swin U-Net test completed!")