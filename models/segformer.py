"""Dependency-free SegFormer-B0 for dense satellite-image prediction."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep_prob) * random_tensor.floor()


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, kernel_size, stride):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=kernel_size, stride=stride,
            padding=kernel_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        height, width = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), height, width


class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio=1, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, height, width):
        batch, tokens, channels = x.shape
        q = self.q(x).reshape(batch, tokens, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        kv_input = x
        if self.sr_ratio > 1:
            kv_input = x.transpose(1, 2).reshape(batch, channels, height, width)
            kv_input = self.sr(kv_input).flatten(2).transpose(1, 2)
            kv_input = self.norm(kv_input)
        kv = self.kv(kv_input).reshape(
            batch, -1, 2, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        key, value = kv[0], kv[1]
        attention = (q @ key.transpose(-2, -1)) * self.scale
        attention = self.attn_drop(attention.softmax(dim=-1))
        x = (attention @ value).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(x))


class MixFFN(nn.Module):
    def __init__(self, dim, mlp_ratio=4, drop=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim
        )
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, height, width):
        batch = x.shape[0]
        x = self.fc1(x)
        x = x.transpose(1, 2).reshape(batch, -1, height, width)
        x = self.dwconv(x).flatten(2).transpose(1, 2)
        x = self.drop(self.act(x))
        return self.drop(self.fc2(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, sr_ratio)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim)

    def forward(self, x, height, width):
        x = x + self.drop_path(self.attn(self.norm1(x), height, width))
        x = x + self.drop_path(self.mlp(self.norm2(x), height, width))
        return x


class MixTransformer(nn.Module):
    """MiT-B0 hierarchical transformer encoder."""

    def __init__(self, in_channels=7, drop_path_rate=0.1):
        super().__init__()
        embed_dims = (32, 64, 160, 256)
        num_heads = (1, 2, 5, 8)
        depths = (2, 2, 2, 2)
        sr_ratios = (8, 4, 2, 1)
        patch_settings = ((7, 4), (3, 2), (3, 2), (3, 2))
        input_dims = (in_channels,) + embed_dims[:-1]
        self.patch_embeds = nn.ModuleList([
            OverlapPatchEmbed(inp, out, kernel, stride)
            for inp, out, (kernel, stride) in zip(input_dims, embed_dims, patch_settings)
        ])
        drop_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        offset = 0
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        for dim, heads, depth, sr_ratio in zip(embed_dims, num_heads, depths, sr_ratios):
            self.blocks.append(nn.ModuleList([
                TransformerBlock(dim, heads, sr_ratio, drop_rates[offset + index])
                for index in range(depth)
            ]))
            self.norms.append(nn.LayerNorm(dim))
            offset += depth

    def forward(self, x):
        features = []
        for patch_embed, blocks, norm in zip(self.patch_embeds, self.blocks, self.norms):
            x, height, width = patch_embed(x)
            for block in blocks:
                x = block(x, height, width)
            x = norm(x)
            x = x.transpose(1, 2).reshape(x.shape[0], -1, height, width)
            features.append(x)
        return features


class SegFormerHead(nn.Module):
    def __init__(self, num_classes=1, channels=256, dropout=0.1):
        super().__init__()
        in_channels = (32, 64, 160, 256)
        self.projections = nn.ModuleList([
            nn.Conv2d(in_dim, channels, kernel_size=1) for in_dim in in_channels
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, features):
        output_size = features[0].shape[-2:]
        projected = []
        for feature, projection in zip(features, self.projections):
            feature = projection(feature)
            if feature.shape[-2:] != output_size:
                feature = F.interpolate(
                    feature, size=output_size, mode="bilinear", align_corners=False
                )
            projected.append(feature)
        return self.classifier(self.dropout(self.fuse(torch.cat(projected, dim=1))))


class SegFormer(nn.Module):
    """SegFormer-B0 with configurable input and output channel counts."""

    def __init__(self, num_classes=1, in_channels=7,
                 backbone="segformer_b0", pretrained=False):
        super().__init__()
        if backbone not in ("segformer_b0", "mit_b0", "b0"):
            raise ValueError("Only the SegFormer-B0/MiT-B0 backbone is supported")
        if pretrained:
            raise ValueError(
                "Pretrained weights are not bundled; set pretrained=False or load a checkpoint"
            )
        self.encoder = MixTransformer(in_channels=in_channels)
        self.decode_head = SegFormerHead(num_classes=num_classes)
        self.backbone = backbone
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        input_size = x.shape[-2:]
        logits = self.decode_head(self.encoder(x))
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

    def freeze_backbone(self):
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self):
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True
