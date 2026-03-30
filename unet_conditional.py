"""
Conditional U-Net Architecture for Diffusion Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)


class LabelEmbedding(nn.Module):
    def __init__(self, label_dim, emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(label_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, labels):
        return self.mlp(labels)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.time_emb = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(x)
        h = h + self.time_emb(emb)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        head_dim = C // self.num_heads
        q = q.view(B, self.num_heads, head_dim, H*W).transpose(2, 3)
        k = k.view(B, self.num_heads, head_dim, H*W).transpose(2, 3)
        v = v.view(B, self.num_heads, head_dim, H*W).transpose(2, 3)

        h = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        h = h.transpose(2, 3).reshape(B, C, H, W)
        return x + self.proj(h)


class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, label_dim=2,
                 base_channels=64, channel_multipliers=(1,2,4,8),
                 attention_levels=(2,3), dropout=0.1, time_emb_dim=256, label_emb_dim=256):
        super().__init__()
        self.label_dim = label_dim

        self.time_embedding = TimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(nn.Linear(time_emb_dim, time_emb_dim*4), nn.SiLU(), nn.Linear(time_emb_dim*4, time_emb_dim))

        self.label_embedding = LabelEmbedding(label_dim, label_emb_dim)
        self.combined_emb_dim = time_emb_dim + label_emb_dim
        self.combined_mlp = nn.Sequential(nn.Linear(self.combined_emb_dim, time_emb_dim*4), nn.SiLU(), nn.Linear(time_emb_dim*4, time_emb_dim))

        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels

        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            for _ in range(2):
                self.down_blocks.append(ResidualBlock(now_channels, out_ch, time_emb_dim, dropout))
                if i in attention_levels:
                    self.down_blocks.append(AttentionBlock(out_ch))
                now_channels = out_ch
                channels.append(now_channels)
            if i != len(channel_multipliers) - 1:
                self.down_blocks.append(nn.Conv2d(now_channels, now_channels, kernel_size=3, stride=2, padding=1))
                channels.append(now_channels)

        self.middle = nn.ModuleList([
            ResidualBlock(now_channels, now_channels, time_emb_dim, dropout),
            AttentionBlock(now_channels),
            ResidualBlock(now_channels, now_channels, time_emb_dim, dropout)
        ])

        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = base_channels * mult
            for _ in range(3):
                self.up_blocks.append(ResidualBlock(now_channels + channels.pop(), out_ch, time_emb_dim, dropout))
                if i in attention_levels:
                    self.up_blocks.append(AttentionBlock(out_ch))
                now_channels = out_ch
            if i != 0:
                self.up_blocks.append(nn.ConvTranspose2d(now_channels, now_channels, kernel_size=4, stride=2, padding=1))

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, now_channels),
            nn.SiLU(),
            nn.Conv2d(now_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t, labels=None):
        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb)

        if labels is not None:
            label_emb = self.label_embedding(labels)
            combined = torch.cat([t_emb, label_emb], dim=-1)
            emb = self.combined_mlp(combined)
        else:
            emb = t_emb

        h = self.conv_in(x)
        skips = [h]

        for module in self.down_blocks:
            if isinstance(module, ResidualBlock):
                h = module(h, emb)
                skips.append(h)
            elif isinstance(module, AttentionBlock):
                h = module(h)
            else:
                h = module(h)
                skips.append(h)

        for module in self.middle:
            if isinstance(module, ResidualBlock):
                h = module(h, emb)
            else:
                h = module(h)

        for module in self.up_blocks:
            if isinstance(module, ResidualBlock):
                h = torch.cat([h, skips.pop()], dim=1)
                h = module(h, emb)
            elif isinstance(module, AttentionBlock):
                h = module(h)
            else:
                h = module(h)

        return self.conv_out(h)
