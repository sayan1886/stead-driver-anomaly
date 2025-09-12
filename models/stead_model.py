
import torch
import torch.nn as nn
from einops import rearrange

class TemporalAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return attn_out + x

class STEAD(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=64, seq_len=16, num_heads=4):
        super().__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.flatten = nn.Flatten(start_dim=2)
        self.temporal_attn = TemporalAttention(dim=64*seq_len, num_heads=num_heads)
        self.fc = nn.Linear(64*seq_len, 64)
        self.decoder = nn.Linear(64, 64*seq_len)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.conv2d(x)
        x = self.flatten(x)
        x = rearrange(x, '(b t) f -> b t f', b=B, t=T)
        x = self.temporal_attn(x)
        x = x.mean(dim=1)
        z = self.fc(x)
        recon = self.decoder(z)
        return recon, z
