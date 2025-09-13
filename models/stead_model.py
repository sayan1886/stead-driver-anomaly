import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return attn_out + x

class STEAD(nn.Module):
    def __init__(self, feature_dim=1024, hidden_dim=64, seq_len=16, num_heads=4):
        super().__init__()
        self.temporal_attn = TemporalAttention(dim=feature_dim, num_heads=num_heads)
        self.fc = nn.Linear(feature_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x):
        # x: [B, T, F] -> already pre-extracted features
        x = self.temporal_attn(x)   # [B, T, F]
        x = x.mean(dim=1)           # temporal average pooling
        z = self.fc(x)              # latent embedding
        recon = self.decoder(z)     # reconstructed features
        return recon, z
