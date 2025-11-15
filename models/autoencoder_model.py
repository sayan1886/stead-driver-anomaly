# models/autoencoder_model.py
import torch
import torch.nn as nn

class FeatureAutoencoder(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512):
        super().__init__()
        # Simple 2-layer MLP
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, T, F]
        x_avg = x.mean(dim=1)          # [B, F]
        z = self.encoder(x_avg)        # [B, hidden_dim]
        out = self.decoder(z)          # [B, F]
        # ensure batch sizes match
        if out.shape[0] != x_avg.shape[0]:
            out = out[: x_avg.shape[0], :]
        return out