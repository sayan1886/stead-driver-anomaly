# models/stead_model.py
import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        """
        Multi-head temporal attention module.

        Args:
            dim (int): Feature dimension
            num_heads (int): Number of attention heads
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    # def forward(self, x):
    #     """
    #     Args:
    #         x: [B, T, F] tensor of pre-extracted features
    #     Returns:
    #         x + attention output: [B, T, F]
    #     """
    #     attn_out, _ = self.attn(x, x, x)
    #     return attn_out + x

    def forward(self, x):
        """
        Args:
            x: Ideally [B, T, F], but may come in with extra dims.
        Returns:
            x + attention output: [B, T, F]
        """
        # Make sure x is at least 3D: [B, T, F]
        if x.dim() == 2:
            # [B, F] -> [B, 1, F]
            x = x.unsqueeze(1)
        elif x.dim() > 3:
            # e.g. [B, T, H, W, C] or [B, T, ...]
            B, T = x.shape[0], x.shape[1]
            x = x.contiguous().view(B, T, -1)  # flatten all remaining dims into F

        if x.dim() != 3:
            raise RuntimeError(f"TemporalAttention expected 3D input, got {x.shape}")

        attn_out, _ = self.attn(x, x, x)
        return attn_out + x


class STEAD(nn.Module):
    def __init__(self, feature_dim=1024, hidden_dim=64, seq_len=16, num_heads=4):
        """
        STEAD autoencoder model.

        Args:
            feature_dim (int): Dimensionality of X3D feature vector
            hidden_dim (int): Bottleneck dimension for latent embedding
            seq_len (int): Temporal sequence length (clip length)
            num_heads (int): Number of attention heads
        """
        super().__init__()
        self.seq_len = seq_len
        self.temporal_attn = TemporalAttention(dim=feature_dim, num_heads=num_heads)
        self.fc = nn.Linear(feature_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x):
        """
        Args:
            x: [B, T, F] pre-extracted features
        Returns:
            recon: [B, F] reconstructed features
            z: [B, hidden_dim] latent embedding
        """
        # Temporal attention
        x = self.temporal_attn(x)   # [B, T, F]

        # Temporal average pooling
        x = x.mean(dim=1)           # [B, F]

        # Encode
        z = self.fc(x)              # [B, hidden_dim]

        # Decode
        recon = self.decoder(z)     # [B, F]

        return recon, z

# Optional: factory method to instantiate from YAML config
def build_stead_from_cfg(cfg):
    return STEAD(
        feature_dim=cfg["model"]["feature_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        seq_len=cfg["model"]["seq_len"],
        num_heads=cfg["model"]["num_heads"]
    )