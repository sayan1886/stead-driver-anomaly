import torch
from torch.utils.data import DataLoader
from models.stead_model import STEAD
from dataset_x3d import X3DFeatureDataset  # not DashcamDataset if you're using pre-extracted X3D features
import numpy as np
import argparse

# ------------------------------
# Args
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True, help="Path to X3D features")
parser.add_argument("--checkpoint", type=str, default="checkpoints/stead_driver.pt")
args = parser.parse_args()

# ------------------------------
# Device & Model
# ------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

feature_dim = 2048  # MUST match what was used in training (x3d_m)
seq_len = 16
model = STEAD(feature_dim=feature_dim, seq_len=seq_len).to(device)

checkpoint_path = args.checkpoint
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# ------------------------------
# Dataset
# ------------------------------
test_dataset = X3DFeatureDataset(root_dir=args.data_dir, split="test")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ------------------------------
# Evaluation
# ------------------------------
scores = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)  # [B, T, F]
        recon, _ = model(xb)

        # target = mean feature per clip
        target = xb.mean(dim=1)  # [B, F]
        mse = ((recon - target) ** 2).mean(dim=1)
        scores.append(mse.item())

print("Anomaly scores (first 10):", scores[:10])