# evaluate.py
import torch
from torch.utils.data import DataLoader
from models.stead_model import STEAD
from dataset_x3d import X3DFeatureDataset
import numpy as np
import argparse

import config.config as cfg_loader

# ------------------------------
# Evaluation function
# ------------------------------
def evaluate_stead(cfg, data_dir, checkpoint):
    DEBUG = cfg.get("debug", False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if DEBUG:
        print(f"[DEBUG] Using device: {device}")

    # Model setup
    feature_dim = int(cfg["model"]["feature_dim"])
    hidden_dim = int(cfg["model"]["hidden_dim"])
    seq_len = int(cfg["model"]["seq_len"])
    num_heads = int(cfg["model"]["num_heads"])

    model = STEAD(feature_dim=feature_dim,
                  hidden_dim=hidden_dim,
                  seq_len=seq_len,
                  num_heads=num_heads).to(device)

    if DEBUG:
        print(f"[DEBUG] Loading checkpoint from {checkpoint}")

    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Dataset setup
    test_dataset = X3DFeatureDataset(root_dir=data_dir, split="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if DEBUG:
        print(f"[DEBUG] Test dataset size: {len(test_dataset)}")

    # Evaluation
    scores = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)  # [B, T, F]
            recon, _ = model(xb)

            # Target: temporal-averaged feature per clip
            target = xb.mean(dim=1)  # [B, F]
            mse = ((recon - target) ** 2).mean(dim=1)
            scores.append(mse.item())

    print("Anomaly scores (first 10):", scores[:10])
    print(f"Average anomaly score: {np.mean(scores):.6f}")


# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate STEAD model on X3D feature dataset.\n"
                    "Usage:\n"
                    "  python evaluate.py --config config/config.yaml --data_dir X3D_Videos",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to X3D feature dataset")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/stead_driver.pt", help="Path to model checkpoint")
    args = parser.parse_args()

    cfg = cfg_loader.load_config(args.config)
    evaluate_stead(cfg, args.data_dir, args.checkpoint)


if __name__ == "__main__":
    main()
