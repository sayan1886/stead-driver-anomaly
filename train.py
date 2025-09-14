# train.py
import os
import torch
from torch.utils.data import DataLoader
from models.stead_model import STEAD
from dataset_x3d import X3DFeatureDataset
from tqdm import tqdm
import argparse

import config.config as cfg_loader

# ------------------------------
# Training function
# ------------------------------
def train_stead(cfg):
    # ------------------------------
    # Debug & device
    # ------------------------------
    DEBUG = cfg.get("debug", False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if DEBUG:
        print(f"[DEBUG] Using device: {device}")

    # ------------------------------
    # Dataset setup
    # ------------------------------
    data_root = cfg["dataset"]["root_dir"]
    batch_size = int(cfg["dataset"].get("batch_size", 16))

    train_dataset = X3DFeatureDataset(root_dir=data_root, split="train")
    test_dataset = X3DFeatureDataset(root_dir=data_root, split="test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if DEBUG:
        print(f"[DEBUG] Train dataset size: {len(train_dataset)}")
        print(f"[DEBUG] Test dataset size: {len(test_dataset)}")

    # ------------------------------
    # Model setup
    # ------------------------------
    feature_dim = int(cfg["model"]["feature_dim"])
    hidden_dim = int(cfg["model"]["hidden_dim"])
    seq_len = int(cfg["model"]["seq_len"])
    num_heads = int(cfg["model"]["num_heads"])

    model = STEAD(feature_dim=feature_dim, hidden_dim=hidden_dim,
                  seq_len=seq_len, num_heads=num_heads).to(device)

    if DEBUG:
        print(f"[DEBUG] Model initialized: {model}")

    # ------------------------------
    # Optimizer & loss
    # ------------------------------
    opt_cfg = cfg.get("optimizer", {})
    learning_rate = float(cfg["training"]["learning_rate"])
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    loss_fn = torch.nn.MSELoss()

    # ------------------------------
    # Training loop
    # ------------------------------
    epochs = int(cfg["training"]["epochs"])

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            xb = xb.to(device)  # [B, T, F]

            # Forward
            recon, _ = model(xb)

            # Target: temporal-averaged feature per clip
            target = xb.mean(dim=1)  # [B, F]
            loss = loss_fn(recon, target)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} loss: {avg_loss:.6f}")

    # ------------------------------
    # Save model
    # ------------------------------
    os.makedirs("checkpoints", exist_ok=True)
    model_path = os.path.join("checkpoints", "stead_driver.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Training finished. Model saved to {model_path}")


# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train STEAD model on X3D feature dataset.\n"
                    "Usage:\n"
                    "  python train.py --config config/config.yaml",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g., config/config.yaml)"
    )
    args = parser.parse_args()

    cfg = cfg_loader.load_config(args.config)
    train_stead(cfg)


if __name__ == "__main__":
    main()