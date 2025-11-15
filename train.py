# train.py
import os
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from dataset_x3d import X3DFeatureDataset

from models.stead_model import STEAD
from models.autoencoder_model import FeatureAutoencoder
import config.config as cfg_loader


# ------------------------------
# Model factory
# ------------------------------
def build_model(cfg, device, DEBUG=False):
    model_type = cfg["model"]["type"].lower()

    if model_type == "stead":
        stead_cfg = cfg["model"]["stead"]
        model = STEAD(
            feature_dim=int(stead_cfg["feature_dim"]),
            hidden_dim=int(stead_cfg["hidden_dim"]),
            seq_len=int(stead_cfg["seq_len"]),
            num_heads=int(stead_cfg["num_heads"])
        )
    elif model_type == "autoencoder":
        ae_cfg = cfg["model"]["autoencoder"]
        model = FeatureAutoencoder(
            input_dim=int(ae_cfg["input_dim"]),
            hidden_dim=int(ae_cfg["hidden_dim"])
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if DEBUG:
        print(f"[DEBUG] Initialized model: {model}")

    return model.to(device)


# ------------------------------
# Training function
# ------------------------------
def train(cfg):
    DEBUG = cfg.get("debug", False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if DEBUG:
        print(f"[DEBUG] Using device: {device}")

    # ------------------------------
    # Dataset setup
    # ------------------------------
    data_root = cfg["dataset"]["root_dir"]
    batch_size = int(cfg["dataset"].get("batch_size", 16))
    anomaly_classes = cfg["training"].get("anomaly_class", ["normal", "anomaly"])

    train_dataset = X3DFeatureDataset(
        root_dir=data_root, split="train",
        anomaly_classes=anomaly_classes, DEBUG=DEBUG
    )
    test_dataset = X3DFeatureDataset(
        root_dir=data_root, split="test",
        anomaly_classes=anomaly_classes, DEBUG=DEBUG
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if DEBUG:
        print(f"[DEBUG] Train dataset size: {len(train_dataset)}")
        print(f"[DEBUG] Test dataset size: {len(test_dataset)}")

    # ------------------------------
    # Model setup
    # ------------------------------
    model = build_model(cfg, device, DEBUG)

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

        for xb, labels_idx, labels_class in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            xb = xb.to(device)

            # Forward depending on model type
            if cfg["model"]["type"].lower() == "stead":
                recon, _ = model(xb)
                target = xb.mean(dim=1)  # temporal-averaged features
            else:  # autoencoder
                recon = model(xb)                 # [B, F]
                target = xb.mean(dim=1)           # [B, F], ensure same B as recon
                if recon.shape != target.shape:
                    target = target[: recon.shape[0], :]  # trim if last batch smaller

            loss = loss_fn(recon, target)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if DEBUG:
                print(f"[DEBUG] Batch labels (numeric): {labels_idx}")
                print(f"[DEBUG] Batch labels (class): {labels_class}")
                print(f"[DEBUG] Batch loss={loss.item():.6f}, Labels={labels_class}")

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} loss: {avg_loss:.6f}")

    # ------------------------------
    # Save model
    # ------------------------------
    os.makedirs("checkpoints", exist_ok=True)
    model_path = os.path.join("checkpoints", f"{cfg['model']['type']}_driver.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Training finished. Model saved to {model_path}")


# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train model (STEAD or AutoEncoder) on X3D feature dataset.\n"
                    "Usage:\n"
                    "  python train.py --config config/config.yaml",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = cfg_loader.load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()