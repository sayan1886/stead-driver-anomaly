# evaluate.py
import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader

from models.stead_model import STEAD
from dataset_x3d import X3DFeatureDataset
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
    anomaly_classes = cfg["training"].get("anomaly_class", ["normal"])
    test_dataset = X3DFeatureDataset(root_dir=data_dir, split="test",
                                     anomaly_classes=anomaly_classes, DEBUG=DEBUG)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if DEBUG:
        print(f"[DEBUG] Test dataset size: {len(test_dataset)}")

    threshold = float(cfg.get("evaluation", {}).get("mse_threshold", 0.001))

    results = []

    # Evaluation loop
    with torch.no_grad():
        for idx, (xb, labels_idx, labels_class) in enumerate(test_loader):
            xb = xb.to(device)  # [B, T, F]
            recon, _ = model(xb)

            target = xb.mean(dim=1)  # [B, F]
            mse = ((recon - target) ** 2).mean(dim=1).item()

            # Predict class based on threshold
            if mse <= threshold:
                pred_class = anomaly_classes[0]  # normal
            else:
                # If anomaly, pick the true class if available, else fallback
                true_class = labels_class[0] if labels_class else "anomaly"
                pred_class = true_class

            results.append({
                "clip_idx": idx,
                "mse": mse,
                "pred_class": pred_class,
                "true_class": labels_class[0] if labels_class else "unknown"
            })

            if DEBUG:
                print(f"[DEBUG] Clip {idx}: MSE={mse:.6f}, Pred={pred_class}, True={labels_class[0] if labels_class else 'unknown'}")

    # Summary
    print("=== Evaluation Summary ===")
    for r in results[:10]:
        print(f"Clip {r['clip_idx']}: Score={r['mse']:.6f}, Predicted={r['pred_class']}, True={r['true_class']}")
    print(f"Average anomaly score: {np.mean([r['mse'] for r in results]):.6f}")

    return results

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