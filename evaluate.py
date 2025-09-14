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

    # ------------------------------
    # Model setup
    # ------------------------------
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

    # ------------------------------
    # Dataset setup
    # ------------------------------
    anomaly_classes = cfg["training"].get("anomaly_class", ["normal"])
    test_dataset = X3DFeatureDataset(root_dir=data_dir, split="test",
                                     anomaly_classes=anomaly_classes, DEBUG=DEBUG)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if DEBUG:
        print(f"[DEBUG] Test dataset size: {len(test_dataset)}")

    threshold = float(cfg.get("evaluation", {}).get("mse_threshold", 0.001))
    results = []

    # ------------------------------
    # Evaluation loop
    # ------------------------------
    with torch.no_grad():
        for idx, (xb, labels_idx, labels_class) in enumerate(test_loader):
            xb = xb.to(device)  # [B, T, F]
            recon, _ = model(xb)

            target = xb.mean(dim=1)  # [B, F]
            mse = ((recon - target) ** 2).mean(dim=1).item()

            # True class fallback
            if labels_class is None or len(labels_class) == 0:
                true_class = "unknown"
            else:
                true_class = str(labels_class[0])

            # Predict class
            if mse <= threshold:
                pred_class = anomaly_classes[0]  # normal
            else:
                pred_class = f"anomaly-{true_class}" if true_class != "normal" else "anomaly-unknown"

            results.append({
                "clip_idx": idx,
                "mse": mse,
                "pred_class": pred_class,
                "true_class": true_class
            })

            if DEBUG:
                print(f"[DEBUG] Clip {idx}: MSE={mse:.6f}, Pred={pred_class}, True={true_class}")


    # ------------------------------
    # Summary
    # ------------------------------
    print("=== Evaluation Summary ===")
    for r in results[:10]:
        clip_idx = r['clip_idx']
        mse = r['mse']
        pred = r['pred_class']
        true = r['true_class'] if r['true_class'] != "" else "unknown"
        print(f"Clip {clip_idx}: Score={mse:.6f}, Predicted={pred}, True={true}")

    avg_mse = np.mean([r['mse'] for r in results])
    print(f"Average anomaly score: {avg_mse:.6f}")

    # Optional: show a breakdown by predicted subcategory
    from collections import Counter
    pred_counter = Counter(r['pred_class'] for r in results)
    print("\nPredicted class distribution:")
    for cls, cnt in pred_counter.items():
        print(f"  {cls}: {cnt}")

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