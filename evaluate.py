import numpy as np
import argparse
from sklearn.metrics import roc_auc_score

import torch
from torch.utils.data import DataLoader, random_split

from models.stead_model import STEAD
from dataset_x3d import X3DFeatureDataset
import config.config as cfg_loader


# ------------------------------
# Metric functions
# ------------------------------
def compute_psnr(target, recon):
    mse = ((target - recon) ** 2).mean().item()
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 1.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def compute_ssim(target, recon):
    # Placeholder: implement proper SSIM for features if needed
    # For now, return dummy 1.0 for demonstration
    return 1.0


# ------------------------------
# Threshold utility
# ------------------------------
def compute_dynamic_threshold(scores, factor=1.5):
    """
    Compute dynamic threshold as mean + factor * std deviation.
    """
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    return mean_score + factor * std_score


# ------------------------------
# Model utilities
# ------------------------------
def load_model(cfg, checkpoint, device, DEBUG=False):
    feature_dim = int(cfg["model"]["feature_dim"])
    hidden_dim = int(cfg["model"]["hidden_dim"])
    seq_len = int(cfg["model"]["seq_len"])
    num_heads = int(cfg["model"]["num_heads"])

    model = STEAD(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        seq_len=seq_len,
        num_heads=num_heads
    ).to(device)

    if DEBUG:
        print(f"[DEBUG] Loading checkpoint from {checkpoint}")

    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def prepare_datasets(cfg, data_dir, DEBUG=False):
    anomaly_classes = cfg["training"].get("anomaly_class", ["normal"])
    full_dataset = X3DFeatureDataset(
        root_dir=data_dir,
        split="test",
        anomaly_classes=anomaly_classes,
        DEBUG=DEBUG
    )

    if DEBUG:
        print(f"[DEBUG] Test dataset size: {len(full_dataset)} samples")

    val_split = float(cfg.get("evaluation", {}).get("validation_split", 0.0))
    if val_split > 0:
        val_len = int(len(full_dataset) * val_split)
        test_len = len(full_dataset) - val_len
        test_dataset, val_dataset = random_split(full_dataset, [test_len, val_len])
        if DEBUG:
            print(f"[DEBUG] Split test={len(test_dataset)}, validation={len(val_dataset)}")
    else:
        test_dataset, val_dataset = full_dataset, None

    return test_dataset, val_dataset, anomaly_classes


# ------------------------------
# Evaluation helpers
# ------------------------------
def evaluate_clip(model, xb, labels_class, anomaly_classes, threshold, metrics_cfg, device):
    xb = xb.to(device)  # [B, T, F]
    recon, _ = model(xb)
    target = xb.mean(dim=1)  # [B, F]
    mse = ((recon - target) ** 2).mean(dim=1).item()

    # True class
    true_class = labels_class[0] if labels_class else "unknown"

    # Predict class
    if mse <= threshold:
        pred_class = anomaly_classes[0]  # assume index 0 is "normal"
        y_true = 0
    else:
        pred_class = true_class if true_class in anomaly_classes else "unknown"
        # map to its index in anomaly_classes (multi-class)
        y_true = anomaly_classes.index(pred_class) if pred_class in anomaly_classes else len(anomaly_classes)

    # Compute additional metrics
    metric_values = {}
    for m in metrics_cfg:
        if m["type"] == "mse":
            metric_values["mse"] = mse
        elif m["type"] == "psnr":
            metric_values["psnr"] = compute_psnr(target, recon)
        elif m["type"] == "ssim":
            metric_values["ssim"] = compute_ssim(target, recon)

    return mse, pred_class, true_class, y_true, metric_values


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
    model = load_model(cfg, checkpoint, device, DEBUG)

    # ------------------------------
    # Dataset setup with validation split
    # ------------------------------
    test_dataset, val_dataset, anomaly_classes = prepare_datasets(cfg, data_dir, DEBUG)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initial threshold from config
    initial_threshold = float(cfg.get("evaluation", {}).get("mse_threshold", 0.001))
    metrics_cfg = cfg.get("evaluation", {}).get("metrics", [])

    # Collect all MSEs (for dynamic threshold computation)
    all_mse_scores = []
    with torch.no_grad():
        for xb, _, _ in test_loader:
            recon, _ = model(xb.to(device))
            target = xb.to(device).mean(dim=1)
            mse = ((recon - target) ** 2).mean(dim=1).item()
            all_mse_scores.append(mse)

    # Hybrid threshold
    dynamic_threshold = compute_dynamic_threshold(all_mse_scores, factor=3.0)
    threshold = max(initial_threshold, dynamic_threshold)

    if DEBUG:
        print(f"[INFO] Initial threshold={initial_threshold:.6f}, Dynamic threshold={dynamic_threshold:.6f}, Using={threshold:.6f}")

    # ------------------------------
    # Final evaluation loop
    # ------------------------------
    results, y_true_multi, y_score = [], [], []
    with torch.no_grad():
        for idx, (xb, labels_idx, labels_class) in enumerate(test_loader):
            mse, pred_class, true_class, y_true, metric_values = evaluate_clip(
                model, xb, labels_class, anomaly_classes, threshold, metrics_cfg, device
            )
            y_true_multi.append(y_true)
            y_score.append(mse)
            results.append({
                "clip_idx": idx,
                "mse": mse,
                "pred_class": pred_class,
                "true_class": true_class,
                "metrics": metric_values
            })
            if DEBUG:
                print(f"[DEBUG] Clip {idx}: MSE={mse:.6f}, Pred={pred_class}, True={true_class}, Metrics={metric_values}")

    # ------------------------------
    # AUC calculation (hybrid)
    # ------------------------------
    y_true_unique = np.unique(y_true_multi)
    if len(y_true_unique) == 1:
        auc_score = None
        if DEBUG:
            print("[WARNING] Only one class present in y_true, cannot compute ROC AUC.")
    else:
        try:
            if len(y_true_unique) == 2:
                # Binary case
                auc_score = roc_auc_score(y_true_multi, y_score)
            else:
                # Multi-class case
                auc_score = roc_auc_score(y_true_multi, y_score, multi_class="ovr")
        except Exception as e:
            auc_score = None
            if DEBUG:
                print(f"[WARNING] AUC computation failed: {e}")

    # ------------------------------
    # Summary
    # ------------------------------
    print("=== Evaluation Summary ===")
    for r in results[:10]:
        metrics_str = ", ".join([f"{k}={v:.6f}" for k, v in r["metrics"].items()])
        print(f"Clip {r['clip_idx']}: Score={r['mse']:.6f}, Pred={r['pred_class']}, True={r['true_class']}, {metrics_str}")

    avg_mse = np.mean([r['mse'] for r in results])
    print(f"Average anomaly score (MSE): {avg_mse:.6f}")
    print(f"AUC: {auc_score:.6f}" if auc_score is not None else "AUC: undefined (single class in y_true)")

    return results, auc_score, val_dataset

# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate STEAD model on X3D feature dataset.",
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