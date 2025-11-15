import os
import argparse
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score

import torch
from torch.utils.data import DataLoader, random_split

from models import build_model
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
    return 1.0


# ------------------------------
# Threshold utility
# ------------------------------
def compute_dynamic_threshold(scores, factor=3.0):
    return np.mean(scores) + factor * np.std(scores)


# ------------------------------
# Model & Dataset Utilities
# ------------------------------
def load_model(cfg, device, DEBUG=False):
    model_type = cfg["model"]["type"].lower()

    if model_type == "stead":
        checkpoint = cfg["model"]["stead"].get("checkpoint", None)
    elif model_type == "autoencoder":
        checkpoint = cfg["model"]["autoencoder"].get("checkpoint", None)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = build_model(cfg, device, DEBUG)

    if checkpoint and os.path.exists(checkpoint):
        if DEBUG:
            print(f"[DEBUG] Loading checkpoint: {checkpoint}")
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict)
    elif DEBUG:
        print(f"[WARNING] No valid checkpoint found for model type {model_type}")

    model.eval()
    return model


def prepare_datasets(cfg, data_dir, DEBUG=False):
    anomaly_classes = cfg["training"].get("anomaly_class", ["normal"])
    full_dataset = X3DFeatureDataset(
        root_dir=data_dir, split="test",
        anomaly_classes=anomaly_classes, DEBUG=DEBUG
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
# Evaluate a single clip
# ------------------------------
# MSE(predicted future feature,actual next feature)

def evaluate_clip(model, xb, labels_class, anomaly_classes, threshold, metrics_cfg, device, model_type="stead"):
    xb = xb.to(device)  # xb shape: [B, seq_len, feature_dim]

    # ------------------------------
    # Model forward & MSE computation
    # ------------------------------
    if model_type == "stead":
        # STEAD predicts next-step features
        # Input: xb[:, :-1, :] -> all but last step
        # Target: xb[:, 1:, :] -> all but first step (next-step)
        input_seq = xb[:, :-1, :]
        target_seq = xb[:, 1:, :]
        pred_seq, _ = model(input_seq)
        mse = ((pred_seq - target_seq) ** 2).mean().item()
        recon = pred_seq
        target = target_seq
    elif model_type == "autoencoder":
        recon = model(xb)
        target = xb
        mse = ((recon - target) ** 2).mean().item()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # ------------------------------
    # Determine prediction & true class
    # ------------------------------
    true_class = labels_class[0] if labels_class else "unknown"
    if mse <= threshold:
        pred_class = anomaly_classes[0]  # normal
    else:
        pred_class = true_class if true_class in anomaly_classes else "unknown"

    # ------------------------------
    # Compute metrics
    # ------------------------------
    metric_values = {}
    for m in metrics_cfg:
        metric_type = str(m.get("type", "")).lower()
        if "mse" in metric_type:
            metric_values["mse"] = mse
        elif "psnr" in metric_type:
            metric_values["psnr"] = compute_psnr(target, recon)
        elif "ssim" in metric_type:
            metric_values["ssim"] = compute_ssim(target, recon)

    return mse, pred_class, true_class, metric_values


# ------------------------------
# Full evaluation
# ------------------------------
def evaluate(cfg, data_dir):
    DEBUG = cfg.get("debug", False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_type = cfg["model"]["type"].lower()

    if DEBUG:
        print(f"[DEBUG] Using device: {device}, Model type: {model_type}")

    model = load_model(cfg, device, DEBUG)
    test_dataset, val_dataset, anomaly_classes = prepare_datasets(cfg, data_dir, DEBUG)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    initial_threshold = float(cfg.get("evaluation", {}).get("mse_threshold", 0.001))
    metrics_cfg = cfg.get("evaluation", {}).get("metrics", [])

    # ------------------------------
    # Collect per-class MSE
    # ------------------------------
    per_class_scores = {cls: [] for cls in anomaly_classes}
    results = []

    with torch.no_grad():
        for idx, (xb, labels_idx, labels_class) in enumerate(test_loader):
            mse, pred_class, true_class, metric_values = evaluate_clip(
                model, xb, labels_class, anomaly_classes, 0, metrics_cfg, device, model_type
            )
            results.append({
                "clip_idx": idx,
                "mse": mse,
                "pred_class": pred_class,
                "true_class": true_class,
                "metrics": metric_values
            })
            if true_class in per_class_scores:
                per_class_scores[true_class].append(mse)

            if DEBUG:
                print(f"[DEBUG] Clip {idx}: MSE={mse:.8f}, Pred={pred_class}, True={true_class}, Metrics={metric_values}")

    # ------------------------------
    # Compute per-class dynamic thresholds
    # ------------------------------
    per_class_thresholds = {}
    print("\n=== Per-class MSE & suggested thresholds ===")
    for cls, scores in per_class_scores.items():
        mean_mse = np.mean(scores) if scores else 0.0
        std_mse = np.std(scores) if scores else 0.0
        threshold_cls = mean_mse + 3 * std_mse
        per_class_thresholds[cls] = threshold_cls
        print(f"{cls:15s}: mean={mean_mse:.8f}, std={std_mse:.8f}, threshold={threshold_cls:.8f}")

    # ------------------------------
    # Flag anomalies using per-class thresholds
    # ------------------------------
    print("\n=== Clip anomaly detection using per-class thresholds ===")
    for r in results[:10]:  # top 10 clips
        true_cls = r['true_class']
        cls_thresh = per_class_thresholds.get(true_cls, initial_threshold)
        is_anomaly = r['mse'] > cls_thresh
        metrics_str = ", ".join([f"{k}={v:.8f}" for k, v in r["metrics"].items()])
        print(f"Clip {r['clip_idx']}: True={true_cls}, MSE={r['mse']:.8f}, Threshold={cls_thresh:.8f}, Anomaly={is_anomaly}, {metrics_str}")

    avg_mse = np.mean([r['mse'] for r in results])
    print(f"\nAverage anomaly score (MSE): {avg_mse:.8f}")

    return results, per_class_thresholds, val_dataset


# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate STEAD / FeatureAutoencoder on X3D feature dataset"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    args = parser.parse_args()

    cfg = cfg_loader.load_config(args.config)
    evaluate(cfg, args.data_dir)


if __name__ == "__main__":
    main()
