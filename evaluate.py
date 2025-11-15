# evaluate.py (patched full)
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

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
    return 1.0  # Placeholder


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

    checkpoint = None
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
    full_dataset = X3DFeatureDataset(root_dir=data_dir, split="test",
                                     anomaly_classes=anomaly_classes, DEBUG=DEBUG)

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
# Evaluate a single clip (Autoencoder)
# ------------------------------
def evaluate_clip(model, xb, labels_class, anomaly_classes, threshold, metrics_cfg, device, model_type="autoencoder"):
    xb = xb.to(device)
    recon = model(xb)
    mse = ((recon - xb) ** 2).mean().item()
    target = xb

    true_class = labels_class[0] if labels_class else "unknown"
    pred_class = anomaly_classes[0] if mse <= threshold else true_class

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
# Evaluate a single clip (STEAD)
# ------------------------------
def evaluate_clip_temporal(model, xb, labels_class, anomaly_classes, threshold, metrics_cfg, device):
    xb = xb.to(device)
    input_seq = xb[:, :-1, :]
    target_seq = xb[:, 1:, :]
    pred_seq, _ = model(input_seq)

    if pred_seq.dim() == 2:
        pred_seq = pred_seq.unsqueeze(0)
    if target_seq.dim() == 2:
        target_seq = target_seq.unsqueeze(0)

    timestep_errors = [((pred_seq[:, t, :] - target_seq[:, t, :]) ** 2).mean().item()
                       for t in range(pred_seq.shape[1])]
    mse = np.mean(timestep_errors)

    true_class = labels_class[0] if labels_class else "unknown"
    pred_class = anomaly_classes[0] if mse <= threshold else true_class

    metric_values = {}
    for m in metrics_cfg:
        metric_type = str(m.get("type", "")).lower()
        if "mse" in metric_type:
            metric_values["mse"] = mse
        elif "psnr" in metric_type:
            metric_values["psnr"] = compute_psnr(target_seq, pred_seq)
        elif "ssim" in metric_type:
            metric_values["ssim"] = compute_ssim(target_seq, pred_seq)

    return mse, pred_class, true_class, metric_values, timestep_errors


# ------------------------------
# Plot temporal MSE
# ------------------------------
def plot_temporal_errors(timestep_errors, clip_idx, save_dir=None):
    plt.figure(figsize=(8, 4))
    plt.plot(timestep_errors, marker='o')
    plt.title(f"Temporal MSE - Clip {clip_idx}")
    plt.xlabel("Timestep")
    plt.ylabel("MSE")
    plt.grid(True)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"temporal_mse_clip_{clip_idx}.png"))
    plt.show()


# ------------------------------
# Plot per-class summary
# ------------------------------
def plot_summary_table(per_class_scores, model_type, save_dir=None):
    classes = list(per_class_scores.keys())
    means = [np.mean(per_class_scores[cls]) for cls in classes]

    plt.figure(figsize=(12, 5))
    plt.bar(classes, means, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Mean MSE")
    plt.title(f"Per-Class MSE - {model_type.capitalize()}")
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"per_class_mse_{model_type}.png"))
    plt.show()

    # Print table
    df = pd.DataFrame({"Class": classes, "Mean MSE": means})
    print(f"\n=== Summary Table ({model_type}) ===")
    print(df)


# ------------------------------
# Full evaluation
# ------------------------------
def evaluate(cfg, data_dir, plot_temporal=False, temporal_save_dir=None, plot_summary=False):
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

    per_class_scores = {cls: [] for cls in anomaly_classes}
    results = []

    y_true_all, y_score_all = [], []

    with torch.no_grad():
        for idx, (xb, labels_idx, labels_class) in enumerate(test_loader):
            if model_type == "stead":
                mse, pred_class, true_class, metric_values, timestep_errors = evaluate_clip_temporal(
                    model, xb, labels_class, anomaly_classes, 0, metrics_cfg, device
                )
                if plot_temporal:
                    plot_temporal_errors(timestep_errors, idx, temporal_save_dir)
            else:
                mse, pred_class, true_class, metric_values = evaluate_clip(
                    model, xb, labels_class, anomaly_classes, 0, metrics_cfg, device, model_type
                )
                timestep_errors = None

            results.append({
                "clip_idx": idx,
                "mse": mse,
                "pred_class": pred_class,
                "true_class": true_class,
                "metrics": metric_values,
                "timestep_errors": timestep_errors
            })
            per_class_scores[true_class].append(mse)

            y_true_all.append(0 if true_class == anomaly_classes[0] else 1)
            y_score_all.append(mse)

            if DEBUG:
                print(f"[DEBUG] Clip {idx}: MSE={mse:.8f}, Pred={pred_class}, True={true_class}")

    # Per-class thresholds
    per_class_thresholds = {cls: np.mean(per_class_scores[cls]) + 3 * np.std(per_class_scores[cls])
                            for cls in anomaly_classes}
    print("\n=== Per-class MSE & thresholds ===")
    for cls, threshold in per_class_thresholds.items():
        print(f"{cls:15s}: mean={np.mean(per_class_scores[cls]):.8f}, std={np.std(per_class_scores[cls]):.8f}, threshold={threshold:.8f}")

    # Flag anomalies for first 10 clips
    print("\n=== Clip anomaly detection (first 10 clips) ===")
    for r in results[:10]:
        cls_thresh = per_class_thresholds[r['true_class']]
        is_anomaly = r['mse'] > cls_thresh
        metrics_str = ", ".join([f"{k}={v:.8f}" for k, v in r["metrics"].items()])
        print(f"Clip {r['clip_idx']}: True={r['true_class']}, MSE={r['mse']:.8f}, Threshold={cls_thresh:.8f}, Anomaly={is_anomaly}, {metrics_str}")

    avg_mse = np.mean([r['mse'] for r in results])
    print(f"\nAverage MSE: {avg_mse:.8f}")

    try:
        roc_auc = roc_auc_score(y_true_all, y_score_all)
        print(f"ROC-AUC: {roc_auc:.4f}")
    except ValueError:
        roc_auc = None
        print("ROC-AUC could not be computed")

    # Plot summary (only if enabled)
    if plot_summary:
        plot_summary_table(per_class_scores, model_type)

    return results, per_class_thresholds, val_dataset


# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate STEAD / Autoencoder on X3D features")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--plot_temporal", action='store_true', help="Plot temporal MSE")
    parser.add_argument("--temporal_save_dir", type=str, default=None, help="Save directory for temporal plots")
    args = parser.parse_args()

    cfg = cfg_loader.load_config(args.config)
    evaluate(cfg, args.data_dir, plot_temporal=args.plot_temporal, temporal_save_dir=args.temporal_save_dir)


if __name__ == "__main__":
    main()
