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
def evaluate_clip(model, xb, labels_class, anomaly_classes, threshold, metrics_cfg, device, model_type="stead"):
    xb = xb.to(device)

    # Forward & MSE
    if model_type == "stead":
        recon, _ = model(xb)
        target = xb.mean(dim=1)
    elif model_type == "autoencoder":
        recon = model(xb)
        target = xb
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    mse = ((recon - target) ** 2).mean().item()

    true_class = labels_class[0] if labels_class else "unknown"

    if mse <= threshold:
        pred_class = anomaly_classes[0]  # normal
        y_true = 0
    else:
        pred_class = true_class if true_class in anomaly_classes else "unknown"
        y_true = anomaly_classes.index(pred_class) if pred_class in anomaly_classes else len(anomaly_classes)

    # Metrics
    metric_values = {}
    for m in metrics_cfg:
        metric_type = str(m.get("type", "")).lower()
        if "mse" in metric_type:
            metric_values["mse"] = mse
        elif "psnr" in metric_type:
            metric_values["psnr"] = compute_psnr(target, recon)
        elif "ssim" in metric_type:
            metric_values["ssim"] = compute_ssim(target, recon)

    return mse, pred_class, true_class, y_true, metric_values


# ------------------------------
# Full evaluation
# ------------------------------
def evaluate(cfg, data_dir):
    DEBUG = cfg.get("debug", False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_type = cfg["model"]["type"].lower()

    if DEBUG:
        print(f"[DEBUG] Using device: {device}, Model type: {model_type}")

    # Load model
    model = load_model(cfg, device, DEBUG)

    # Prepare dataset
    test_dataset, val_dataset, anomaly_classes = prepare_datasets(cfg, data_dir, DEBUG)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Metrics config & initial threshold
    metrics_cfg = cfg.get("evaluation", {}).get("metrics", [])
    initial_threshold = float(cfg.get("evaluation", {}).get("mse_threshold", 0.001))

    # ------------------------------
    # Compute per-class MSE
    # ------------------------------
    per_class_scores = defaultdict(list)
    all_mse_scores = []

    with torch.no_grad():
        for xb, _, labels_class in test_loader:
            true_cls = labels_class[0] if labels_class else "unknown"
            mse, *_ = evaluate_clip(model, xb, labels_class, anomaly_classes, 0, metrics_cfg, device, model_type)
            per_class_scores[true_cls].append(mse)
            all_mse_scores.append(mse)

    # Compute per-class mean/std & suggested threshold
    per_class_thresholds = {}
    print("\n=== Per-class MSE & suggested thresholds ===")
    for cls, scores in per_class_scores.items():
        mean_mse = np.mean(scores)
        std_mse = np.std(scores)
        threshold_cls = mean_mse + 3 * std_mse
        per_class_thresholds[cls] = threshold_cls
        print(f"{cls:15s}: mean={mean_mse:.6f}, std={std_mse:.6f}, threshold={threshold_cls:.6f}")

    # Global threshold
    dynamic_threshold = compute_dynamic_threshold(all_mse_scores, factor=3.0)
    threshold = max(initial_threshold, dynamic_threshold)

    if DEBUG:
        print(f"\n[INFO] Initial threshold={initial_threshold:.6f}, Dynamic threshold={dynamic_threshold:.6f}, Using={threshold:.6f}")

    # ------------------------------
    # Final evaluation
    # ------------------------------
    results, y_true_multi, y_score = [], [], []
    with torch.no_grad():
        for idx, (xb, labels_idx, labels_class) in enumerate(test_loader):
            mse, pred_class, true_class, y_true, metric_values = evaluate_clip(
                model, xb, labels_class, anomaly_classes, threshold, metrics_cfg, device, model_type
            )
            results.append({
                "clip_idx": idx,
                "mse": mse,
                "pred_class": pred_class,
                "true_class": true_class,
                "metrics": metric_values
            })
            y_true_multi.append(y_true)
            y_score.append(mse)

            if DEBUG:
                print(f"[DEBUG] Clip {idx}: MSE={mse:.6f}, Pred={pred_class}, True={true_class}, Metrics={metric_values}")

    # Compute AUC if multi-class
    auc_score = None
    y_true_unique = np.unique(y_true_multi)
    if len(y_true_unique) > 1:
        try:
            if len(y_true_unique) == 2:
                auc_score = roc_auc_score(y_true_multi, y_score)
            else:
                auc_score = roc_auc_score(y_true_multi, y_score, multi_class="ovr")
        except Exception as e:
            if DEBUG:
                print(f"[WARNING] AUC computation failed: {e}")

    # ------------------------------
    # Summary
    # ------------------------------
    print("\n=== Evaluation Summary ===")
    for r in results[:10]:
        metrics_str = ", ".join([f"{k}={v:.6f}" for k, v in r["metrics"].items()])
        print(f"Clip {r['clip_idx']}: Score={r['mse']:.6f}, Pred={r['pred_class']}, True={r['true_class']}, {metrics_str}")

    avg_mse = np.mean([r['mse'] for r in results])
    print(f"Average anomaly score (MSE): {avg_mse:.6f}")
    print(f"AUC: {auc_score:.6f}" if auc_score is not None else "AUC: undefined (single class)")

    return results, auc_score, val_dataset


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
