# compare_models.py
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from train import train as train_model
from evaluate import evaluate
import config.config as cfg_loader

MODEL_TYPES = ["autoencoder", "stead"]

def compare_models(cfg, data_dir):
    compare_cfg = cfg.get("compare", {})
    COMPARE_DIR = compare_cfg.get("save_dir", "./compare")
    os.makedirs(COMPARE_DIR, exist_ok=True)
    PLOT_COMPARE = compare_cfg.get("plot", True)
    FORCE_TRAIN = compare_cfg.get("force_train", True)

    results_dict = {}
    thresholds_dict = {}

    for model_type in MODEL_TYPES:
        print(f"\n==== Processing {model_type} ====")
        cfg["model"]["type"] = model_type

        # Train model if force_train or checkpoint missing
        checkpoint = cfg["model"][model_type].get("checkpoint", None)
        if FORCE_TRAIN or not (checkpoint and os.path.exists(checkpoint)):
            print(f"[INFO] Training {model_type} model...")
            train_model(cfg)
        else:
            print(f"[INFO] Using existing checkpoint for {model_type}.")

        # Evaluate model
        print(f"[INFO] Evaluating {model_type} model...")
        results, per_class_thresholds, _ = evaluate(
            cfg, 
            data_dir, 
            plot_temporal=cfg.get("compare", {}).get("evaluate_plot", False), 
            temporal_save_dir=None,
            plot_summary=False  # skip plot_summary_table
        )

        # Prepare summary dataframe
        summary_df = pd.DataFrame([
            {
                "Class": cls,
                "Mean_MSE": float(np.mean([r['mse'] for r in results if r['true_class'] == cls]))
            }
            for cls in cfg["training"]["anomaly_class"]
        ])
        csv_path = os.path.join(COMPARE_DIR, f"{model_type}_summary.csv")
        summary_df.to_csv(csv_path, index=False)
        print(f"[INFO] Saved {model_type} summary CSV: {csv_path}")

        results_dict[model_type] = summary_df
        thresholds_dict[model_type] = per_class_thresholds

    # -------------------------
    # Combined summary table log
    # -------------------------
    print("\n=== Combined Autoencoder vs STEAD Mean MSE ===")
    combined_df = pd.DataFrame({
        "Class": cfg["training"]["anomaly_class"],
        "Autoencoder_MSE": [results_dict["autoencoder"].loc[results_dict["autoencoder"]["Class"] == cls, "Mean_MSE"].values[0] for cls in cfg["training"]["anomaly_class"]],
        "STEAD_MSE": [results_dict["stead"].loc[results_dict["stead"]["Class"] == cls, "Mean_MSE"].values[0] for cls in cfg["training"]["anomaly_class"]],
    })
    print(combined_df.to_string(index=False))

    # -------------------------
    # Save combined CSV
    # -------------------------
    combined_csv_path = os.path.join(COMPARE_DIR, "combined_summary.csv")
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"[INFO] Combined summary CSV saved at {combined_csv_path}")

    # -------------------------
    # Combined comparison plot (dual-axis)
    # -------------------------
    if PLOT_COMPARE:
        classes = cfg["training"]["anomaly_class"]
        x = np.arange(len(classes))
        width = 0.35

        ae_mse = combined_df["Autoencoder_MSE"].values
        stead_mse = combined_df["STEAD_MSE"].values

        fig, ax1 = plt.subplots(figsize=(12, 6))
        color_ae = "skyblue"
        color_stead = "salmon"

        # Autoencoder bars (left y-axis)
        bars1 = ax1.bar(x - width/2, ae_mse, width=width, label="Autoencoder", color=color_ae)
        ax1.set_ylabel("Autoencoder Mean MSE", color=color_ae)
        ax1.tick_params(axis='y', labelcolor=color_ae)

        # STEAD bars (right y-axis)
        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + width/2, stead_mse, width=width, label="STEAD", color=color_stead)
        ax2.set_ylabel("STEAD Mean MSE", color=color_stead)
        ax2.tick_params(axis='y', labelcolor=color_stead)

        ax1.set_xticks(x)
        ax1.set_xticklabels(classes, rotation=45, ha='right')
        plt.title("Autoencoder vs STEAD: Per-Class Mean MSE Comparison")

        # Combined legend above the plot
        handles = [bars1[0], bars2[0]]
        labels = ["Autoencoder", "STEAD"]
        ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)

        plt.tight_layout()
        plot_path = os.path.join(COMPARE_DIR, "comparison_plot.png")
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"[INFO] Comparison plot saved at {plot_path}")

    print("\n[INFO] Comparison complete.")
    return results_dict, thresholds_dict, combined_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare Autoencoder vs STEAD on X3D dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    args = parser.parse_args()

    cfg = cfg_loader.load_config(args.config)
    compare_models(cfg, args.data_dir)
