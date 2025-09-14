# scripts/preprocess_single_video.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import torch
import torch.nn as nn
from pytorchvideo.models.hub import x3d_m

import config.config as cfg_loader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# Feature extraction function
# ------------------------------
def extract_features_from_npy(npy_path, model, clip_len, crop_size, DEBUG=False):
    data = np.load(npy_path)  # [num_frames, 16, H, W]
    if DEBUG:
        print(f"[DEBUG] Loaded {npy_path}: {data.shape}")

    if data.shape[0] < clip_len:
        if DEBUG:
            print(f"[DEBUG] Not enough frames, skipping")
        return None

    clip = data[:clip_len]
    if DEBUG:
        print(f"[DEBUG] Clip selected: {clip.shape}")

    # Convert 16-channel → 3-channel
    clip_rgb = np.mean(clip, axis=1, keepdims=True)
    clip_rgb = np.repeat(clip_rgb, 3, axis=1)
    if DEBUG:
        print(f"[DEBUG] Clip after channel conversion: {clip_rgb.shape}")

    # Resize & normalize
    frames = []
    mean = torch.tensor([0.45, 0.45, 0.45])[:, None, None]
    std = torch.tensor([0.225, 0.225, 0.225])[:, None, None]

    for f in clip_rgb:
        resized = np.stack([cv2.resize(f[c], (crop_size, crop_size)) for c in range(3)], axis=0)
        f_tensor = torch.tensor(resized, dtype=torch.float32) / 255.0
        f_tensor = (f_tensor - mean) / std
        frames.append(f_tensor)

    clip_tensor = torch.stack(frames).unsqueeze(0).to(DEVICE)  # [1,T,C,H,W]
    if DEBUG:
        print(f"[DEBUG] Clip tensor after stacking: {clip_tensor.shape}")
    clip_tensor = clip_tensor.permute(0, 2, 1, 3, 4)  # [B,C,T,H,W]
    if DEBUG:
        print(f"[DEBUG] Clip tensor after permute: {clip_tensor.shape}")

    with torch.no_grad():
        feats = model(clip_tensor)  # [1, feature_dim]
        if DEBUG:
            print(f"[DEBUG] Features raw: {feats.shape}")
        features = feats.repeat(clip_len, 1)  # [clip_len, feature_dim]
        if DEBUG:
            print(f"[DEBUG] Features repeated: {features.shape}")

    return features.cpu().numpy()

# ------------------------------
# Main
# ------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract features from a single .npy dashcam video using X3D backbone",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g., config/config.yaml)"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="X3D_Raw_Videos/RoadAccidents/RoadAccidents001_x264.npy",
        help="Path to .npy video file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="X3D_Videos/SingleTestVideo.npy",
        help="Path to save extracted features"
    )
    args = parser.parse_args()

    # Load config
    cfg = cfg_loader.load_config(args.config)
    CLIP_LEN = cfg["model"]["seq_len"]
    CROP_SIZE = cfg["model"]["crop_size"]
    DEBUG = cfg.get("debug", False)

    if DEBUG:
        print(f"[DEBUG] Using device: {DEVICE}")
        print(f"[DEBUG] Loading video: {args.video}")

    # Initialize X3D model
    model = x3d_m(pretrained=True)
    model = model.eval().to(DEVICE)
    if hasattr(model.blocks[-1], "proj"):
        model.blocks[-1].proj = nn.Identity()

    # Extract features
    features = extract_features_from_npy(args.video, model, CLIP_LEN, CROP_SIZE, DEBUG=DEBUG)
    if features is not None:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        np.save(args.output, features)
        print(f"Features saved to {args.output}")

if __name__ == "__main__":
    main()