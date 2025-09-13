# scripts/preprocess_single_video.py
import os
import torch
import numpy as np
import cv2
from pytorchvideo.models.hub import x3d_m
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_LEN = 16
CROP_SIZE = 224
DEBUG = True

# ------------------------------
# Initialize X3D (pretrained)
# ------------------------------
model = x3d_m(pretrained=True)
model = model.eval().to(DEVICE)

# Remove only the classification projection
if hasattr(model.blocks[-1], "proj"):
    model.blocks[-1].proj = nn.Identity()

def extract_features_from_npy(npy_path):
    data = np.load(npy_path)  # [num_frames, 16, H, W]
    if DEBUG: print(f"[DEBUG] Loaded {npy_path}: {data.shape}")

    if data.shape[0] < CLIP_LEN:
        if DEBUG: print(f"[DEBUG] Not enough frames, skipping")
        return None

    clip = data[:CLIP_LEN]
    if DEBUG: print(f"[DEBUG] Clip selected: {clip.shape}")

    # Convert 16-channel → 3-channel
    clip_rgb = np.mean(clip, axis=1, keepdims=True)
    clip_rgb = np.repeat(clip_rgb, 3, axis=1)
    if DEBUG: print(f"[DEBUG] Clip after channel conversion: {clip_rgb.shape}")

    # Resize & normalize
    frames = []
    mean = torch.tensor([0.45, 0.45, 0.45])[:, None, None]
    std = torch.tensor([0.225, 0.225, 0.225])[:, None, None]

    for f in clip_rgb:
        resized = np.stack([cv2.resize(f[c], (CROP_SIZE, CROP_SIZE)) for c in range(3)], axis=0)
        f_tensor = torch.tensor(resized, dtype=torch.float32) / 255.0
        f_tensor = (f_tensor - mean) / std
        frames.append(f_tensor)

    clip_tensor = torch.stack(frames).unsqueeze(0).to(DEVICE)  # [1,T,C,H,W]
    if DEBUG: print(f"[DEBUG] Clip tensor after stacking: {clip_tensor.shape}")
    clip_tensor = clip_tensor.permute(0, 2, 1, 3, 4)           # [B,C,T,H,W]
    if DEBUG: print(f"[DEBUG] Clip tensor after permute: {clip_tensor.shape}")

    with torch.no_grad():
        feats = model(clip_tensor)  # [1, 1024]
        if DEBUG: print(f"[DEBUG] Features raw: {feats.shape}")
        features = feats.repeat(CLIP_LEN, 1)  # [16, 1024]
        if DEBUG: print(f"[DEBUG] Features repeated: {features.shape}")

    return features.cpu().numpy()

if __name__ == "__main__":
    video_path = "X3D_Raw_Videos/RoadAccidents/RoadAccidents001_x264.npy"
    features = extract_features_from_npy(video_path)
    if features is not None:
        if DEBUG: print(f"[DEBUG] Final feature shape: {features.shape}")
        out_path = "X3D_Videos/SingleTestVideo.npy"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, features)
        print(f"Features saved to {out_path}")