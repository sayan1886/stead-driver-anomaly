# scripts/preprocess_x3d.py
import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
from pytorchvideo.models.hub import x3d_m
import torch.nn as nn

# ------------------------------
# Configuration
# ------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_LEN = 16
CROP_SIZE = 224
DEBUG = False  # Set False to disable debug prints

# ------------------------------
# Initialize X3D (pretrained)
# ------------------------------
model = x3d_m(pretrained=True)
model = model.eval().to(DEVICE)

# Remove only the classification projection (keep pooling!)
if hasattr(model.blocks[-1], "proj"):
    model.blocks[-1].proj = nn.Identity()

# ------------------------------
# Feature extraction function
# ------------------------------
def extract_features_from_npy(npy_path):
    """
    Extract X3D features per clip.
    Returns: [T=CLIP_LEN, F=feature_dim]
    """
    data = np.load(npy_path)  # [num_frames, 16, H, W]
    if DEBUG: print(f"[DEBUG] Loaded {npy_path}: {data.shape}")

    if data.shape[0] < CLIP_LEN:
        if DEBUG: print(f"[DEBUG] Skipping {npy_path}, not enough frames")
        return None

    clip = data[:CLIP_LEN]  # [CLIP_LEN,16,H,W]
    if DEBUG: print(f"[DEBUG] Clip selected: {clip.shape}")

    # Convert 16-channel → 3-channel
    clip_rgb = np.mean(clip, axis=1, keepdims=True)  # [CLIP_LEN,1,H,W]
    clip_rgb = np.repeat(clip_rgb, 3, axis=1)        # [CLIP_LEN,3,H,W]
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

    # Forward through model
    with torch.no_grad():
        feats = model(clip_tensor)           # [1, 2048] for x3d_m
        if DEBUG: print(f"[DEBUG] Features raw: {feats.shape}")
        features = feats.repeat(CLIP_LEN, 1) # [T, F]
        if DEBUG: print(f"[DEBUG] Features repeated: {features.shape}")

    return features.cpu().numpy()

# ------------------------------
# Process all .npy files recursively
# ------------------------------
def process_npy_files(input_dir, output_dir):
    all_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".npy"):
                all_files.append(os.path.join(root, f))

    for npy_path in tqdm(all_files, desc="Processing npy files"):
        rel_path = os.path.relpath(os.path.dirname(npy_path), input_dir)
        out_dir = os.path.join(output_dir, rel_path)
        os.makedirs(out_dir, exist_ok=True)

        features = extract_features_from_npy(npy_path)
        if features is None:
            continue

        out_path = os.path.join(out_dir, os.path.basename(npy_path))
        np.save(out_path, features)
        if DEBUG:
            print(f"[DEBUG] Processed {npy_path} -> {features.shape}")

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Preprocess .npy videos through X3D backbone to extract features per frame.\n"
                    "Example:\n"
                    "python scripts/preprocess_x3d.py "
                    "--input_dir X3D_Raw_Videos "
                    "--output_dir X3D_Videos",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Path to raw .npy videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save features .npy")
    args = parser.parse_args()

    print(f"Starting preprocessing...")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    process_npy_files(args.input_dir, args.output_dir)
    print("Feature extraction completed!")