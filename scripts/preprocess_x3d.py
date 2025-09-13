# scripts/preprocess_x3d.py
import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from tqdm import tqdm
from pytorchvideo.models.hub import x3d_s
from torchvision.transforms import Compose, Normalize, Lambda

# ------------------------------
# Configuration
# ------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_LEN = 16
CROP_SIZE = 224
DEBUG = True  # Set False to disable debug prints

# Transformation
transform = Compose([
    Lambda(lambda x: x / 255.0),
    Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

# ------------------------------
# Initialize X3D model (pretrained)
# ------------------------------
model = x3d_s(pretrained=True, progress=True)
model = model.eval().to(DEVICE)

def extract_features_from_npy(npy_path):
    data = np.load(npy_path)  # shape: [num_frames, 16, H, W]
    
    # Take first CLIP_LEN frames
    clip = data[:CLIP_LEN]  # [CLIP_LEN, 16, H, W]

    # Reduce 16 channels → 3 channels (take first 3)
    clip = clip[:, :3, :, :]  # [CLIP_LEN, 3, H, W]

    # Resize and normalize
    frames = []
    for f in clip:  # f: [3, H, W]
        resized = np.stack([cv2.resize(f[c], (CROP_SIZE, CROP_SIZE)) for c in range(3)], axis=0)
        # Normalize
        f_tensor = torch.tensor(resized, dtype=torch.float32) / 255.0
        mean = torch.tensor([0.45, 0.45, 0.45])[:, None, None]
        std = torch.tensor([0.225, 0.225, 0.225])[:, None, None]
        f_tensor = (f_tensor - mean) / std
        frames.append(f_tensor)

    clip_tensor = torch.stack(frames).unsqueeze(0).to(DEVICE)  # [1, T, C, H, W]
    clip_tensor = clip_tensor.permute(0,2,1,3,4)  # -> [B, C, T, H, W]

    with torch.no_grad():
        features = model(clip_tensor)

    return features.squeeze(0).cpu().numpy()

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
# Usage: python scripts/preprocess_x3d.py --input_dir X3D_Raw_Videos --output_dir X3D_Videos
# ------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Preprocess .npy videos through X3D to extract features.\n"
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