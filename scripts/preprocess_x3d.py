# scripts/preprocess_x3d.py
import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from tqdm import tqdm
from pytorchvideo.models import x3d
from torchvision.transforms import Compose, Normalize, Lambda

# ------------------------------
# Configuration
# ------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_LEN = 16
CROP_SIZE = 224

# Transformation
transform = Compose([
    Lambda(lambda x: x / 255.0),
    Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

# Initialize X3D model
model = x3d.create_x3d(
    input_clip_length=CLIP_LEN,
    input_crop_size=CROP_SIZE,
    model_num_class=400,
    dropout_rate=0.5,
    width_factor=2.2,
    depth_factor=2.2,
    norm=nn.BatchNorm3d,
    norm_eps=1e-5,
    norm_momentum=0.1,
    activation=nn.ReLU,
    stem_dim_in=3,
    stem_conv_kernel_size=(3, 7, 7),
    stem_conv_stride=(1, 2, 2),
    stage_conv_kernel_size=[(1, 3, 3)]*4,
    stage_spatial_stride=[(1, 2, 2)]*4,
    stage_temporal_stride=[(1, 1, 1)]*4,  # avoids temporal stride mismatch
    bottleneck_factor=1.0,
    se_ratio=0.25,
)
model = model.eval().to(DEVICE)

# ------------------------------
# Feature extraction
# ------------------------------
def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < CLIP_LEN:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (CROP_SIZE, CROP_SIZE))
        frames.append(frame)
    cap.release()

    if len(frames) < CLIP_LEN:
        return None

    frames = np.stack(frames)
    frames = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)  # [T, C, H, W]
    frames = transform(frames).unsqueeze(0).to(DEVICE)  # [1, T, C, H, W]

    with torch.no_grad():
        features = model(frames)
    return features.squeeze(0).cpu().numpy()

# ------------------------------
# Process all videos recursively
# ------------------------------
def process_videos(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)
        out_dir = os.path.join(output_dir, rel_path)
        os.makedirs(out_dir, exist_ok=True)

        for f in files:
            if not f.endswith(".mp4"):
                continue
            video_path = os.path.join(root, f)
            features = extract_features(video_path)
            if features is None:
                print(f"Skipping {video_path}: Not enough frames")
                continue
            np.save(os.path.join(out_dir, f"{os.path.splitext(f)[0]}.npy"), features)
            print(f"Processed {video_path}")

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Preprocess X3D videos into [T, F] .npy features.\n"
                    "Example usage:\n"
                    "python scripts/preprocess_x3d.py "
                    "--input_dir X3D_Raw_Videos "
                    "--output_dir X3D_Videos",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Path to raw videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save .npy features")
    args = parser.parse_args()

    print(f"Starting preprocessing...")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    process_videos(args.input_dir, args.output_dir)
    print("Feature extraction completed!")
