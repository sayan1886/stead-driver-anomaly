# scripts/inspect_npy_videos.py
import os
import numpy as np
import random
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config.config as cfg_loader

def inspect_npy_file(npy_path, max_frames=3):
    data = np.load(npy_path)
    print(f"\nFile: {npy_path}")
    print("  Array type:", type(data))
    print("  Shape:", data.shape)
    print("  Dtype:", data.dtype)
    print("  Min / Max:", data.min(), data.max())

    # Peek at first few frames
    if data.ndim >= 3:
        for i in range(min(max_frames, data.shape[0])):
            print(f"  Frame {i} slice:")
            print(data[i, 0:3, 0:3, 0:3] if data.ndim == 4 else data[i, 0:3, 0:3])
    elif data.ndim == 2:
        print("  Sample slice:", data[0:3, 0:3])
    else:
        print("  Data:", data)

def inspect_random_npy(input_dir, sample_size=10):
    all_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".npy"):
                all_files.append(os.path.join(root, f))

    if not all_files:
        print("No .npy files found in directory.")
        return

    sampled_files = random.sample(all_files, min(sample_size, len(all_files)))
    for npy_path in sampled_files:
        inspect_npy_file(npy_path, max_frames=3)


# ------------------------------
# Main
# Usage: python scripts/inspect_npy_videos.py --config config/config.yaml --sample_size 10
# ------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect random .npy video files to understand shape, dtype, and content\n"
                    "Usage:\n"
                    "  python scripts/inspect_npy_videos.py --config config/config.yaml --sample_size 10\n"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g., config/config.yaml)"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of random files to inspect (overrides config if provided)"
    )
    args = parser.parse_args()

    # Load YAML config
    cfg = cfg_loader.load_config(args.config)
    input_dir = cfg["dataset"].get("raw_dir", "./X3D_Raw_Videos")
    sample_size = args.sample_size if args.sample_size is not None else 10

    print("=== Inspecting random .npy video files ===")
    print(f"Input directory : {input_dir}")
    print(f"Sample size     : {sample_size}")
    print("-----------------------------------------\n")

    inspect_random_npy(input_dir, sample_size=sample_size)
    
if __name__ == "__main__":
    main()