# scripts/inspect_npy_videos.py
import os
import numpy as np
import random

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
        inspect_npy_file(npy_path)

# ------------------------------
# Main
# Usage: python scripts/inspect_npy_videos.py --input_dir X3D_Raw_Videos --sample_size 10
# ------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Inspect random .npy video files to understand shape, dtype, and content"
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .npy video files")
    parser.add_argument("--sample_size", type=int, default=10, help="Number of random files to inspect")
    args = parser.parse_args()

    print("Inspecting random .npy files from:", args.input_dir)
    inspect_random_npy(args.input_dir, sample_size=args.sample_size)
    print("\nUsage example:")
    print("python scripts/inspect_npy_videos.py --input_dir X3D_Raw_Videos --sample_size 10")