# dataset_x3d.py
import os

import numpy as np
import torch
from torch.utils.data import Dataset

class X3DFeatureDataset(Dataset):
    def __init__(self, root_dir, split="train", anomaly_classes=None, DEBUG=False, default_shape=(16, 2048)):
        """
        Dataset for X3D features with optional anomaly class labels.

        Args:
            root_dir (str): Root directory where feature .npy files are stored.
            split (str): "train" or "test".
            anomaly_classes (list[str]): List of anomaly class names from config.
                                         anomaly_classes[0] should be "normal".
            DEBUG (bool): Enable debug messages.
            default_shape (tuple): Shape of fallback tensor if loading fails.
        """
        self.samples = []
        self.labels = []
        self.class_labels = []
        self.DEBUG = DEBUG
        self.default_shape = default_shape
        self.anomaly_classes = anomaly_classes or ["normal", "anomaly"]

        if split == "train":
            normal_dir = os.path.join(root_dir, "Training_Normal_Videos_Anomaly")
            self._load_dir(normal_dir, class_idx=0)

        elif split == "test":
            # Normal videos
            self._load_dir(os.path.join(root_dir, "Testing_Normal_Videos_Anomaly"), class_idx=0)

            # Anomaly videos
            for class_idx, cls_name in enumerate(self.anomaly_classes[1:], start=1):
                cls_dir = os.path.join(root_dir, cls_name)
                self._load_dir(cls_dir, class_idx=class_idx)

        if self.DEBUG:
            print(f"[DEBUG] Loaded {len(self.samples)} samples for split='{split}'")

    def _load_dir(self, dir_path, class_idx):
        """Helper to load .npy files from a directory and assign class label."""
        if not os.path.isdir(dir_path):
            if self.DEBUG:
                print(f"[DEBUG] Directory does not exist: {dir_path}")
            return

        class_name = self.anomaly_classes[class_idx]
        for f in os.listdir(dir_path):
            if f.lower().endswith(".npy"):
                full_path = os.path.join(dir_path, f)
                self.samples.append(full_path)
                self.labels.append(class_idx)      # numeric class
                self.class_labels.append(class_name)  # string class
                if self.DEBUG:
                    print(f"[DEBUG] Added sample: {full_path}, label={class_idx}, class={class_name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        label_idx = self.labels[idx]
        label_class = self.class_labels[idx]
        try:
            feat = np.load(path)  # [T, F]
            feat = torch.from_numpy(feat).float() # convert to torch tensor
        except Exception as e:
            if self.DEBUG:
                print(f"[DEBUG] Failed to load {path}: {e}")
            # Return a zero tensor if loading fails
            feat = torch.zeros(self.default_shape, dtype=torch.float32) # default shape [T, F]

        return feat, label_idx, label_class