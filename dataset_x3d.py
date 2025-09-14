# dataset_x3d.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class X3DFeatureDataset(Dataset):
    def __init__(self, root_dir, split="train", DEBUG=False):
        """
        Args:
            root_dir (str): Root directory where features are stored
            split (str): "train" or "test"
            DEBUG (bool): print debug information
        """
        self.samples = []
        self.labels = []
        self.DEBUG = DEBUG

        if split == "train":
            normal_dir = os.path.join(root_dir, "Training_Normal_Videos_Anomaly")
            self._load_dir(normal_dir, label=0)

        elif split == "test":
            # Normal videos
            normal_dir = os.path.join(root_dir, "Testing_Normal_Videos_Anomaly")
            self._load_dir(normal_dir, label=0)

            # Anomaly videos
            anomaly_classes = [
                "RoadAccidents", "Arson", "Shoplifting", "Stealing", "Burglary",
                "Fighting", "Vandalism", "Explosion", "Arrest", "Abuse",
                "Robbery", "Assault", "Shooting"
            ]
            for cls in anomaly_classes:
                cls_dir = os.path.join(root_dir, cls)
                self._load_dir(cls_dir, label=1)

        if self.DEBUG:
            print(f"[DEBUG] Loaded {len(self.samples)} samples for split='{split}'")

    def _load_dir(self, dir_path, label):
        """Helper to load .npy files from a directory"""
        if not os.path.isdir(dir_path):
            if self.DEBUG:
                print(f"[DEBUG] Directory does not exist: {dir_path}")
            return

        for f in os.listdir(dir_path):
            if f.lower().endswith(".npy"):
                full_path = os.path.join(dir_path, f)
                self.samples.append(full_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            feat = np.load(self.samples[idx])  # shape [T, F]
            feat = torch.from_numpy(feat).float()  # convert to torch tensor
        except Exception as e:
            if self.DEBUG:
                print(f"[DEBUG] Failed to load {self.samples[idx]}: {e}")
            # Return a zero tensor if loading fails
            feat = torch.zeros((16, 2048), dtype=torch.float32)  # default shape [T, F]
        label = self.labels[idx]
        return feat, label