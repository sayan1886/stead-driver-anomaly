import os
import numpy as np
import torch
from torch.utils.data import Dataset

class X3DFeatureDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.samples = []
        self.labels = []

        if split == "train":
            normal_dir = os.path.join(root_dir, "Training_Normal_Videos_Anomaly")
            for f in os.listdir(normal_dir):
                if f.endswith(".npy"):
                    self.samples.append(os.path.join(normal_dir, f))
                    self.labels.append(0)  # 0 = normal

        elif split == "test":
            # normal videos
            normal_dir = os.path.join(root_dir, "Testing_Normal_Videos_Anomaly")
            for f in os.listdir(normal_dir):
                if f.endswith(".npy"):
                    self.samples.append(os.path.join(normal_dir, f))
                    self.labels.append(0)

            # anomaly videos
            anomaly_classes = [
                "RoadAccidents", "Arson", "Shoplifting", "Stealing", "Burglary",
                "Fighting", "Vandalism", "Explosion", "Arrest", "Abuse",
                "Robbery", "Assault", "Shooting"
            ]
            for cls in anomaly_classes:
                cls_dir = os.path.join(root_dir, cls)
                if os.path.isdir(cls_dir):
                    for f in os.listdir(cls_dir):
                        if f.endswith(".npy"):
                            self.samples.append(os.path.join(cls_dir, f))
                            self.labels.append(1)  # 1 = anomaly

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat = np.load(self.samples[idx])  # shape [T, F]
        feat = torch.from_numpy(feat).float()  # convert to torch tensor
        return feat, self.labels[idx]