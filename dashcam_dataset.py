
import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np

class DashcamDataset(Dataset):
    def __init__(self, folder, seq_len=16, stride=8, resize=(128,128), labels_dict=None):
        self.folder = folder
        self.seq_len = seq_len
        self.stride = stride
        self.resize = resize
        self.video_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.mp4')]
        self.samples = []
        self.labels_dict = labels_dict

        for vf in self.video_files:
            cap = cv2.VideoCapture(vf)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            for start in range(0, max(1, total_frames - seq_len + 1), stride):
                self.samples.append((vf, start))

    def read_window(self, video_path, start_frame):
        cap = cv2.VideoCapture(video_path)
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        count = 0
        while count < self.seq_len:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, self.resize)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            count += 1
        cap.release()
        if len(frames) < self.seq_len:
            pad_len = self.seq_len - len(frames)
            frames += [frames[-1]] * pad_len
        frames = np.array(frames, dtype=np.float32)/255.0
        frames = np.transpose(frames, (0,3,1,2))
        return frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, start_frame = self.samples[idx]
        frames = self.read_window(video_path, start_frame)
        sample = torch.tensor(frames)
        label = None
        if self.labels_dict:
            vid_name = os.path.basename(video_path)
            label_list = self.labels_dict.get(vid_name, [])
            window_idx = idx
            if window_idx < len(label_list):
                label = torch.tensor(label_list[window_idx], dtype=torch.float32)
        return sample, label
