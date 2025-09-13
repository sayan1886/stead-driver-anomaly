# STEAD Driver Behaviour Anomaly Detection

This repository implements a simplified version of **STEAD (2025)** for anomaly detection in driver behaviour, with a focus on Indian urban traffic scenarios.

## Features

- **X3D backbone** for spatio-temporal feature extraction
- **2+1D convolution + temporal attention** architecture (STEAD)
- Dashcam / in-car camera dataset support
- Sliding window extraction for long trips
- Modular **training** and **evaluation** pipelines
- Lightweight & easy to extend

---

## Setup

```bash
pip install -r requirements.txt
```

## Data

The workflow uses pre-extracted X3D features from dashcam videos.

- A small `X3D_Videos_sample/` folder (few `.npy` files) is included for quick testing.

- The full dataset (~1.9 GB) cannot be hosted here.
Please download it from Google Drive:
👉 [Download Dataset](https://drive.google.com/file/d/1LBTddU2mKuWvpbFOrqylJrZQ4u-U-zxG/view)

After download, place it under:

```pgsql
X3D_Videos/
  ├── Training_Normal_Videos_Anomaly/
  ├── Testing_Normal_Videos_Anomaly/
  ├── RoadAccidents/
  └── ...
```

## Train

Train STEAD with extracted features:

```bash
python train.py --dataset x3d --data_dir X3D_Videos
```

Model checkpoints are saved to `checkpoints/`.

## Evaluate

Run evaluation on the test set:

```bash
python evaluate.py --data_dir X3D_Videos
```

This produces anomaly scores (per clip) and summary statistics.

## Folder Structure

```bash
.
├── scripts/
│   ├── preprocess_single_video.py   # Test preprocessing on one video
│   ├── preprocess_x3d.py            # Extract features for full dataset
├── models/
│   └── stead_model.py               # STEAD architecture
├── X3D_Videos_sample/               # Small demo feature set
├── train.py                         # Training script
├── evaluate.py                      # Evaluation script
└── README.md
```

## Next Steps

- Extend STEAD with multi-scale temporal memory
- Add ROC / AUC metrics for anomaly detection
- Explore online learning for real-time anomaly detection