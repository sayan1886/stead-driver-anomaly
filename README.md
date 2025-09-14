# STEAD Driver Behaviour Anomaly Detection

This repository implements **STEAD (2025)** for anomaly detection in driver behaviour, with a focus on Indian urban traffic scenarios.

## Features

- **X3D backbone** for spatio-temporal feature extraction
- **2+1D convolution + temporal attention** architecture (STEAD)
- Dashcam / in-car camera dataset support
- Sliding window extraction for long trips
- Modular **training, evaluation**, and **preprocessing** pipelines
- Lightweight & easy to extend

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data

The workflow uses **pre-extracted X3D features** from dashcam videos.

- A small X3D_Videos_sample/ folder (few .npy files) is included for quick testing.
- The full dataset (~1.9 GB) must be downloaded separately:
👉 [Download Dataset](https://drive.google.com/file/d/1LBTddU2mKuWvpbFOrqylJrZQ4u-U-zxG/view)

After downloading, place it under:

```bash
X3D_Videos/
  ├── Training_Normal_Videos_Anomaly/
  ├── Testing_Normal_Videos_Anomaly/
  ├── RoadAccidents/
  └── ... (other anomaly classes)
```

## Preprocessing

### Single video

Test preprocessing for a single `.npy` video:

```bash
python scripts/preprocess_single_video.py --config config/config.yaml --video <video.npy> --output <output.npy>
```

### Full dataset

Extract X3D features for all videos:

```bash
python scripts/preprocess_x3d.py --config config/config.yaml
```

Features are saved in the structure under X3D_Videos/.

## Training

Train STEAD on pre-extracted features:

```bash
python train.py --config config/config.yaml
```

- Checkpoints are saved to checkpoints/stead_driver.pt.
- Training uses configurable hyperparameters from config/config.yaml.

## Evaluation

Run evaluation on the test set:

```bash
python evaluate.py --config config/config.yaml --data_dir X3D_Videos
```

- Outputs **anomaly scores** per clip and summary statistics.
- Use the `DEBUG` flag in the config to see detailed logs.

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
├── dataset_x3d.py                   # Dataset for pre-extracted features
├── dashcam_dataset.py                # Optional: raw video dataset loader
├── config/
│   └── config.yaml                  # Training and model config
└── README.md
```

## Next Steps

- Extend STEAD with multi-scale temporal memory
- Add ROC / AUC metrics for anomaly detection
- Explore online learning for real-time anomaly detection