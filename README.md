
# STEAD Driver Behaviour Anomaly Detection

This repo implements a simplified version of STEAD (2025) for driver behaviour anomaly detection in Indian urban traffic.

## Features

- 2+1D convolution + temporal attention architecture
- Dashcam / in-car camera dataset support
- Sliding window extraction for long trips
- Training & evaluation scripts
- Lightweight and modular

## Setup

```python
pip install -r requirements.txt
```

## Data

Place dashcam videos in `data/train/`, `data/val/`, `data/test/`.

## Dataset Setup

This repo uses pre-extracted **X3D features** from the STEAD dataset.

- We include a small **X3D_Videos_sample/** folder (few `.npy` files) so you can quickly test the pipeline.
- The **full dataset** (~55 GB) cannot be hosted on GitHub.  
  Please download from: [Google Drive link](https://drive.google.com/file/d/1LBTddU2mKuWvpbFOrqylJrZQ4u-U-zxG/view)

After downloading, place the full dataset in:
X3D_Videos/
  ├── Training_Normal_Videos_Anomaly/
  ├── Testing_Normal_Videos_Anomaly/
  ├── RoadAccidents/
  ...

## Train

```python
python train.py
```

## Evaluate

```python
python eval.py
```
