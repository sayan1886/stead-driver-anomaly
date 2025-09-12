
# STEAD Driver Behaviour Anomaly Detection

This repo implements a simplified version of STEAD (2025) for driver behaviour anomaly detection in Indian urban traffic.

## Features
- 2+1D convolution + temporal attention architecture
- Dashcam / in-car camera dataset support
- Sliding window extraction for long trips
- Training & evaluation scripts
- Lightweight and modular

## Setup
```
pip install -r requirements.txt
```

## Data
Place dashcam videos in `data/train/`, `data/val/`, `data/test/`.

## Train
```
python train.py
```

## Evaluate
```
python eval.py
```
