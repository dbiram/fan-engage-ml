# Fan Engage ML

Machine learning experiments and training code for the Fan Engage project.

## Overview

This repository contains scripts and utilities for training, evaluating, and experimenting with computer vision models for football (soccer) analytics, including:

- **Object detection** (players, ball) using YOLOv8
- **Pose/keypoint detection** for pitch and players
- **Team clustering** and feature extraction prototypes

## Directory Structure

- `train_yolov8.py` — Script for fine-tuning YOLOv8 models on football datasets, with MinIO integration for artifact storage.
- `pose/` — Training and utilities for pitch keypoint detection (e.g., `train_pitch_kpts.py`).
- `prototypes/` — Experimental scripts, e.g., `team_clustering.py` for unsupervised team assignment and feature analysis.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Key packages:
- ultralytics
- pandas, numpy, pyarrow
- opencv-python-headless, Pillow
- minio
- scikit-learn, umap-learn
- tqdm, open-clip-torch

## Usage

### Training YOLOv8

```bash
python train_yolov8.py --data /path/to/data.yaml --model yolov8n.pt --epochs 50 --batch 16
```

See `python train_yolov8.py --help` for all options, including MinIO artifact upload.

### Training Pitch Keypoints

```bash
python pose/train_pitch_kpts.py --data /path/to/data.yaml --model yolov8s-pose.pt --epochs 50
```

### Team Clustering Prototype

```bash
python prototypes/team_clustering.py
```

## Docker

A `Dockerfile` is provided for reproducible training environments (with PyTorch, CUDA, and all dependencies).

## Notes

- Data and model artifacts are typically stored in MinIO buckets.
- Scripts are designed for modular experimentation and can be adapted for new datasets or tasks.