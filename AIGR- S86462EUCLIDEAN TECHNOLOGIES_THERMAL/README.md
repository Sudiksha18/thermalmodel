# Thermal Anomaly Detection System
## Euclidean Technologies - Deep Learning Model for High-Accuracy Thermal Anomaly Detection

### Overview
This project implements a state-of-the-art deep learning model for detecting thermal anomalies in satellite and video data using PyTorch. The system is optimized for real-time GPU inference on NVIDIA A100 and focuses on detecting non-natural/manmade anomalies while suppressing natural heat sources.

### Key Features
- **High Accuracy**: Achieves excellent F1, ROC-AUC, and PR-AUC scores
- **Real-time GPU Inference**: Optimized for NVIDIA A100/CUDA with mixed precision
- **Multiple Data Formats**: Supports .tif, .he5, .png images and thermal video streams
- **Transformer Architecture**: Uses Swin-UNet with optional temporal fusion
- **Comprehensive Outputs**: GeoTIFF, PNG overlays, Excel reports, and model hashes

### Architecture
- **Backbone**: Swin Transformer or ConvNeXt for feature extraction
- **Decoder**: U-Net style decoder for pixel-level anomaly segmentation
- **Temporal Fusion**: Optional ConvLSTM for video sequences
- **Anomaly Head**: PatchCore-inspired anomaly scoring

### Dataset Support
- Landsat-8/9 thermal infrared bands
- FLAME-2/3 thermal datasets
- KAIST thermal datasets
- MIRSat-QL thermal data
- Custom thermal .he5 and .tif files

### Quick Start

#### Installation
```bash
pip install -r requirements.txt
```

#### Training
```bash
python main.py --mode train --config config.yaml
```

#### Inference
```bash
python main.py --mode inference --input_path data/thermal_image.tif --model_path outputs/models/best_model.pth
```

#### Batch Processing
```bash
python main.py --mode batch_inference --input_dir data/test_images/ --output_dir submission/
```

### Output Files
All outputs follow the Stage-1 submission format:
- `EuclideanTechnologies_AnomalyMap.tif` - GeoTIFF anomaly map
- `EuclideanTechnologies_AnomalyMap.png` - Visualization overlay
- `EuclideanTechnologies_Metrics.xlsx` - Performance metrics
- `EuclideanTechnologies_ModelHash.txt` - Model SHA-256 hash
- `README_Submission.txt` - Submission summary

### Performance Targets
- **Accuracy**: >95%
- **F1 Score**: >0.90
- **ROC-AUC**: >0.95
- **PR-AUC**: >0.90
- **Inference Speed**: >30 FPS on A100

### Hardware Requirements
- NVIDIA A100 80GB (recommended)
- CUDA 11.8+
- 32GB+ RAM
- 100GB+ storage

### Directory Structure
```
thermal-anomaly-detection/
├── data/                    # Datasets and splits
├── src/                     # Source code modules
├── outputs/                 # Model outputs and logs
├── notebooks/               # Jupyter notebooks for analysis
├── scripts/                 # Utility scripts
└── submission/              # Final submission outputs
```

### License
MIT License - See LICENSE file for details.

### Contact
Euclidean Technologies
Email: contact@euclideantech.ai