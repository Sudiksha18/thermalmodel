EUCLIDEAN TECHNOLOGIES THERMAL ANOMALY DETECTION OUTPUTS
========================================================

Generated: 2025-10-20 12:34:58

FOLDER STRUCTURE:
1_HashValue/ - Model verification hash
2_AnomalyDetectionResults/ - Boolean GeoTIFF (0/1) and PNG visualization  
3_AccuracyReport/ - Performance metrics CSV
4_ModelDocumentation/ - Detailed model report

OUTPUT FORMAT:
- GeoTIFF: Boolean anomaly map (0=normal, 1=anomaly)
- Detected anomalies: 39106 pixels
- Total pixels: 58455691 pixels
- Anomaly percentage: 0.07%

TECHNICAL SPECS:
- Input: Landsat 8 thermal infrared
- Model: Swin Transformer + U-Net
- Output: Boolean detection (0/1)
- Framework: PyTorch + CUDA
