#!/bin/bash
# Training script for Euclidean Technologies Thermal Anomaly Detection
# Optimized for NVIDIA A100 GPU

echo "==================================================================="
echo "Euclidean Technologies - Thermal Anomaly Detection Training"
echo "==================================================================="

# Check if config file exists
if [ ! -f "config.yaml" ]; then
    echo "Error: config.yaml not found!"
    echo "Please create a configuration file before training."
    exit 1
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo "Warning: nvidia-smi not found. Training will use CPU."
fi

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create output directories
mkdir -p outputs/models
mkdir -p outputs/logs
mkdir -p submission

echo "Starting training..."
echo "Configuration: config.yaml"
echo "Mode: GPU-accelerated with mixed precision"
echo "Expected training time: 4-8 hours on A100"
echo ""

# Run training
python main.py \
    --mode train \
    --config config.yaml \
    --device cuda \
    --debug

echo ""
echo "Training completed!"
echo "Check outputs/models/ for saved models"
echo "Check outputs/logs/ for training logs"
echo "Check submission/ for generated outputs"