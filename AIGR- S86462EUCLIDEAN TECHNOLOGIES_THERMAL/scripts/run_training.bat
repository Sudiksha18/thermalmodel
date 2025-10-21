@echo off
REM Training script for Euclidean Technologies Thermal Anomaly Detection (Windows)
REM Optimized for NVIDIA A100 GPU

echo ===================================================================
echo Euclidean Technologies - Thermal Anomaly Detection Training
echo ===================================================================

REM Check if config file exists
if not exist "config.yaml" (
    echo Error: config.yaml not found!
    echo Please create a configuration file before training.
    pause
    exit /b 1
)

REM Check GPU availability
where nvidia-smi >nul 2>&1
if %errorlevel% == 0 (
    echo GPU Information:
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo.
) else (
    echo Warning: nvidia-smi not found. Training will use CPU.
)

REM Set environment variables for optimal performance
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

REM Create output directories
if not exist "outputs\models" mkdir "outputs\models"
if not exist "outputs\logs" mkdir "outputs\logs"
if not exist "submission" mkdir "submission"

echo Starting training...
echo Configuration: config.yaml
echo Mode: GPU-accelerated with mixed precision
echo Expected training time: 4-8 hours on A100
echo.

REM Run training
python main.py ^
    --mode train ^
    --config config.yaml ^
    --device cuda ^
    --debug

echo.
echo Training completed!
echo Check outputs\models\ for saved models
echo Check outputs\logs\ for training logs
echo Check submission\ for generated outputs

pause