#!/bin/bash
# Inference script for Euclidean Technologies Thermal Anomaly Detection
# Batch processing for multiple thermal images

echo "==================================================================="
echo "Euclidean Technologies - Thermal Anomaly Detection Inference"
echo "==================================================================="

# Default parameters
INPUT_DIR="data/"
OUTPUT_DIR="submission/"
MODEL_PATH="outputs/models/best_model.pth"
BATCH_SIZE=16
THRESHOLD=0.5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --input-dir DIR      Input directory with thermal images (default: data/)"
            echo "  --output-dir DIR     Output directory for results (default: submission/)"
            echo "  --model-path PATH    Path to trained model (default: outputs/models/best_model.pth)"
            echo "  --batch-size SIZE    Batch size for inference (default: 16)"
            echo "  --threshold THRESH   Anomaly threshold (default: 0.5)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate inputs
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' not found!"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file '$MODEL_PATH' not found!"
    echo "Please train a model first or provide a valid model path."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Count input files
THERMAL_FILES=$(find "$INPUT_DIR" -name "*.tif" -o -name "*.he5" -o -name "*.png" | wc -l)

echo "Configuration:"
echo "  Input directory: $INPUT_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Model path: $MODEL_PATH"
echo "  Batch size: $BATCH_SIZE"
echo "  Threshold: $THRESHOLD"
echo "  Thermal files found: $THERMAL_FILES"
echo ""

if [ "$THERMAL_FILES" -eq 0 ]; then
    echo "Warning: No thermal files found in input directory!"
    echo "Supported formats: .tif, .he5, .png"
    exit 1
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

echo "Starting batch inference..."
echo "Expected processing time: ~2-5 seconds per image on A100"
echo ""

# Run batch inference
python main.py \
    --mode batch_inference \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_path "$MODEL_PATH" \
    --batch_size "$BATCH_SIZE" \
    --threshold "$THRESHOLD" \
    --device cuda

INFERENCE_EXIT_CODE=$?

echo ""
if [ $INFERENCE_EXIT_CODE -eq 0 ]; then
    echo "Inference completed successfully!"
    echo ""
    echo "Generated outputs:"
    echo "  Anomaly maps: $OUTPUT_DIR/EuclideanTechnologies_AnomalyMap.*"
    echo "  Metrics report: $OUTPUT_DIR/EuclideanTechnologies_Metrics.xlsx"
    echo "  Model hash: $OUTPUT_DIR/EuclideanTechnologies_ModelHash.txt"
    echo "  Submission README: $OUTPUT_DIR/README_Submission.txt"
    echo ""
    echo "Ready for Stage-1 submission!"
else
    echo "Inference failed with exit code: $INFERENCE_EXIT_CODE"
    echo "Check logs for details."
    exit $INFERENCE_EXIT_CODE
fi