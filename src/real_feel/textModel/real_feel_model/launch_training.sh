#!/bin/bash
# Production Training Launcher for EC2 (Poetry + Docker)
# Usage: ./launch_training.sh [EXECUTION_MODE] [CONFIG] [DATA_PATH] [OUTPUT_DIR]
# EXECUTION_MODE: docker | poetry | native

set -e  # Exit on any error

echo "🚀 CLS + MaxPool Ensemble Training Launcher"
echo "============================================"

# Parse arguments
EXECUTION_MODE=${1:-docker}
CONFIG=${2:-production}
DATA_PATH=${3:-"../datasets/datasets_full.csv"}
OUTPUT_DIR=${4:-"./trained_models"}
GPU_ID=${5:-0}

# Get absolute paths
DATA_PATH=$(realpath "$DATA_PATH" 2>/dev/null || echo "$DATA_PATH")
OUTPUT_DIR=$(realpath "$OUTPUT_DIR" 2>/dev/null || mkdir -p "$OUTPUT_DIR" && realpath "$OUTPUT_DIR")

echo "📊 System Information:"
echo "  Execution Mode: $EXECUTION_MODE"
echo "  Config: $CONFIG"
echo "  Dataset: $DATA_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  GPU ID: $GPU_ID"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits | head -1
    GPU_AVAILABLE=true
else
    echo "  GPU: Not available (using CPU)"
    GPU_AVAILABLE=false
fi

# Validate dataset
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Error: Dataset not found at $DATA_PATH"
    echo "Usage: ./launch_training.sh [docker|poetry|native] [config] [data_path] [output_dir] [gpu_id]"
    exit 1
fi

DATASET_SIZE=$(wc -l < "$DATA_PATH" 2>/dev/null || echo "unknown")
echo "  Dataset size: $DATASET_SIZE lines"

echo ""
echo "🏋️‍♀️ Starting training with $EXECUTION_MODE..."
echo "============================================"

case $EXECUTION_MODE in
    "docker")
        echo "🐳 Using Docker execution..."

        # Build Docker image if it doesn't exist
        if ! docker images | grep -q "realfeel-training"; then
            echo "Building Docker image..."
            docker build -f Dockerfile.training -t realfeel-training ../../../../
        fi

        # Prepare Docker run command
        DOCKER_CMD="docker run --rm"

        # Add GPU support if available
        if [ "$GPU_AVAILABLE" = true ]; then
            DOCKER_CMD="$DOCKER_CMD --gpus all"
        fi

        # Add volume mounts
        DOCKER_CMD="$DOCKER_CMD -v $(dirname $DATA_PATH):/data"
        DOCKER_CMD="$DOCKER_CMD -v $OUTPUT_DIR:/models"
        DOCKER_CMD="$DOCKER_CMD -v $(pwd):/logs"

        # Run training
        $DOCKER_CMD realfeel-training python train_ensemble.py \
            --config "$CONFIG" \
            --data_path "/data/$(basename $DATA_PATH)" \
            --output_dir "/models" \
            --gpu_id "$GPU_ID" \
            --experiment_name "ensemble_$(date +%Y%m%d_%H%M%S)" \
            2>&1 | tee "$OUTPUT_DIR/training_$(date +%Y%m%d_%H%M%S).log"
        ;;

    "poetry")
        echo "📝 Using Poetry execution..."

        # Check if we're in the right directory for Poetry
        if [ ! -f "../../../../pyproject.toml" ]; then
            echo "❌ Error: pyproject.toml not found. Run from the model directory."
            exit 1
        fi

        # Navigate to project root for Poetry
        cd ../../../../

        # Run with Poetry
        poetry run python src/real_feel/textModel/real_feel_model/train_ensemble.py \
            --config "$CONFIG" \
            --data_path "$DATA_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --gpu_id "$GPU_ID" \
            --experiment_name "ensemble_$(date +%Y%m%d_%H%M%S)" \
            2>&1 | tee "$OUTPUT_DIR/training_$(date +%Y%m%d_%H%M%S).log"
        ;;

    "native")
        echo "🐍 Using native Python execution..."

        # Set Python path
        export PYTHONPATH="${PYTHONPATH}:$(realpath ../../../../src)"

        # Run directly with Python
        python train_ensemble.py \
            --config "$CONFIG" \
            --data_path "$DATA_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --gpu_id "$GPU_ID" \
            --experiment_name "ensemble_$(date +%Y%m%d_%H%M%S)" \
            2>&1 | tee "$OUTPUT_DIR/training_$(date +%Y%m%d_%H%M%S).log"
        ;;

    *)
        echo "❌ Error: Invalid execution mode '$EXECUTION_MODE'"
        echo "Valid modes: docker, poetry, native"
        exit 1
        ;;
esac

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "🎉 Training completed successfully!"
    echo "📁 Results saved to: $OUTPUT_DIR"
    echo "💾 Model ready for deployment"
else
    echo ""
    echo "❌ Training failed with exit code $EXIT_CODE"
    echo "📋 Check the log file for details"
    exit $EXIT_CODE
fi