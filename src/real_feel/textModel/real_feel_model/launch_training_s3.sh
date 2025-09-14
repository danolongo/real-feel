#!/bin/bash
# EC2 + S3 Training Launcher for CLS + MaxPool Ensemble
# Usage: ./launch_training_s3.sh [EXECUTION_MODE] [CONFIG] [S3_DATA_BUCKET] [OUTPUT_DIR]

set -e

echo "🚀 CLS + MaxPool Ensemble Training Launcher (EC2 + S3)"
echo "======================================================"

# Parse arguments
EXECUTION_MODE=${1:-docker}
CONFIG=${2:-fast}  # Default to fast for quick training
S3_BUCKET=${3:-""}
OUTPUT_DIR=${4:-"./trained_models"}
GPU_ID=${5:-0}

# Local dataset path
LOCAL_DATASET_PATH="../datasets/datasets_full.csv"

echo "📊 Configuration:"
echo "  Execution Mode: $EXECUTION_MODE"
echo "  Config: $CONFIG"
echo "  S3 Bucket: ${S3_BUCKET:-"Using local datasets"}"
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

# Handle S3 dataset download if specified
if [ ! -z "$S3_BUCKET" ]; then
    echo ""
    echo "☁️ Downloading dataset from S3..."

    # Check if AWS CLI is available
    if ! command -v aws &> /dev/null; then
        echo "❌ AWS CLI not found. Please install it:"
        echo "   curl 'https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip' -o 'awscliv2.zip'"
        echo "   unzip awscliv2.zip && sudo ./aws/install"
        exit 1
    fi

    # Create local datasets directory
    mkdir -p ../datasets

    # Download from S3
    echo "📥 Syncing from s3://$S3_BUCKET/datasets/..."
    aws s3 sync "s3://$S3_BUCKET/datasets/" "../datasets/" --recursive --quiet

    if [ $? -eq 0 ]; then
        echo "✅ Dataset downloaded successfully"
        ls -la ../datasets/
    else
        echo "❌ Failed to download dataset from S3"
        exit 1
    fi
fi

# Validate dataset
if [ ! -d "$LOCAL_DATASET_PATH" ]; then
    echo "❌ Error: Dataset not found at $LOCAL_DATASET_PATH"
    echo "Please ensure the dataset is available locally or provide S3 bucket"
    exit 1
fi

echo ""
echo "🏋️‍♀️ Starting training with $EXECUTION_MODE (config: $CONFIG)..."
echo "============================================"

# Set training timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

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
        DOCKER_CMD="$DOCKER_CMD -v $(dirname $(realpath $LOCAL_DATASET_PATH)):/data"
        DOCKER_CMD="$DOCKER_CMD -v $(realpath $OUTPUT_DIR):/models"
        DOCKER_CMD="$DOCKER_CMD -v $(pwd):/logs"

        # Run training
        $DOCKER_CMD realfeel-training python train_ensemble.py \
            --config "$CONFIG" \
            --data_path "/data/datasets_full.csv" \
            --output_dir "/models" \
            --gpu_id "$GPU_ID" \
            --experiment_name "ensemble_${CONFIG}_${TIMESTAMP}" \
            2>&1 | tee "$OUTPUT_DIR/training_${CONFIG}_${TIMESTAMP}.log"
        ;;

    "poetry")
        echo "📝 Using Poetry execution..."

        # Navigate to project root for Poetry
        cd ../../../../

        # Run with Poetry
        poetry run python src/real_feel/textModel/real_feel_model/train_ensemble.py \
            --config "$CONFIG" \
            --data_path "$LOCAL_DATASET_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --gpu_id "$GPU_ID" \
            --experiment_name "ensemble_${CONFIG}_${TIMESTAMP}" \
            2>&1 | tee "$OUTPUT_DIR/training_${CONFIG}_${TIMESTAMP}.log"
        ;;

    *)
        echo "❌ Error: Invalid execution mode '$EXECUTION_MODE'"
        echo "Valid modes: docker, poetry"
        exit 1
        ;;
esac

EXIT_CODE=$?

# Upload results to S3 if bucket specified
if [ $EXIT_CODE -eq 0 ] && [ ! -z "$S3_BUCKET" ]; then
    echo ""
    echo "☁️ Uploading results to S3..."
    aws s3 sync "$OUTPUT_DIR" "s3://$S3_BUCKET/trained_models/" --exclude "*.log" --quiet

    # Upload only the final log
    aws s3 cp "$OUTPUT_DIR/training_${CONFIG}_${TIMESTAMP}.log" \
        "s3://$S3_BUCKET/logs/training_${CONFIG}_${TIMESTAMP}.log" --quiet

    echo "✅ Results uploaded to s3://$S3_BUCKET/trained_models/"
    echo "📋 Log uploaded to s3://$S3_BUCKET/logs/"
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "🎉 Training completed successfully!"
    echo "📁 Results saved to: $OUTPUT_DIR"
    if [ ! -z "$S3_BUCKET" ]; then
        echo "☁️ Backup saved to: s3://$S3_BUCKET/trained_models/"
    fi
    echo "💾 Model ready for deployment"
else
    echo ""
    echo "❌ Training failed with exit code $EXIT_CODE"
    echo "📋 Check the log file for details"
    exit $EXIT_CODE
fi