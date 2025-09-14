#!/bin/bash
# EC2 Deployment Script for CLS + MaxPool Ensemble Training
# Usage: ./deploy_ec2.sh [dataset_path]

set -e

echo "🚀 EC2 Training Deployment Setup"
echo "================================="

DATASET_PATH=${1:-"../datasets/datasets_full.csv"}

echo "📋 Deployment Options:"
echo "1. 🐳 Docker (Recommended for EC2)"
echo "2. 📝 Poetry (Local development)"
echo "3. 🐍 Native Python"
echo ""

# Make scripts executable
chmod +x launch_training.sh

echo "🔧 Setup Instructions for EC2:"
echo ""
echo "1. 📦 Transfer files to EC2:"
echo "   scp -r . ec2-user@your-instance:/home/ec2-user/realfeel-training/"
echo ""
echo "2. 📊 Dataset is included in project:"
echo "   📁 ../datasets/datasets_full.csv (existing dataset)"
echo "   💡 Or upload custom dataset: scp $DATASET_PATH ec2-user@your-instance:/home/ec2-user/realfeel-training/datasets/"
echo ""
echo "3. 🐳 For Docker execution (recommended):"
echo "   ssh ec2-user@your-instance"
echo "   cd /home/ec2-user/realfeel-training"
echo "   ./launch_training.sh docker production ../datasets/datasets_full.csv ./trained_models"
echo ""
echo "4. 📝 For Poetry execution:"
echo "   ssh ec2-user@your-instance"
echo "   cd /home/ec2-user/realfeel-training"
echo "   ./launch_training.sh poetry production ../datasets/datasets_full.csv ./trained_models"
echo ""
echo "5. 📈 Monitor training:"
echo "   tail -f /home/ec2-user/realfeel-training/trained_models/training_*.log"
echo ""

echo "⚙️  EC2 Instance Requirements:"
echo "  - Instance type: p3.2xlarge or g4dn.xlarge (for GPU)"
echo "  - AMI: Deep Learning AMI (Ubuntu 20.04)"
echo "  - Storage: 100GB+ EBS volume"
echo "  - Security groups: SSH (22) access"
echo ""

echo "📋 Dataset Information:"
echo "  🎯 Using Cresci-2017 Bot Detection Dataset (included in project)"
echo "  📊 Contains: ~37K human tweets, ~100K+ bot tweets"
echo "  🤖 Bot types: Social spambots, Traditional spambots, Fake followers"
echo "  👤 Human tweets: Genuine account activities"
echo "  📁 Automatically loads all categories with proper labels"
echo ""

echo "🎯 Expected training times (Cresci-2017 dataset ~137K samples):"
echo "  - Fast config (testing): ~1 hour"
echo "  - Production config (full): ~4-6 hours"
echo "  - Custom large datasets (1M+): ~12+ hours"
echo ""

echo "💾 Output files will be saved to:"
echo "  - trained_models/ensemble_TIMESTAMP_best.pt"
echo "  - trained_models/training_results.json"
echo "  - trained_models/training_TIMESTAMP.log"
echo ""

echo "✅ Deployment preparation complete!"
echo "   Ready to transfer to EC2 and start training."