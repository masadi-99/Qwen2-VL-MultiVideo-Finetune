#!/bin/bash

# Wandb Setup and Training Script
# Complete setup for multi-video training with Wandb monitoring

echo "=== Wandb Multi-Video Training Setup ==="

# Check if wandb is installed
if ! command -v wandb &> /dev/null; then
    echo "📦 Installing Wandb..."
    pip install wandb
fi

# Login to wandb (if not already logged in)
echo "🔐 Setting up Wandb authentication..."
echo "If you haven't logged in to Wandb yet, run:"
echo "wandb login"
echo ""
echo "Or set your API key as environment variable:"
echo "export WANDB_API_KEY=your_api_key_here"
echo ""

# Set Wandb project name
read -p "Enter your Wandb project name (default: qwen2-5-vl-multivideo): " WANDB_PROJECT
WANDB_PROJECT=${WANDB_PROJECT:-"qwen2-5-vl-multivideo"}

read -p "Enter your Wandb run name (default: multivideo-training-eval): " WANDB_RUN_NAME
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"multivideo-training-eval"}

echo ""
echo "🚀 Starting training with Wandb monitoring..."
echo "Project: $WANDB_PROJECT"
echo "Run name: $WANDB_RUN_NAME"
echo "Dashboard: https://wandb.ai"
echo ""

# Export Wandb settings
export WANDB_PROJECT=$WANDB_PROJECT
export WANDB_RUN_NAME=$WANDB_RUN_NAME

# Run the training script with Wandb
bash scripts/your_training_with_eval.sh