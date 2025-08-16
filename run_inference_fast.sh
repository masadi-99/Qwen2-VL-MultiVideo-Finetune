#!/bin/bash

# Fast Multi-Video Inference Script - Optimized for speed
# This version includes several optimizations for faster model loading

# Configuration - EDIT THESE PATHS
MODEL_PATH="output/Qwen2.5-VL-3B-Instruct-multivideo_training"
MODEL_BASE="Qwen/Qwen2.5-VL-3B-Instruct"

TEST_DATA_PATH="/oak/stanford/groups/euan/users/masadi/stanford_echo/lv_vqa_llm_only_video_test.json"
VIDEO_FOLDER="/"
OUTPUT_PATH="inference_results_fast_$(date +%Y%m%d_%H%M%S).json"

echo "=== Fast Multi-Video Inference ==="
echo "Model Path: $MODEL_PATH"
echo "Model Base: $MODEL_BASE"
echo "Test Data: $TEST_DATA_PATH"
echo "Output: $OUTPUT_PATH"
echo "=================================="

# Speed optimizations via environment variables
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_OFFLINE=0
export HF_HUB_DISABLE_TELEMETRY=1

# Optional: Use specific CUDA device for consistency
# export CUDA_VISIBLE_DEVICES=0

echo "⚡ Speed optimizations enabled"
echo "🚀 Starting fast inference..."

python inference_fast.py \
    --model_path "$MODEL_PATH" \
    --model_base "$MODEL_BASE" \
    --test_data_path "$TEST_DATA_PATH" \
    --video_folder "$VIDEO_FOLDER" \
    --output_path "$OUTPUT_PATH" \
    \
    `# Speed optimizations` \
    --fast_loading \
    \
    `# Video settings` \
    --fps 0.25 \
    \
    `# Generation settings (optimized for speed)` \
    --max_new_tokens 256 \
    --temperature 0.7 \
    --do_sample \
    \
    `# Performance settings` \
    --device cuda \
    --verbose

echo ""
echo "✅ Fast inference completed!"
echo "📊 Results saved to: $OUTPUT_PATH"
echo ""
echo "To analyze results:"
echo "python analyze_results.py --results_path $OUTPUT_PATH --show_examples 3"