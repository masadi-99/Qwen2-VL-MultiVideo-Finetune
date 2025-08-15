#!/bin/bash

# Multi-Video Inference Script for Qwen2.5-VL
# Usage example for running inference on test datasets

# Configuration
MODEL_PATH="output/multivideo_training"  # Path to your fine-tuned model
# MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"  # Or use base model for comparison

TEST_DATA_PATH="/path/to/your/test_dataset.json"
VIDEO_FOLDER="/path/to/your/video/folder"
OUTPUT_PATH="inference_results_$(date +%Y%m%d_%H%M%S).json"

echo "=== Multi-Video Inference ==="
echo "Model: $MODEL_PATH"
echo "Test Data: $TEST_DATA_PATH"
echo "Video Folder: $VIDEO_FOLDER"
echo "Output: $OUTPUT_PATH"
echo "=========================="

python inference.py \
    --model_path "$MODEL_PATH" \
    --test_data_path "$TEST_DATA_PATH" \
    --video_folder "$VIDEO_FOLDER" \
    --output_path "$OUTPUT_PATH" \
    \
    `# Video processing settings (match your training settings)` \
    --video_max_pixels $((64 * 64)) \
    --video_resized_width 64 \
    --video_resized_height 64 \
    --fps 0.25 \
    \
    `# Generation settings` \
    --max_new_tokens 512 \
    --temperature 0.7 \
    --top_p 0.9 \
    --do_sample \
    \
    `# Other settings` \
    --device auto \
    --batch_size 1 \
    --verbose

echo ""
echo "Inference completed! Results saved to: $OUTPUT_PATH"
echo ""
echo "To analyze results, you can use:"
echo "python analyze_results.py --results_path $OUTPUT_PATH"