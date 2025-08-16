#!/bin/bash

# Working Multi-Video Inference Script 
# Uses the same model loading method as src.serve.app

# Configuration - EDIT THESE PATHS
MODEL_PATH="output/multivideo_training"  # Your checkpoint path
MODEL_BASE="Qwen/Qwen2.5-VL-3B-Instruct"  # Base model (same as training)

TEST_DATA_PATH="/path/to/your/test_dataset.json"
VIDEO_FOLDER="/path/to/your/video/folder"
OUTPUT_PATH="inference_results_working_$(date +%Y%m%d_%H%M%S).json"

echo "=== Working Multi-Video Inference (Using Serving App Method) ==="
echo "Model Path: $MODEL_PATH"
echo "Model Base: $MODEL_BASE"
echo "Test Data: $TEST_DATA_PATH"
echo "Video Folder: $VIDEO_FOLDER" 
echo "Output: $OUTPUT_PATH"
echo "=============================================================="

python inference_working.py \
    --model_path "$MODEL_PATH" \
    --model_base "$MODEL_BASE" \
    --test_data_path "$TEST_DATA_PATH" \
    --video_folder "$VIDEO_FOLDER" \
    --output_path "$OUTPUT_PATH" \
    \
    `# Video processing settings` \
    --fps 0.25 \
    --video_max_pixels $((64 * 64)) \
    --video_resized_width 64 \
    --video_resized_height 64 \
    \
    `# Generation settings` \
    --max_new_tokens 512 \
    --temperature 0.7 \
    --repetition_penalty 1.0 \
    --do_sample \
    \
    `# Device settings` \
    --device cuda \
    --batch_size 1 \
    --verbose

echo ""
echo "✅ Inference completed! Results saved to: $OUTPUT_PATH"
echo ""
echo "To analyze results:"
echo "python analyze_results.py --results_path $OUTPUT_PATH --show_examples 5"