#!/bin/bash

# Optimized Multi-Video Inference for Full SFT Models
# Designed for models trained with finetune_multivideo.sh (no LoRA)

# Configuration - EDIT THESE PATHS
MODEL_PATH="output/Qwen2.5-VL-3B-Instruct-multivideo_training"
TEST_DATA_PATH="/oak/stanford/groups/euan/users/masadi/stanford_echo/lv_vqa_llm_only_video_test.json"
VIDEO_FOLDER="/"
OUTPUT_PATH="inference_sft_results_$(date +%Y%m%d_%H%M%S).json"

echo "=== Optimized SFT Model Inference ==="
echo "Model: $MODEL_PATH"
echo "Test Data: $TEST_DATA_PATH"
echo "Video Folder: $VIDEO_FOLDER"
echo "Output: $OUTPUT_PATH"
echo "====================================="

# Speed optimizations
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_DISABLE_TELEMETRY=1

echo "⚡ Optimizations enabled"
echo "🚀 Starting SFT inference..."

# Use 4-bit quantization for much faster loading
python inference_sft.py \
    --model_path "$MODEL_PATH" \
    --test_data_path "$TEST_DATA_PATH" \
    --video_folder "$VIDEO_FOLDER" \
    --output_path "$OUTPUT_PATH" \
    \
    `# Speed optimization - Use 4-bit quantization` \
    --load_4bit \
    \
    `# Video settings (match training exactly)` \
    --fps 1 \
    --video_max_pixels 4096 \
    --video_resized_width 64 \
    --video_resized_height 64 \
    \
    `# Generation settings` \
    --max_new_tokens 512 \
    --temperature 0.7 \
    --top_p 0.9 \
    --do_sample \
    \
    `# Performance` \
    --device cuda \
    --verbose

echo ""
echo "✅ SFT inference completed!"
echo "📊 Results saved to: $OUTPUT_PATH"
echo ""
echo "To analyze results:"
echo "python analyze_results.py --results_path $OUTPUT_PATH --show_examples 5"

# Optional: Show quick performance summary
if [ -f "$OUTPUT_PATH" ]; then
    echo ""
    echo "📈 Quick Performance Summary:"
    python -c "
import json
with open('$OUTPUT_PATH') as f:
    data = json.load(f)
meta = data['metadata']
print(f'  ⏱️  Load time: {meta[\"load_time_seconds\"]:.1f}s')
print(f'  🔄 Inference time: {meta[\"inference_time_seconds\"]:.1f}s')  
print(f'  📈 Speed: {meta[\"samples_per_second\"]:.2f} samples/sec')
print(f'  ✅ Success: {meta[\"successful_samples\"]}/{meta[\"total_samples\"]}')
"
fi