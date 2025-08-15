#!/bin/bash

# Multi-Video Training Script for Qwen2.5-VL
# Optimized for datasets with 30+ videos per sample
# 
# This script handles the memory and processing requirements for training
# with large numbers of videos per conversation sample.

MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"  # Use if you have more VRAM

export PYTHONPATH=src:$PYTHONPATH

# BATCH SIZE CONFIGURATION FOR MULTI-VIDEO
# With 30 videos per sample, memory usage is significantly higher
GLOBAL_BATCH_SIZE=16  # Reduced from 128 for multi-video
BATCH_PER_DEVICE=1    # Keep at 1 for 30+ videos  
NUM_DEVICES=8         # Adjust based on your setup
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

echo "=== Multi-Video Training Configuration ==="
echo "Model: $MODEL_NAME"
echo "Global Batch Size: $GLOBAL_BATCH_SIZE"
echo "Batch Per Device: $BATCH_PER_DEVICE"
echo "Gradient Accumulation Steps: $GRAD_ACCUM_STEPS"
echo "=========================================="

deepspeed src/train/train_sft.py \
    --use_liger True \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path /path/to/your/multivideo/data.json \
    --image_folder /path/to/your/video/folder \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm False \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/multivideo_training \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    \
    `# VIDEO PROCESSING SETTINGS FOR 30+ VIDEOS` \
    --video_max_pixels $((64 * 64)) \
    --video_resized_width 64 \
    --video_resized_height 64 \
    --fps 0.25 \
    \
    `# MEMORY OPTIMIZATION` \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    \
    `# LEARNING RATES` \
    --learning_rate 1e-5 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    \
    `# LOGGING AND SAVING` \
    --logging_steps 1 \
    --tf32 True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 3

echo "Training completed!"