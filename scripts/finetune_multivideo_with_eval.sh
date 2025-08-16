#!/bin/bash

# Multi-Video Training Script with Live Evaluation
# Optimized for datasets with 30+ videos per sample + evaluation during training
# 
# This script handles training with live evaluation metrics

MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"  # Use if you have more VRAM

export PYTHONPATH=src:$PYTHONPATH

# BATCH SIZE CONFIGURATION FOR MULTI-VIDEO
# With 30 videos per sample, memory usage is significantly higher
GLOBAL_BATCH_SIZE=16  # Reduced from 128 for multi-video
BATCH_PER_DEVICE=1    # Keep at 1 for 30+ videos  
NUM_DEVICES=8         # Adjust based on your setup
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

echo "=== Multi-Video Training with Live Evaluation ==="
echo "Model: $MODEL_NAME"
echo "Global Batch Size: $GLOBAL_BATCH_SIZE"
echo "Batch Per Device: $BATCH_PER_DEVICE"
echo "Gradient Accumulation Steps: $GRAD_ACCUM_STEPS"
echo "=================================================="

deepspeed src/train/train_sft.py \
    --use_liger True \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    \
    `# TRAINING DATA` \
    --data_path /path/to/your/train_dataset.json \
    --image_folder /path/to/your/video/folder \
    \
    `# EVALUATION DATA - ADD THESE LINES` \
    --eval_path /path/to/your/eval_dataset.json \
    --eval_image_folder /path/to/your/eval/video/folder \
    \
    `# EVALUATION SETTINGS` \
    --do_eval True \
    --evaluation_strategy steps \
    --eval_steps 50 \
    --eval_accumulation_steps 1 \
    --per_device_eval_batch_size 1 \
    \
    `# MODEL SETTINGS` \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm False \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    \
    `# OUTPUT SETTINGS` \
    --output_dir output/multivideo_training_with_eval \
    --run_name "multivideo_training_eval" \
    \
    `# TRAINING SCHEDULE` \
    --num_train_epochs 3 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    \
    `# VIDEO PROCESSING SETTINGS FOR 30+ VIDEOS` \
    --video_max_pixels $((64 * 64)) \
    --video_resized_width 64 \
    --video_resized_height 64 \
    --fps 1 \
    \
    `# MEMORY & PERFORMANCE OPTIMIZATIONS` \
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
    `# LOGGING AND SAVING WITH EVALUATION` \
    --logging_steps 10 \
    --logging_strategy steps \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 3 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    \
    `# REPORTING` \
    --tf32 True \
    --report_to tensorboard \
    --lazy_preprocess True

echo ""
echo "Training with evaluation completed!"
echo ""
echo "📊 To view live metrics:"
echo "tensorboard --logdir output/multivideo_training_with_eval/runs"
echo ""
echo "📈 Evaluation results are logged every 50 steps"
echo "📁 Best model saved based on eval_loss"