#!/bin/bash

# Working Multi-Video Training Script with Evaluation
# This version addresses all the argument parsing issues systematically

MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

# BATCH SIZE CONFIGURATION FOR MULTI-VIDEO
GLOBAL_BATCH_SIZE=16
BATCH_PER_DEVICE=1
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

echo "=== Working Multi-Video Training with Evaluation ==="
echo "Model: $MODEL_NAME"
echo "Global Batch Size: $GLOBAL_BATCH_SIZE"
echo "Batch Per Device: $BATCH_PER_DEVICE"
echo "Gradient Accumulation Steps: $GRAD_ACCUM_STEPS"
echo "====================================================="

# Note: Using explicit TRUE/FALSE values instead of boolean flags
# This addresses the custom TrainingArguments parsing issues

deepspeed src/train/train_sft.py \
    --use_liger True \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    \
    `# TRAINING DATA` \
    --data_path /oak/stanford/groups/euan/users/masadi/stanford_echo/lv_vqa_llm_only_video_train.json \
    --image_folder / \
    \
    `# EVALUATION DATA` \
    --eval_path /oak/stanford/groups/euan/users/masadi/stanford_echo/lv_vqa_llm_only_video_test.json \
    --eval_image_folder / \
    \
    `# EVALUATION SETTINGS - Using working pattern from cls script` \
    --eval_strategy steps \
    --eval_steps 50 \
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
    `# VIDEO PROCESSING SETTINGS` \
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
    `# LOGGING AND SAVING - Using working pattern from cls script` \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 3 \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_loss" \
    --greater_is_better False \
    \
    `# REPORTING` \
    --tf32 True \
    --report_to wandb \
    --lazy_preprocess True

echo ""
echo "Training with evaluation completed!"
echo ""
echo "📊 Check your Wandb dashboard for live metrics"
echo "📈 Evaluation results logged every 50 steps"
echo "📁 Model checkpoints saved every 50 steps"