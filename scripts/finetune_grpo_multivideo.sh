#!/bin/bash

# Multi-Video GRPO Training Script for Numerical Video Analysis
# Optimized for 30+ videos per sample with custom numerical reward function
#
# This script uses GRPO (Group Relative Policy Optimization) with a custom reward
# function that evaluates numerical answers in video analysis tasks.

MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"  # Use if you have more VRAM

export PYTHONPATH=src:$PYTHONPATH

# BATCH SIZE CONFIGURATION FOR MULTI-VIDEO GRPO
# GRPO requires generating multiple samples per prompt, memory usage is high
GLOBAL_BATCH_SIZE=16   # Updated to match user's settings
BATCH_PER_DEVICE=2     # Updated to match user's settings
NUM_DEVICES=8          # Adjust based on your setup
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

echo "=== Multi-Video GRPO Training with Numerical Rewards ==="
echo "Model: $MODEL_NAME"
echo "Global Batch Size: $GLOBAL_BATCH_SIZE"
echo "Batch Per Device: $BATCH_PER_DEVICE"
echo "Gradient Accumulation Steps: $GRAD_ACCUM_STEPS"
echo "Reward Function: numerical_video_reward"
echo "========================================================="

deepspeed src/train/train_grpo_multivideo.py \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    \
    `# TRAINING DATA` \
    --data_path /oak/stanford/groups/euan/users/masadi/stanford_echo/lv_vqa_llm_only_video_train.json \
    --image_folder / \
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
    --output_dir output/grpo_multivideo_numerical \
    --run_name "grpo_multivideo_numerical" \
    \
    `# GRPO-SPECIFIC TRAINING SCHEDULE` \
    --num_train_epochs 3 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    \
    `# GRPO ALGORITHM PARAMETERS` \
    --num_iterations 1 \
    --epsilon 0.05 \
    --epsilon_high 0.1 \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_completion_length 512 \
    --num_generations 2 \
    \
    `# VIDEO PROCESSING SETTINGS FOR 30+ VIDEOS` \
    --video_max_pixels $((84 * 84)) \
    --video_resized_width 84 \
    --video_resized_height 84 \
    --fps 1 \
    \
    `# MEMORY & PERFORMANCE OPTIMIZATIONS` \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    \
    `# LEARNING RATES (Conservative for GRPO)` \
    --learning_rate 5e-6 \
    --merger_lr 5e-6 \
    --vision_lr 1e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    \
    `# LOGGING AND SAVING` \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 3 \
    \
    `# REPORTING AND DEBUGGING` \
    --tf32 True \
    --report_to wandb \
    --lazy_preprocess True

echo ""
echo "✅ GRPO training with numerical rewards completed!"
echo ""
echo "🎯 Custom Reward Function Features:"
echo "  - Numerical extraction from text (handles percentages)"
echo "  - RMSE-based scoring (1.0 for exact, decreases with error)"
echo "  - Range emphasis (higher rewards for correct answers far from 50)"
echo "  - Neutral rewards (0.5) for non-numerical questions"
echo ""
echo "📊 To view training metrics:"
echo "  - Wandb dashboard: https://wandb.ai"
echo "  - Local logs: output/grpo_multivideo_numerical/"
echo ""
echo "🏆 Model optimized for numerical video analysis tasks!"