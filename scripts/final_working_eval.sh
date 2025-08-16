#!/bin/bash

# FINAL WORKING Multi-Video Training with Evaluation
# Uses CLSArguments (proven to work) and exact parameter names from cls script

MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

# BATCH SIZE CONFIGURATION
GLOBAL_BATCH_SIZE=16
BATCH_PER_DEVICE=1
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

echo "=== FINAL WORKING Multi-Video Training with Evaluation ==="
echo "Model: $MODEL_NAME"
echo "Using CLSArguments (proven working class)"
echo "Global Batch Size: $GLOBAL_BATCH_SIZE"
echo "=========================================================="

deepspeed src/train/train_sft.py \
    --use_liger True \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    \
    `# DATA PATHS` \
    --data_path /oak/stanford/groups/euan/users/masadi/stanford_echo/lv_vqa_llm_only_video_train.json \
    --image_folder / \
    --eval_path /oak/stanford/groups/euan/users/masadi/stanford_echo/lv_vqa_llm_only_video_test.json \
    --eval_image_folder / \
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
    `# TRAINING CONFIG` \
    --output_dir output/multivideo_training_final \
    --run_name "multivideo_final_eval" \
    --num_train_epochs 3 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    \
    `# VIDEO SETTINGS` \
    --video_max_pixels $((64 * 64)) \
    --video_resized_width 64 \
    --video_resized_height 64 \
    --fps 1 \
    \
    `# EVALUATION - EXACT SAME AS WORKING CLS SCRIPT` \
    --eval_strategy steps \
    --eval_steps 50 \
    --per_device_eval_batch_size 1 \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_loss" \
    --greater_is_better False \
    \
    `# OPTIMIZATION` \
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
    `# LOGGING AND SAVING - MATCH EVAL STRATEGY` \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 3 \
    \
    `# REPORTING` \
    --tf32 True \
    --report_to wandb \
    --lazy_preprocess True

echo ""
echo "✅ FINAL TRAINING COMPLETE!"
echo "📊 Check Wandb dashboard for results"
echo "📁 Best model saved in output/multivideo_training_final/"