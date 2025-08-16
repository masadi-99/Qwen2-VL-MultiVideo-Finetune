# Live Evaluation During Training

This guide shows how to add evaluation datasets to monitor your model's performance live during training.

## 🎯 What You Get

- ✅ **Live evaluation metrics** during training
- ✅ **Automatic best model saving** based on eval performance  
- ✅ **TensorBoard logging** for real-time monitoring
- ✅ **Early stopping** capabilities
- ✅ **Multi-video evaluation** support

## 📊 Quick Setup

### 1. Prepare Your Evaluation Dataset

Create an evaluation dataset in the same format as your training data:

```json
[
  {
    "id": "eval_sample_1",
    "video": [
      "eval_video_001.mp4",
      "eval_video_002.mp4", 
      "eval_video_003.mp4"
    ],
    "conversations": [
      {
        "from": "human",
        "value": "<video>\n<video>\n<video>\nWhat do you see in these videos?"
      },
      {
        "from": "assistant",
        "value": "In these videos, I can see..."
      }
    ]
  }
]
```

**Recommended eval dataset size:**
- Small: 50-100 samples (fast evaluation)
- Medium: 200-500 samples (balanced)
- Large: 1000+ samples (comprehensive)

### 2. Use the Enhanced Training Script

```bash
# Copy and edit the evaluation training script
cp scripts/finetune_multivideo_with_eval.sh scripts/my_training_with_eval.sh

# Edit the paths:
nano scripts/my_training_with_eval.sh
```

**Key parameters to set:**
```bash
# Training data
--data_path /path/to/your/train_dataset.json
--image_folder /path/to/your/train/videos/

# Evaluation data  
--eval_path /path/to/your/eval_dataset.json
--eval_image_folder /path/to/your/eval/videos/  # Can be same as training

# Evaluation frequency
--eval_steps 50  # Evaluate every 50 training steps
```

### 3. Run Training with Evaluation

```bash
bash scripts/my_training_with_eval.sh
```

## 📈 Monitoring Live Results

### TensorBoard (Recommended)

```bash
# In a separate terminal, start TensorBoard
tensorboard --logdir output/multivideo_training_with_eval/runs

# Open in browser: http://localhost:6006
```

**Key metrics to watch:**
- `eval/loss` - Lower is better
- `train/loss` - Should decrease over time  
- `eval/runtime` - Time per evaluation
- `learning_rate` - LR schedule

### Console Output

During training, you'll see:
```
***** Running Evaluation *****
  Num examples = 100
  Batch size = 1
{'eval_loss': 1.234, 'eval_runtime': 45.67, 'eval_samples_per_second': 2.19, 'epoch': 0.5}
```

## ⚙️ Configuration Options

### Evaluation Frequency

```bash
# Evaluate every N steps
--evaluation_strategy steps --eval_steps 50

# Evaluate every epoch  
--evaluation_strategy epoch
```

### Best Model Saving

```bash
# Save best model based on eval_loss (lower is better)
--load_best_model_at_end True
--metric_for_best_model eval_loss
--greater_is_better False

# Or save best based on custom metric
--metric_for_best_model eval_accuracy
--greater_is_better True
```

### Memory Optimization for Evaluation

```bash
# Smaller eval batch size for memory
--per_device_eval_batch_size 1

# Accumulate gradients for evaluation (if needed)
--eval_accumulation_steps 1
```

## 🔧 Advanced Settings

### Early Stopping

```bash
# Stop if eval performance doesn't improve
--early_stopping_patience 3
--early_stopping_threshold 0.01
```

### Custom Evaluation Intervals

```bash
# More frequent early on, less frequent later
--eval_steps 25              # First 1000 steps: every 25 steps
--eval_steps_schedule "25:1000,50:2000,100"  # Custom schedule
```

### Evaluation-Only Mode

```bash
# Skip training, just run evaluation
--do_train False
--do_eval True
```

## 📊 Sample Evaluation Output

```bash
=== Multi-Video Training with Live Evaluation ===
Model: Qwen/Qwen2.5-VL-3B-Instruct
Global Batch Size: 16
==================================================

Loading evaluation dataset from: /path/to/eval.json
Evaluation dataset loaded: 150 samples

***** Running training *****
  Num examples = 1000
  Num Epochs = 3
  Instantaneous batch size per device = 1
  Gradient Accumulation steps = 2
  Total train batch size = 16

Step 50:
{'train_loss': 2.145, 'learning_rate': 9.5e-06, 'epoch': 0.15}

***** Running Evaluation *****
{'eval_loss': 1.876, 'eval_runtime': 23.4, 'eval_samples_per_second': 6.41, 'epoch': 0.15}

Step 100:
{'train_loss': 1.923, 'learning_rate': 9.0e-06, 'epoch': 0.30}

***** Running Evaluation *****
{'eval_loss': 1.654, 'eval_runtime': 22.8, 'eval_samples_per_second': 6.58, 'epoch': 0.30}
📈 New best model! (eval_loss improved from 1.876 to 1.654)
```

## 🎯 Best Practices

### 1. Dataset Balance
- **Training set**: Larger, diverse examples
- **Eval set**: Representative sample, smaller for speed
- **Same distribution**: Ensure eval set represents training data

### 2. Evaluation Frequency
- **Early training**: More frequent (every 25-50 steps)
- **Later training**: Less frequent (every 100-200 steps)
- **Large datasets**: Less frequent to save time

### 3. Video Count Considerations
- **30+ videos per sample**: Use `--per_device_eval_batch_size 1`
- **Mixed video counts**: Ensure eval set has similar distribution
- **Memory constraints**: Reduce eval batch size or accumulation steps

### 4. Metric Selection
- **General training**: Use `eval_loss` (lower is better)
- **Classification**: Use `eval_accuracy` (higher is better)
- **Custom metrics**: Implement in trainer class

## 🐛 Troubleshooting

### "CUDA out of memory" during evaluation
```bash
# Reduce eval batch size
--per_device_eval_batch_size 1

# Use accumulation steps
--eval_accumulation_steps 1

# Skip evaluation temporarily
--do_eval False
```

### Evaluation taking too long
```bash
# Reduce eval dataset size
# Increase eval_steps (evaluate less frequently)
--eval_steps 100

# Use smaller video resolution for eval
--video_max_pixels 2304  # 48x48 instead of 64x64
```

### Best model not saving
```bash
# Check metric name
--metric_for_best_model eval_loss  # Must match logged metric

# Check direction
--greater_is_better False  # For loss metrics
--greater_is_better True   # For accuracy metrics
```

## 📁 File Structure

After training with evaluation, you'll have:

```
output/multivideo_training_with_eval/
├── runs/                          # TensorBoard logs
├── checkpoint-100/               # Regular checkpoints
├── checkpoint-200/
├── pytorch_model.bin             # Best model (based on eval)
├── config.json
├── trainer_state.json           # Contains eval history
└── training_args.bin
```

## 🔄 Converting from Regular Training

To add evaluation to existing training:

1. **Add eval parameters** to your existing script
2. **Create eval dataset** in same format as training
3. **Set evaluation frequency** with `--eval_steps`
4. **Enable best model saving** with `--load_best_model_at_end True`

The evaluation system works with all training methods (SFT, DPO, GRPO) and handles multi-video datasets seamlessly!