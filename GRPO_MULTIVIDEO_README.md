# Multi-Video GRPO Training for Qwen2.5-VL

This system enables **GRPO (Group Relative Policy Optimization)** training on datasets with **multiple videos per conversation** (up to 30+ videos) using a **custom numerical reward function**.

## 🎯 Key Features

### ✅ Multi-Video GRPO Support
- **Process up to 5 videos per sample** (GRPO tensor constraint)
- **Automatic video limiting** based on batch processing constraints
- **Memory-optimized processing** for multi-video datasets
- **Compatible with existing GRPO training pipeline**

⚠️ **GRPO Limitation**: Due to tensor dimension constraints in the GRPO trainer architecture, videos are automatically limited to 5 per sample to prevent IndexError. For training with 30+ videos, use SFT instead.

### 🔢 Custom Numerical Reward Function
- **Automatic number extraction** from text (handles percentages like "65.4%")
- **RMSE-based scoring**: 1.0 for exact matches, decreases quadratically with error
- **Range emphasis**: Higher rewards for correct answers far from center (50)
- **Neutral rewards**: 0.5 for non-numerical questions
- **Debug logging** support for reward analysis

## 🛠️ Usage

### 1. Prepare Your Dataset

Your dataset should have this structure for GRPO training:

```json
[
  {
    "id": "sample_1",
    "video": [
      "video1.mp4", 
      "video2.mp4", 
      "video3.mp4"
      // ... up to 30+ videos
    ],
    "conversations": [
      {
        "from": "human", 
        "value": "<video>\n<video>\n<video>\nWhat is the ejection fraction in these videos?"
      },
      {
        "from": "gpt",
        "value": "The ejection fraction is approximately 45.2%"
      }
    ]
  }
]
```

### 2. Run GRPO Training

```bash
# Update paths in the script:
vim scripts/finetune_grpo_multivideo.sh

# Set your data paths:
# --data_path /path/to/your/train_dataset.json
# --image_folder /path/to/your/video/folder

# Run training
bash scripts/finetune_grpo_multivideo.sh
```

## 🎯 Numerical Reward Function Details

### Reward Calculation Logic

```python
def numerical_video_reward(completions, assistant, **kwargs):
    """
    Rewards structure:
    - Ground truth has number: RMSE-based reward (1.0 for exact, decreases with error)
    - Ground truth no number: neutral reward (0.5)
    - Prediction has number when GT has number: base reward (0.1) + RMSE bonus
    - Range emphasis: Numbers far from 50 get higher weight for correct answers
    """
```

### Example Rewards

| Ground Truth | Prediction | Extracted GT | Extracted Pred | Error | Base Reward | RMSE Reward | Final Reward |
|--------------|------------|--------------|----------------|-------|-------------|-------------|--------------|
| "EF is 60%" | "EF is 60%" | 60.0 | 60.0 | 0.0 | 0.1 | 0.9 × 1.17 | **1.0** |
| "EF is 60%" | "EF is 55%" | 60.0 | 55.0 | 5.0 | 0.1 | 0.9 × 0.96 | **0.97** |
| "EF is 30%" | "EF is 30%" | 30.0 | 30.0 | 0.0 | 0.1 | 0.9 × 1.33 | **1.0** |
| "EF is 60%" | "No number" | 60.0 | None | N/A | 0.0 | 0.0 | **0.0** |
| "Good quality" | "Excellent" | None | None | N/A | 0.5 | N/A | **0.5** |

### Range Emphasis

Numbers farther from the center (50) receive bonus rewards:
- **Center (50)**: 1.0× multiplier
- **Edge (20, 80)**: 1.5× multiplier
- **Encourages model to get extreme values correct**

## 📊 Memory Configuration

### For 30 Videos per Sample on 8x H100:

| Configuration | Video Resolution | Memory per Sample | GRPO Samples | Total Memory | Status |
|---------------|------------------|-------------------|--------------|--------------|---------|
| **Recommended** | 84×84 | ~12-15GB | 2-4 samples | ~50-60GB | ✅ **Optimal** |
| Conservative | 64×64 | ~8-12GB | 2-4 samples | ~30-45GB | ✅ **Safe** |
| Aggressive | 112×112 | ~25-30GB | 2-4 samples | ~100+GB | ⚠️ **Risky** |

### Key GRPO Parameters

```bash
# Algorithm settings
--num_iterations 1          # Number of policy updates per batch
--epsilon 0.05              # Lower bound for policy ratio
--epsilon_high 0.1          # Upper bound for policy ratio
--temperature 0.7           # Generation temperature
--max_completion_length 512 # Maximum response length

# Memory settings (reduced for GRPO)
--global_batch_size 8       # Smaller than SFT due to multiple samples
--per_device_train_batch_size 1
--gradient_accumulation_steps 1
```

## 🔬 Technical Implementation

### GRPO Dataset Fix

The original GRPO dataset processed videos individually, causing fps conflicts with 30+ videos. The fix ensures proper path resolution:

```python
# Fixed video processing in src/dataset/grpo_dataset.py
processed_video_files = []
for video_file in video_files:
    if not os.path.exists(video_file):
        if not video_file.startswith("http"):
            video_file = os.path.join(video_folder, video_file)
    processed_video_files.append(video_file)

# Process each video with content structure
for video_file in processed_video_files:
    contents.append(get_video_content(video_file, ...))
```

### Custom Training Script

- **`src/train/train_grpo_multivideo.py`**: Multi-video GRPO trainer
- **Enhanced logging**: Detailed progress and reward tracking
- **Checkpoint support**: Resume training from interruptions
- **LoRA compatible**: Supports parameter-efficient fine-tuning

## 🐛 Debug Mode

Enable detailed reward logging:

```bash
export DEBUG_MODE=true
export LOG_PATH=grpo_rewards.log

# Run training - rewards will be logged to grpo_rewards.log
bash scripts/finetune_grpo_multivideo.sh
```

## 📈 Expected Results

With the numerical reward function, the model should learn to:

1. **Extract numbers accurately** from video content
2. **Handle percentage formats** (65.4% → 65.4)
3. **Improve precision** on edge cases (values far from 50)
4. **Maintain performance** on non-numerical questions

## 🚀 Next Steps

After training completes:

1. **Evaluate** the model on your test set using the inference scripts
2. **Monitor** Wandb dashboards for reward trends
3. **Analyze** debug logs to understand reward distributions
4. **Fine-tune** reward function parameters if needed

The GRPO-trained model should show improved numerical accuracy compared to standard SFT training!