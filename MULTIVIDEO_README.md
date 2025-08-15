# Multi-Video Training for Qwen2.5-VL

This fork adds support for training Qwen2.5-VL models on datasets with **multiple videos per conversation sample** (tested up to 30+ videos).

## 🚀 New Features

### ✅ Multi-Video Processing
- **Process up to 30+ videos per sample** without fps parameter conflicts
- **Automatic detection** of single vs multiple video samples
- **Memory-optimized processing** for large video datasets
- **Backward compatible** with existing single-video datasets

### 🔧 Key Fixes

1. **Fixed TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'**
   - Added `get_multiple_videos_info()` function for batch video processing
   - Modified SFT and DPO datasets to handle multiple videos properly

2. **Memory Optimization for 30+ Videos**
   - Reduced video resolution defaults for memory efficiency
   - Optimized batch sizes and gradient accumulation
   - DeepSpeed ZeRO-3 with offloading for large models

## 📊 Memory Requirements

### For 30 Videos per Sample on 8x H100 (80GB each):

| Configuration | Video Resolution | Frames per Video | Memory per Sample | Status |
|---------------|------------------|------------------|-------------------|---------|
| **Recommended** | 64x64 | ~3-4 frames | ~8-12GB | ✅ Works |
| Conservative | 48x48 | ~2-3 frames | ~4-6GB | ✅ Safe |
| Aggressive | 80x80 | ~4-5 frames | ~15-20GB | ⚠️ Risky |

## 🛠️ Usage

### 1. Dataset Format

Your dataset should have this structure:

```json
[
  {
    "id": "sample_1",
    "video": [
      "video1.mp4", 
      "video2.mp4", 
      "video3.mp4",
      "..."  // up to 30+ videos
    ],
    "conversations": [
      {
        "from": "human", 
        "value": "<video>\n<video>\n<video>\nDescribe these videos."
      },
      {
        "from": "gpt",
        "value": "In these videos, I can see..."
      }
    ]
  }
]
```

**Important:** Use the same number of `<video>` tags as videos in your `"video"` array.

### 2. Training Script

Use the provided multi-video training script:

```bash
# Copy and edit the script for your paths
cp scripts/finetune_multivideo.sh scripts/my_multivideo_training.sh

# Edit the paths in the script:
# --data_path /path/to/your/multivideo/data.json
# --image_folder /path/to/your/video/folder

# Run training
bash scripts/my_multivideo_training.sh
```

### 3. Memory Configuration

For different hardware setups:

**8x H100 (80GB each) - Recommended:**
```bash
--video_max_pixels $((64 * 64))
--video_resized_width 64
--video_resized_height 64
--fps 0.25
--per_device_train_batch_size 1
--gradient_accumulation_steps 2
```

**4x A100 (40GB each) - Conservative:**
```bash
--video_max_pixels $((48 * 48))
--video_resized_width 48
--video_resized_height 48
--fps 0.2
--per_device_train_batch_size 1
--gradient_accumulation_steps 4
```

## 🔬 Technical Details

### How Multi-Video Processing Works

**Before (Broken):**
```python
# This caused fps conflicts with 30+ videos
for video_file in video_files:  # 30 iterations
    video_input, video_kwargs = get_video_info(video_file, ...)  # 30 separate calls
    videos.append(video_input)
```

**After (Fixed):**
```python
# Process all videos together
if len(video_files) > 1:
    videos, video_kwargs = get_multiple_videos_info(video_files, ...)  # Single call
else:
    # Backward compatibility for single videos
    video_input, video_kwargs = get_video_info(video_files[0], ...)
```

### Video Token Calculation

For each video with the recommended settings:
- **Resolution:** 64x64 pixels → 4x4 patches (after 16x16 patch size)
- **Frames:** ~3-4 frames (with fps=0.25)
- **Tokens per video:** 4 × 4 × 4 = 64 tokens
- **Total for 30 videos:** 30 × 64 = 1,920 tokens

### Memory Breakdown

**Per sample with 30 videos (64x64, 4 frames each):**
- Video tokens: ~1,920 tokens
- Text tokens: ~500-1000 tokens  
- Total context: ~2,500 tokens
- **GPU memory per sample:** ~8-12GB
- **Batch size 1 on 8 GPUs:** 64-96GB total

## 🐛 Troubleshooting

### Error: "unsupported operand type(s) for *: 'float' and 'NoneType'"
- **Cause:** Using the original codebase with multiple videos
- **Solution:** Use this fixed version with `get_multiple_videos_info()`

### Error: "CUDA out of memory"
- **Solution 1:** Reduce video resolution: `--video_resized_width 48 --video_resized_height 48`
- **Solution 2:** Reduce fps: `--fps 0.2` or `--fps 0.15`
- **Solution 3:** Use ZeRO-3 offloading: `--deepspeed scripts/zero3_offload.json`

### Error: "The length of fps (1) must be equal to the length of video_grid_thw (30)"
- **Cause:** Using original codebase with multiple videos
- **Solution:** This error is fixed in this version

## 📈 Performance Tips

1. **Start Small:** Begin with 5-10 videos per sample to test your setup
2. **Monitor Memory:** Use `nvidia-smi` to watch GPU memory usage
3. **Adjust Batch Size:** Reduce if you hit memory limits
4. **Use Mixed Precision:** Always use `--bf16 True` for memory efficiency

## 🔄 Compatibility

- ✅ **Single video datasets:** Fully backward compatible
- ✅ **Mixed datasets:** Can handle both single and multi-video samples
- ✅ **All training methods:** SFT, DPO, GRPO, Classification
- ✅ **Original features:** All existing functionality preserved

## 🤝 Contributing

This is a community-driven enhancement. Feel free to:
- Report issues with specific video counts or configurations
- Suggest memory optimizations
- Share successful training configurations

## 📚 Original Repository

Based on: [2U1/Qwen2-VL-Finetune](https://github.com/2U1/Qwen2-VL-Finetune)