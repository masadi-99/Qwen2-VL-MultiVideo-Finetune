# Multi-Video Inference for Qwen2.5-VL

This directory contains comprehensive inference tools for evaluating fine-tuned Qwen2.5-VL models on test datasets with multiple videos per sample.

## 🚀 Features

- ✅ **Multi-video inference** - Handle 30+ videos per sample
- ✅ **Flexible model loading** - Use fine-tuned or base models  
- ✅ **Structured output** - JSON format with metadata and analysis
- ✅ **Error handling** - Graceful failure with detailed error reporting
- ✅ **Results analysis** - Built-in statistics and similarity metrics
- ✅ **Memory efficient** - Optimized for large video datasets

## 📁 Files

| File | Description |
|------|-------------|
| `inference.py` | Main inference script |
| `run_inference.sh` | Example usage script |
| `analyze_results.py` | Results analysis and statistics |
| `examples/test_dataset_example.json` | Example test dataset format |

## 🛠️ Usage

### 1. Basic Inference

```bash
# Edit the paths in the script
cp run_inference.sh my_inference.sh
nano my_inference.sh  # Edit MODEL_PATH, TEST_DATA_PATH, VIDEO_FOLDER

# Run inference
bash my_inference.sh
```

### 2. Manual Inference

```bash
python inference.py \
    --model_path "output/multivideo_training" \
    --test_data_path "path/to/test_data.json" \
    --video_folder "path/to/videos/" \
    --output_path "results.json" \
    --video_max_pixels 4096 \
    --video_resized_width 64 \
    --video_resized_height 64 \
    --fps 0.25 \
    --max_new_tokens 512 \
    --temperature 0.7 \
    --verbose
```

### 3. Analyze Results

```bash
python analyze_results.py \
    --results_path "results.json" \
    --show_examples 10 \
    --show_errors \
    --output_summary "analysis_summary.json"
```

## 📊 Test Dataset Format

Your test dataset should follow the same format as training data:

```json
[
  {
    "id": "test_sample_1",
    "video": [
      "video1.mp4", 
      "video2.mp4",
      "..."
    ],
    "conversations": [
      {
        "from": "human",
        "value": "<video>\n<video>\nYour question here"
      },
      {
        "from": "assistant", 
        "value": "Ground truth response"
      }
    ]
  }
]
```

**Key Points:**
- Use same number of `<video>` tags as videos in the array
- "from": "human" for questions, "from": "assistant" for ground truth
- Single video samples are also supported (backward compatible)

## 🎯 Output Format

The inference script generates a comprehensive JSON file:

```json
{
  "metadata": {
    "model_path": "path/to/model",
    "test_data_path": "path/to/test.json", 
    "total_samples": 100,
    "successful_samples": 98,
    "failed_samples": 2,
    "generation_params": {...},
    "video_params": {...},
    "timestamp": "2024-12-XX..."
  },
  "results": [
    {
      "id": "sample_1",
      "user_question": "Question text",
      "ground_truth": "Expected response", 
      "model_response": "Generated response",
      "video_count": 5,
      "timestamp": "..."
    }
  ],
  "errors": [
    {
      "id": "failed_sample",
      "error": "Error description",
      "timestamp": "..."
    }
  ]
}
```

## 📈 Analysis Features

The `analyze_results.py` script provides:

### Basic Statistics
- Total samples processed
- Average response lengths
- Video count distribution
- Success/failure rates

### Content Analysis
- Question type classification (descriptive, analytical, counting, temporal)
- Response similarity metrics
- Error pattern analysis

### Sample Comparisons
- Side-by-side ground truth vs model responses
- Similarity scores
- Length comparisons

## ⚙️ Configuration Options

### Video Processing
```bash
--video_max_pixels 4096        # Match training settings
--video_resized_width 64       # Match training settings  
--video_resized_height 64      # Match training settings
--fps 0.25                     # Match training settings
```

### Generation Parameters
```bash
--max_new_tokens 512           # Maximum response length
--temperature 0.7              # Sampling temperature
--top_p 0.9                    # Nucleus sampling
--do_sample                    # Enable sampling
```

### Performance Options
```bash
--device auto                  # auto, cuda, cpu
--batch_size 1                 # Keep at 1 for multi-video
--max_samples 100              # Limit for testing
--verbose                      # Detailed progress
```

## 🔧 Memory Considerations

For large datasets with 30+ videos per sample:

**Recommended Settings:**
- `--batch_size 1` (required for multi-video)
- Use `--video_max_pixels 4096` (64x64) 
- Monitor GPU memory with `nvidia-smi`

**If CUDA out of memory:**
- Reduce `--video_resized_width` and `--video_resized_height`
- Lower `--fps` (fewer frames per video)
- Use `--device cpu` as fallback

## 📝 Example Workflows

### 1. Quick Test (5 samples)
```bash
python inference.py \
    --model_path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --test_data_path "examples/test_dataset_example.json" \
    --video_folder "test_videos/" \
    --output_path "quick_test.json" \
    --max_samples 5 \
    --verbose
```

### 2. Full Evaluation
```bash
# Run full inference
python inference.py \
    --model_path "output/my_finetuned_model" \
    --test_data_path "full_test_dataset.json" \
    --video_folder "all_test_videos/" \
    --output_path "full_evaluation.json" \
    --video_max_pixels 4096 \
    --fps 0.25

# Analyze results
python analyze_results.py \
    --results_path "full_evaluation.json" \
    --show_examples 20 \
    --output_summary "evaluation_summary.json"
```

### 3. Compare Models
```bash
# Base model
python inference.py --model_path "Qwen/Qwen2.5-VL-3B-Instruct" --output_path "base_results.json" ...

# Fine-tuned model  
python inference.py --model_path "output/finetuned" --output_path "finetuned_results.json" ...

# Analyze both
python analyze_results.py --results_path "base_results.json"
python analyze_results.py --results_path "finetuned_results.json"
```

## 🐛 Troubleshooting

### Common Issues

**Error: "CUDA out of memory"**
- Reduce video resolution: `--video_resized_width 48 --video_resized_height 48`
- Lower fps: `--fps 0.2`
- Use CPU: `--device cpu`

**Error: "Video file not found"**
- Check `--video_folder` path
- Ensure video files exist
- Check file permissions

**Error: "Model loading failed"**
- Verify `--model_path` is correct
- Check if model is compatible with Qwen2.5-VL
- Ensure sufficient disk space

**Poor results quality**
- Check if video processing settings match training
- Verify test dataset format
- Try different generation parameters

### Performance Tips

1. **Start small** - Test with `--max_samples 5` first
2. **Monitor memory** - Use `nvidia-smi` to watch GPU usage
3. **Batch processing** - For large datasets, split into smaller chunks
4. **Save frequently** - Use unique output filenames to avoid overwrites

## 🤝 Integration

This inference system integrates seamlessly with:
- Training scripts in `scripts/`
- Multi-video dataset processing 
- Standard evaluation pipelines
- Custom analysis workflows

The output format is designed to be compatible with common evaluation metrics and can be easily extended for specific analysis needs.