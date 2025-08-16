#!/usr/bin/env python3
"""
Fast Multi-Video Inference Script - Optimized for quick loading
Handles both LoRA and full checkpoints with speed optimizations
"""

import os
import json
import torch
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# Import the working model loading function from the serving app
from src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init, is_lora_model
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser(description="Fast multi-video inference with optimized loading")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_base", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--force_lora", action="store_true", help="Force treat as LoRA model")
    parser.add_argument("--force_full", action="store_true", help="Force treat as full model") 
    parser.add_argument("--fast_loading", action="store_true", default=True, help="Use optimizations for faster loading")
    parser.add_argument("--load_8bit", action="store_true")
    parser.add_argument("--load_4bit", action="store_true")
    
    # Data arguments  
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--video_folder", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="inference_results.json")
    
    # Video processing
    parser.add_argument("--fps", type=float, default=0.25)
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--do_sample", action="store_true")
    
    # Performance
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    
    return parser.parse_args()


def check_model_type(model_path):
    """Enhanced model type detection"""
    model_path = Path(model_path)
    
    print(f"🔍 Analyzing model at: {model_path}")
    
    # Check for LoRA files
    lora_files = {
        'adapter_config.json': model_path / 'adapter_config.json',
        'adapter_model.safetensors': model_path / 'adapter_model.safetensors', 
        'adapter_model.bin': model_path / 'adapter_model.bin'
    }
    
    found_lora = []
    for name, path in lora_files.items():
        if path.exists():
            found_lora.append(name)
            print(f"  ✅ Found: {name}")
    
    # Check for full model files
    full_model_files = {
        'pytorch_model.bin': model_path / 'pytorch_model.bin',
        'model.safetensors': model_path / 'model.safetensors',
        'pytorch_model-00001-of-*.bin': list(model_path.glob('pytorch_model-*.bin'))
    }
    
    found_full = []
    for name, path in full_model_files.items():
        if name.endswith('*.bin'):
            if path:  # path is a list for glob
                found_full.append(f"sharded model files ({len(path)} shards)")
                print(f"  📦 Found: {len(path)} model shards")
        elif (isinstance(path, Path) and path.exists()) or path:
            found_full.append(name)
            print(f"  📦 Found: {name}")
    
    # Determine model type
    is_lora = len(found_lora) >= 2  # Need at least config and weights
    is_full = len(found_full) > 0
    
    print(f"📊 Detection results:")
    print(f"  LoRA model: {is_lora}")
    print(f"  Full model: {is_full}")
    
    return is_lora, is_full


def load_model_fast(args):
    """Fast model loading with optimizations"""
    print(f"🚀 Fast model loading...")
    
    # Detect model type
    is_lora, is_full = check_model_type(args.model_path)
    
    # Apply user overrides
    if args.force_lora:
        is_lora = True
        print("🔧 Forced LoRA mode")
    elif args.force_full:
        is_lora = False
        print("🔧 Forced full model mode")
    
    # Speed optimizations
    if args.fast_loading:
        print("⚡ Applying speed optimizations...")
        disable_torch_init()
        # Set environment variables for faster loading
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    
    try:
        if is_lora:
            print("📋 Loading as LoRA model (faster)...")
            # Use serving app method for LoRA
            model_name = get_model_name_from_path(args.model_path)
            processor, model = load_pretrained_model(
                model_base=args.model_base,
                model_path=args.model_path, 
                device_map=args.device,
                model_name=model_name, 
                load_4bit=args.load_4bit,
                load_8bit=args.load_8bit,
                device=args.device,
                use_flash_attn=True  # Enable for speed
            )
            
        else:
            print("📦 Loading as full model...")
            # Direct loading for full models with optimizations
            load_kwargs = {
                "torch_dtype": torch.float16,  # Use FP16 for speed
                "device_map": "auto",
                "low_cpu_mem_usage": True,    # Memory optimization
            }
            
            if args.load_4bit:
                from transformers import BitsAndBytesConfig
                load_kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            elif args.load_8bit:
                load_kwargs['load_in_8bit'] = True
            
            processor = AutoProcessor.from_pretrained(args.model_path, use_fast=True)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                args.model_path, 
                **load_kwargs
            )
        
        print("✅ Model loaded successfully!")
        return processor, model, args.device
        
    except Exception as e:
        print(f"❌ Fast loading failed: {str(e)}")
        print("🔄 Falling back to serving app method...")
        
        # Fallback to serving app method
        model_name = get_model_name_from_path(args.model_path)
        processor, model = load_pretrained_model(
            model_base=args.model_base,
            model_path=args.model_path, 
            device_map=args.device,
            model_name=model_name, 
            load_4bit=args.load_4bit,
            load_8bit=args.load_8bit,
            device=args.device,
            use_flash_attn=True
        )
        return processor, model, args.device


def prepare_conversation_fast(sample: Dict[str, Any], video_folder: str, args):
    """Fast conversation preparation"""
    # Get user question and ground truth
    user_question = None
    ground_truth = None
    
    for conv in sample["conversations"]:
        if conv["from"] in ["human", "user"]:
            user_question = conv["value"]
        elif conv["from"] in ["gpt", "assistant"]:
            ground_truth = conv["value"]
            break
    
    if not user_question:
        raise ValueError(f"No user question found in sample {sample.get('id', 'unknown')}")
    
    # Build conversation
    conversation = []
    user_content = []
    
    # Add videos
    if "video" in sample:
        video_files = sample["video"]
        if isinstance(video_files, str):
            video_files = [video_files]
        
        for video_file in video_files:
            if not os.path.exists(video_file):
                if not video_file.startswith("http"):
                    video_file = os.path.join(video_folder, video_file)
            
            # Simple video content (minimal processing for speed)
            user_content.append({
                "type": "video",
                "video": video_file,
                "fps": args.fps
            })
    
    # Add text
    user_content.append({"type": "text", "text": user_question})
    conversation.append({"role": "user", "content": user_content})
    
    return conversation, ground_truth


def generate_response_fast(model, processor, conversation, args, device):
    """Fast response generation"""
    try:
        # Apply chat template
        prompt = processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        
        # Process vision
        image_inputs, video_inputs = process_vision_info(conversation)
        
        # Prepare inputs
        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        # Fast generation settings
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature if args.do_sample else None,
            "pad_token_id": processor.tokenizer.eos_token_id,
            "eos_token_id": processor.tokenizer.eos_token_id,
        }
        
        # Remove None values
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(**inputs, **gen_kwargs)
        
        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text.strip()
        
    except Exception as e:
        return f"[ERROR: {str(e)}]"


def run_inference(args):
    """Main fast inference function"""
    print("=== Fast Multi-Video Inference ===")
    print(f"Model: {args.model_path}")
    print(f"Base: {args.model_base}")
    print(f"Fast loading: {args.fast_loading}")
    print()
    
    # Load model with optimizations
    start_time = datetime.now()
    processor, model, device = load_model_fast(args)
    load_time = (datetime.now() - start_time).total_seconds()
    print(f"⏱️  Model loaded in {load_time:.1f} seconds")
    
    # Load test data
    with open(args.test_data_path, 'r') as f:
        test_data = json.load(f)
    
    if args.max_samples:
        test_data = test_data[:args.max_samples]
    
    print(f"📊 Processing {len(test_data)} samples")
    
    # Run inference
    results = []
    errors = []
    
    inference_start = datetime.now()
    
    for i, sample in enumerate(tqdm(test_data, desc="Inference")):
        try:
            sample_id = sample.get("id", f"sample_{i}")
            
            # Prepare conversation
            conversation, ground_truth = prepare_conversation_fast(sample, args.video_folder, args)
            
            # Generate
            model_response = generate_response_fast(model, processor, conversation, args, device)
            
            # Store result
            result = {
                "id": sample_id,
                "user_question": conversation[0]["content"][-1]["text"],
                "ground_truth": ground_truth,
                "model_response": model_response,
                "video_count": len(sample.get("video", [])) if isinstance(sample.get("video"), list) else (1 if "video" in sample else 0),
                "timestamp": datetime.now().isoformat()
            }
            
            results.append(result)
            
        except Exception as e:
            errors.append({
                "id": sample.get("id", f"sample_{i}"),
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    inference_time = (datetime.now() - inference_start).total_seconds()
    
    # Save results
    output_data = {
        "metadata": {
            "model_path": args.model_path,
            "model_base": args.model_base,
            "load_time_seconds": load_time,
            "inference_time_seconds": inference_time,
            "samples_per_second": len(results) / inference_time if inference_time > 0 else 0,
            "total_samples": len(test_data),
            "successful_samples": len(results),
            "failed_samples": len(errors),
            "timestamp": datetime.now().isoformat()
        },
        "results": results,
        "errors": errors
    }
    
    with open(args.output_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Summary
    print(f"\n=== Performance Summary ===")
    print(f"⏱️  Model load time: {load_time:.1f}s")
    print(f"🔄 Inference time: {inference_time:.1f}s")
    print(f"📈 Speed: {len(results)/inference_time:.2f} samples/sec")
    print(f"✅ Success rate: {len(results)}/{len(test_data)} ({100*len(results)/len(test_data):.1f}%)")
    print(f"💾 Results: {args.output_path}")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)