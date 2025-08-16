#!/usr/bin/env python3
"""
Optimized Multi-Video Inference for Full SFT Models
Specifically designed for models trained with finetune_multivideo.sh (full SFT, not LoRA)
"""

import os
import json
import torch
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from transformers import (
    Qwen2VLForConditionalGeneration, 
    AutoProcessor,
    BitsAndBytesConfig
)
from qwen_vl_utils import process_vision_info


def parse_args():
    parser = argparse.ArgumentParser(description="Optimized inference for full SFT models")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to SFT fine-tuned model")
    parser.add_argument("--load_4bit", action="store_true", help="Use 4-bit quantization for speed")
    parser.add_argument("--load_8bit", action="store_true", help="Use 8-bit quantization for speed")
    
    # Data arguments  
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--video_folder", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="inference_sft_results.json")
    
    # Video processing (match training settings)
    parser.add_argument("--fps", type=int, default=1, help="FPS as integer (match training)")
    parser.add_argument("--video_max_pixels", type=int, default=4096, help="64*64 from training")
    parser.add_argument("--video_resized_width", type=int, default=64)
    parser.add_argument("--video_resized_height", type=int, default=64)
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true")
    
    # Performance
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    
    return parser.parse_args()


def load_sft_model_optimized(model_path: str, args):
    """Optimized loading for full SFT models"""
    print(f"🚀 Loading full SFT model from: {model_path}")
    
    # Set optimizations
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    
    # Prepare loading arguments
    load_kwargs = {
        "torch_dtype": torch.bfloat16,  # Match training bf16=True
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    
    # Add quantization for speed if requested
    if args.load_4bit:
        print("⚡ Using 4-bit quantization for faster loading")
        load_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    elif args.load_8bit:
        print("⚡ Using 8-bit quantization for faster loading")
        load_kwargs['load_in_8bit'] = True
    
    start_time = datetime.now()
    
    try:
        # Load processor with fast tokenizer
        print("📋 Loading processor...")
        processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        
        # Load model
        print("🧠 Loading model...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            **load_kwargs
        )
        
        load_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ Model loaded successfully in {load_time:.1f} seconds")
        
        return processor, model, args.device, load_time
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise e


def prepare_video_content_sft(video_files: List[str], video_folder: str, args):
    """Prepare video content matching SFT training format"""
    contents = []
    
    for video_file in video_files:
        # Handle video file path
        if not os.path.exists(video_file):
            if not video_file.startswith("http"):
                video_file = os.path.join(video_folder, video_file)
        
        # Create video content dict (match training settings exactly)
        content = {
            "type": "video",
            "video": video_file,
            "max_pixels": args.video_max_pixels,
            "fps": args.fps,  # Integer fps as in training
        }
        
        # Add resizing (match training)
        if args.video_resized_width and args.video_resized_height:
            content["resized_width"] = args.video_resized_width
            content["resized_height"] = args.video_resized_height
        
        contents.append(content)
    
    return contents


def prepare_conversation_sft(sample: Dict[str, Any], video_folder: str, args):
    """Prepare conversation for SFT model"""
    # Extract user question and ground truth
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
    
    # Add videos with SFT training format
    if "video" in sample:
        video_files = sample["video"]
        if isinstance(video_files, str):
            video_files = [video_files]
        
        video_contents = prepare_video_content_sft(video_files, video_folder, args)
        user_content.extend(video_contents)
    
    # Add text content
    text_content = {"type": "text", "text": user_question}
    user_content.append(text_content)
    
    # Create conversation
    conversation.append({"role": "user", "content": user_content})
    
    return conversation, ground_truth


def generate_response_sft(model, processor, conversation, args, device):
    """Generate response with SFT model"""
    try:
        # Apply chat template
        prompt = processor.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process vision info (our enhanced multi-video version)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            conversation, 
            return_video_kwargs=True
        )
        
        # Prepare inputs
        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs
        )
        
        # Move to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # Generation settings
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": processor.tokenizer.eos_token_id,
            "eos_token_id": processor.tokenizer.eos_token_id,
        }
        
        if args.do_sample:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
            })
        else:
            gen_kwargs["do_sample"] = False
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(**inputs, **gen_kwargs)
        
        # Decode (excluding input tokens)
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
        print(f"Error during generation: {str(e)}")
        return f"[ERROR: {str(e)}]"


def run_inference(args):
    """Main inference function for SFT models"""
    print("=== Optimized Multi-Video Inference for SFT Models ===")
    print(f"Model: {args.model_path}")
    print(f"FPS: {args.fps} (integer)")
    print(f"Video settings: {args.video_resized_width}x{args.video_resized_height}, {args.video_max_pixels} max pixels")
    print(f"Quantization: {'4-bit' if args.load_4bit else '8-bit' if args.load_8bit else 'None'}")
    print()
    
    # Load model
    processor, model, device, load_time = load_sft_model_optimized(args.model_path, args)
    
    # Load test data
    print("📊 Loading test data...")
    with open(args.test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    if args.max_samples:
        test_data = test_data[:args.max_samples]
        print(f"🔬 Limited to {args.max_samples} samples for testing")
    
    print(f"📋 Processing {len(test_data)} samples")
    print()
    
    # Run inference
    results = []
    errors = []
    
    inference_start = datetime.now()
    
    for i, sample in enumerate(tqdm(test_data, desc="Generating responses")):
        try:
            sample_id = sample.get("id", f"sample_{i}")
            
            if args.verbose:
                print(f"Processing {sample_id}...")
            
            # Prepare conversation
            conversation, ground_truth = prepare_conversation_sft(sample, args.video_folder, args)
            
            # Generate response
            model_response = generate_response_sft(model, processor, conversation, args, device)
            
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
            error_info = {
                "id": sample.get("id", f"sample_{i}"),
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            errors.append(error_info)
            
            if args.verbose:
                print(f"✗ Error: {str(e)}")
    
    inference_time = (datetime.now() - inference_start).total_seconds()
    
    # Save results
    print(f"\n💾 Saving results to {args.output_path}")
    
    output_data = {
        "metadata": {
            "model_path": args.model_path,
            "model_type": "full_sft",
            "quantization": "4bit" if args.load_4bit else "8bit" if args.load_8bit else "none",
            "load_time_seconds": load_time,
            "inference_time_seconds": inference_time,
            "samples_per_second": len(results) / inference_time if inference_time > 0 else 0,
            "video_settings": {
                "fps": args.fps,
                "max_pixels": args.video_max_pixels,
                "width": args.video_resized_width,
                "height": args.video_resized_height
            },
            "total_samples": len(test_data),
            "successful_samples": len(results),
            "failed_samples": len(errors),
            "timestamp": datetime.now().isoformat()
        },
        "results": results,
        "errors": errors
    }
    
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Performance summary
    print(f"\n=== Performance Summary ===")
    print(f"⏱️  Model load time: {load_time:.1f}s")
    print(f"🔄 Inference time: {inference_time:.1f}s") 
    print(f"📈 Speed: {len(results)/inference_time:.2f} samples/sec")
    print(f"✅ Success rate: {len(results)}/{len(test_data)} ({100*len(results)/len(test_data):.1f}%)")
    print(f"💾 Results saved to: {args.output_path}")
    
    if results and args.verbose:
        print(f"\n=== Sample Results ===")
        for i, result in enumerate(results[:2]):
            print(f"\nSample {i+1}: {result['id']}")
            print(f"Videos: {result['video_count']}")
            print(f"Question: {result['user_question'][:100]}...")
            print(f"Response: {result['model_response'][:100]}...")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)