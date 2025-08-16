#!/usr/bin/env python3
"""
Fixed Multi-Video Inference Script for Qwen2.5-VL
Handles checkpoint loading issues and model mismatches
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
    AutoConfig
)
from qwen_vl_utils import process_vision_info


def parse_args():
    parser = argparse.ArgumentParser(description="Fixed multi-video inference for Qwen2.5-VL")
    
    # Model arguments
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the fine-tuned model (or base model)"
    )
    parser.add_argument(
        "--base_model", 
        type=str, 
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Base model name for processor and config"
    )
    parser.add_argument(
        "--force_config", 
        action="store_true",
        help="Force use base model config even if checkpoint has different config"
    )
    parser.add_argument(
        "--strict_loading", 
        action="store_true",
        help="Use strict state dict loading (default: False for compatibility)"
    )
    
    # Data arguments  
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--video_folder", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="inference_results.json")
    
    # Video processing arguments
    parser.add_argument("--video_max_pixels", type=int, default=4096)
    parser.add_argument("--video_resized_width", type=int, default=64)
    parser.add_argument("--video_resized_height", type=int, default=64)
    parser.add_argument("--fps", type=float, default=0.25)
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true")
    
    # Device and performance
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    
    return parser.parse_args()


def safe_load_model_and_processor(model_path: str, base_model: str, device: str, args):
    """Safely load model with fallback strategies for mismatches"""
    print(f"Loading model from: {model_path}")
    print(f"Using base model for processor: {base_model}")
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Always load processor from base model
    processor = AutoProcessor.from_pretrained(base_model)
    
    # Strategy 1: Try loading normally first
    try:
        print("Attempting normal model loading...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        print("✅ Normal loading successful")
        
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print(f"❌ Size mismatch detected: {str(e)}")
            print("🔄 Trying fallback strategies...")
            
            # Strategy 2: Force base model config
            if args.force_config:
                print("Strategy 2: Loading with forced base model config...")
                try:
                    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
                    model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_path,
                        config=config,
                        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                        device_map="auto" if device == "cuda" else None,
                        trust_remote_code=True
                    )
                    print("✅ Forced config loading successful")
                except Exception as e2:
                    print(f"❌ Forced config failed: {str(e2)}")
                    raise e
            
            # Strategy 3: Non-strict loading
            if not args.strict_loading:
                print("Strategy 3: Trying non-strict state dict loading...")
                try:
                    # Load base model first
                    model = Qwen2VLForConditionalGeneration.from_pretrained(
                        base_model,
                        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                        device_map="auto" if device == "cuda" else None,
                        trust_remote_code=True
                    )
                    
                    # Try to load checkpoint state dict non-strictly
                    checkpoint_path = Path(model_path)
                    if (checkpoint_path / "pytorch_model.bin").exists():
                        state_dict = torch.load(checkpoint_path / "pytorch_model.bin", map_location="cpu")
                        missing, unexpected = model.load_state_dict(state_dict, strict=False)
                        print(f"✅ Non-strict loading completed")
                        if missing:
                            print(f"⚠️  Missing keys: {len(missing)} (these will use base model weights)")
                        if unexpected:
                            print(f"⚠️  Unexpected keys: {len(unexpected)} (these will be ignored)")
                    else:
                        print("❌ No pytorch_model.bin found for non-strict loading")
                        raise e
                        
                except Exception as e3:
                    print(f"❌ Non-strict loading failed: {str(e3)}")
                    raise e
            else:
                raise e
        else:
            raise e
    
    except Exception as e:
        print(f"❌ Unexpected error loading model: {str(e)}")
        raise e
    
    # Move to device if needed
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    
    print(f"Model loaded on device: {model.device if hasattr(model, 'device') else device}")
    return model, processor, device


def prepare_video_content(video_files: List[str], video_folder: str, args):
    """Prepare video content for processing"""
    contents = []
    
    for video_file in video_files:
        # Handle video file path
        if not os.path.exists(video_file):
            if not video_file.startswith("http"):
                video_file = os.path.join(video_folder, video_file)
        
        # Create video content dict
        content = {
            "type": "video",
            "video": video_file,
            "max_pixels": args.video_max_pixels,
        }
        
        # Add fps parameter
        content["fps"] = args.fps
        
        # Add resizing if specified
        if args.video_resized_width and args.video_resized_height:
            content["resized_width"] = args.video_resized_width
            content["resized_height"] = args.video_resized_height
        
        contents.append(content)
    
    return contents


def prepare_messages(sample: Dict[str, Any], video_folder: str, args):
    """Prepare messages for the model"""
    # Get the user's question (first conversation item)
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
    
    # Prepare video contents
    contents = []
    
    if "video" in sample:
        video_files = sample["video"]
        if isinstance(video_files, str):
            video_files = [video_files]
        
        # Add video contents
        video_contents = prepare_video_content(video_files, video_folder, args)
        contents.extend(video_contents)
    
    # Add text content
    text_content = {"type": "text", "text": user_question}
    contents.append(text_content)
    
    # Create messages
    messages = [{"role": "user", "content": contents}]
    
    return messages, ground_truth


def generate_response(model, processor, messages: List[Dict], args, device: str):
    """Generate response from the model"""
    try:
        # Apply chat template
        text = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process vision info
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        
        # Prepare inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs
        )
        
        # Move to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        
        # Decode generated tokens (excluding input tokens)
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
    """Main inference function with error handling"""
    print("=== Fixed Multi-Video Inference for Qwen2.5-VL ===")
    print(f"Model: {args.model_path}")
    print(f"Base model: {args.base_model}")
    print(f"Test data: {args.test_data_path}")
    print(f"Force config: {args.force_config}")
    print(f"Strict loading: {args.strict_loading}")
    print()
    
    # Load model and processor with fallback strategies
    try:
        model, processor, device = safe_load_model_and_processor(
            args.model_path, args.base_model, args.device, args
        )
    except Exception as e:
        print(f"\n❌ Failed to load model: {str(e)}")
        print("\n🔧 Try these solutions:")
        print("1. Run: python debug_checkpoint.py --checkpoint_path", args.model_path)
        print("2. Use --force_config flag")
        print("3. Check if this is a LoRA checkpoint that needs merging")
        return
    
    # Load test data
    print("Loading test data...")
    with open(args.test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    if args.max_samples:
        test_data = test_data[:args.max_samples]
        print(f"Limited to {args.max_samples} samples for testing")
    
    print(f"Loaded {len(test_data)} test samples")
    print()
    
    # Run inference
    results = []
    errors = []
    
    for i, sample in enumerate(tqdm(test_data, desc="Running inference")):
        try:
            sample_id = sample.get("id", f"sample_{i}")
            
            if args.verbose:
                print(f"\nProcessing sample {i+1}/{len(test_data)}: {sample_id}")
            
            # Prepare messages
            messages, ground_truth = prepare_messages(sample, args.video_folder, args)
            
            # Generate response
            model_response = generate_response(model, processor, messages, args, device)
            
            # Store result
            result = {
                "id": sample_id,
                "user_question": messages[0]["content"][-1]["text"],
                "ground_truth": ground_truth,
                "model_response": model_response,
                "video_count": len(sample.get("video", [])) if isinstance(sample.get("video"), list) else (1 if "video" in sample else 0),
                "timestamp": datetime.now().isoformat()
            }
            
            results.append(result)
            
            if args.verbose:
                print(f"✓ Generated response for {sample_id}")
                
        except Exception as e:
            error_info = {
                "id": sample.get("id", f"sample_{i}"),
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            errors.append(error_info)
            
            if args.verbose:
                print(f"✗ Error processing {sample.get('id', f'sample_{i}')}: {str(e)}")
    
    # Save results
    print(f"\nSaving results to {args.output_path}")
    
    output_data = {
        "metadata": {
            "model_path": args.model_path,
            "base_model": args.base_model,
            "force_config": args.force_config,
            "strict_loading": args.strict_loading,
            "test_data_path": args.test_data_path,
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
    
    # Print summary
    print("\n=== Inference Complete ===")
    print(f"✓ Successful: {len(results)}/{len(test_data)} samples")
    if errors:
        print(f"✗ Failed: {len(errors)} samples")
    print(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)