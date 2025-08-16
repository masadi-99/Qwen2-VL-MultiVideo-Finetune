#!/usr/bin/env python3
"""
Working Multi-Video Inference Script using the same model loading as src.serve.app
This should work with your 3B LoRA checkpoint that works in the serving app.
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
from src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init
from qwen_vl_utils import process_vision_info


def parse_args():
    parser = argparse.ArgumentParser(description="Working multi-video inference using serving app's model loading")
    
    # Model arguments (same as serving app)
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--model_base", 
        type=str, 
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Base model for LoRA (same as in serving app)"
    )
    parser.add_argument(
        "--load_8bit", 
        action="store_true",
        help="Load model in 8bit"
    )
    parser.add_argument(
        "--load_4bit", 
        action="store_true",
        help="Load model in 4bit"
    )
    parser.add_argument(
        "--disable_flash_attention", 
        action="store_true",
        help="Disable flash attention"
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
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    
    # Device and performance
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    
    return parser.parse_args()


def load_model_like_serving_app(args):
    """Load model using the exact same method as the serving app"""
    print(f"Loading model using serving app method...")
    print(f"Model path: {args.model_path}")
    print(f"Model base: {args.model_base}")
    
    # Disable torch init for faster loading (same as serving app)
    disable_torch_init()
    
    # Get model name (same as serving app)
    model_name = get_model_name_from_path(args.model_path)
    print(f"Model name: {model_name}")
    
    # Use flash attention unless disabled
    use_flash_attn = not args.disable_flash_attention
    
    # Load model using the exact same function as serving app
    try:
        processor, model = load_pretrained_model(
            model_base=args.model_base,
            model_path=args.model_path, 
            device_map=args.device,
            model_name=model_name, 
            load_4bit=args.load_4bit,
            load_8bit=args.load_8bit,
            device=args.device,
            use_flash_attn=use_flash_attn
        )
        
        print("✅ Model loaded successfully using serving app method!")
        return processor, model, args.device
        
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        raise e


def prepare_video_content(video_files: List[str], video_folder: str, args):
    """Prepare video content - same format as serving app"""
    contents = []
    
    for video_file in video_files:
        # Handle video file path
        if not os.path.exists(video_file):
            if not video_file.startswith("http"):
                video_file = os.path.join(video_folder, video_file)
        
        # Create video content dict (same as serving app line 43, 60)
        content = {
            "type": "video",
            "video": video_file,
            "fps": args.fps  # Use the fps from args
        }
        
        # Note: The serving app doesn't use max_pixels, resized_width, resized_height
        # but we can still add them for compatibility
        if args.video_max_pixels:
            content["max_pixels"] = args.video_max_pixels
        if args.video_resized_width and args.video_resized_height:
            content["resized_width"] = args.video_resized_width
            content["resized_height"] = args.video_resized_height
        
        contents.append(content)
    
    return contents


def prepare_conversation(sample: Dict[str, Any], video_folder: str, args):
    """Prepare conversation in the exact format as serving app"""
    # Get the user's question and ground truth
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
    
    # Build conversation format (same as serving app structure)
    conversation = []
    
    # Prepare content list
    user_content = []
    
    # Add videos if present
    if "video" in sample:
        video_files = sample["video"]
        if isinstance(video_files, str):
            video_files = [video_files]
        
        # Add video contents (same format as serving app)
        video_contents = prepare_video_content(video_files, video_folder, args)
        user_content.extend(video_contents)
    
    # Add text content
    text_content = {"type": "text", "text": user_question}
    user_content.append(text_content)
    
    # Create conversation (same structure as serving app line 64)
    conversation.append({"role": "user", "content": user_content})
    
    return conversation, ground_truth


def generate_response(model, processor, conversation: List[Dict], args, device: str):
    """Generate response using the same method as serving app"""
    try:
        # Apply chat template (same as serving app line 66)
        prompt = processor.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process vision info (same as serving app line 67)
        image_inputs, video_inputs = process_vision_info(conversation)
        
        # Prepare inputs (same as serving app line 69)
        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        # Generation arguments (same structure as serving app)
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "do_sample": True if args.temperature > 0 else False,
            "repetition_penalty": args.repetition_penalty,
            "eos_token_id": processor.tokenizer.eos_token_id,
        }
        
        # Add top_p if using sampling
        if args.do_sample and args.top_p:
            generation_kwargs["top_p"] = args.top_p
        
        # Generate (same method as serving app)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                **generation_kwargs
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
    """Main inference function using serving app's approach"""
    print("=== Working Multi-Video Inference (Using Serving App Method) ===")
    print(f"Model: {args.model_path}")
    print(f"Base model: {args.model_base}")
    print(f"Test data: {args.test_data_path}")
    print()
    
    # Load model using serving app method
    processor, model, device = load_model_like_serving_app(args)
    
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
            
            # Prepare conversation
            conversation, ground_truth = prepare_conversation(sample, args.video_folder, args)
            
            # Generate response
            model_response = generate_response(model, processor, conversation, args, device)
            
            # Store result
            result = {
                "id": sample_id,
                "user_question": conversation[0]["content"][-1]["text"],  # Last item is text
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
            "model_base": args.model_base,
            "test_data_path": args.test_data_path,
            "total_samples": len(test_data),
            "successful_samples": len(results),
            "failed_samples": len(errors),
            "loading_method": "serving_app_compatible",
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
    
    # Show sample results
    if results and args.verbose:
        print(f"\n=== Sample Results ===")
        for i, result in enumerate(results[:2]):
            print(f"\nSample {i+1}: {result['id']}")
            print(f"Videos: {result['video_count']}")
            print(f"Question: {result['user_question'][:100]}...")
            print(f"Model Response: {result['model_response'][:100]}...")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)