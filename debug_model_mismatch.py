#!/usr/bin/env python3
"""
Debug script to identify model architecture mismatches
Helps diagnose checkpoint loading errors in Qwen2.5-VL models
"""

import torch
import argparse
from transformers import Qwen2VLForConditionalGeneration, AutoConfig
from pathlib import Path
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Debug model checkpoint compatibility")
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        required=True,
        help="Path to the checkpoint/model directory"
    )
    parser.add_argument(
        "--base_model", 
        type=str, 
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Base model to compare against"
    )
    return parser.parse_args()


def get_model_info(model_path_or_name: str):
    """Get model configuration and architecture info"""
    try:
        print(f"\n=== Analyzing: {model_path_or_name} ===")
        
        # Load config
        config = AutoConfig.from_pretrained(model_path_or_name, trust_remote_code=True)
        
        print(f"Model type: {config.model_type}")
        print(f"Architecture: {config.architectures}")
        
        # Key dimension parameters
        if hasattr(config, 'hidden_size'):
            print(f"Hidden size: {config.hidden_size}")
        if hasattr(config, 'intermediate_size'):
            print(f"Intermediate size: {config.intermediate_size}")
        if hasattr(config, 'num_attention_heads'):
            print(f"Attention heads: {config.num_attention_heads}")
        if hasattr(config, 'num_hidden_layers'):
            print(f"Hidden layers: {config.num_hidden_layers}")
        if hasattr(config, 'vocab_size'):
            print(f"Vocab size: {config.vocab_size}")
        
        # Vision-specific parameters
        if hasattr(config, 'vision_config'):
            vision_config = config.vision_config
            print(f"Vision hidden size: {getattr(vision_config, 'hidden_size', 'N/A')}")
            print(f"Vision patch size: {getattr(vision_config, 'patch_size', 'N/A')}")
            print(f"Vision image size: {getattr(vision_config, 'image_size', 'N/A')}")
        
        return config
        
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        return None


def check_checkpoint_files(checkpoint_path: str):
    """Check what files are in the checkpoint directory"""
    checkpoint_path = Path(checkpoint_path)
    
    print(f"\n=== Checkpoint Directory Contents ===")
    if checkpoint_path.is_dir():
        files = list(checkpoint_path.iterdir())
        for file in sorted(files):
            print(f"  {file.name}")
            
        # Check if it's a PyTorch checkpoint
        pytorch_files = list(checkpoint_path.glob("*.pt")) + list(checkpoint_path.glob("*.pth"))
        safetensor_files = list(checkpoint_path.glob("*.safetensors"))
        
        if pytorch_files:
            print(f"\nPyTorch checkpoint files found: {[f.name for f in pytorch_files]}")
        if safetensor_files:
            print(f"SafeTensor files found: {[f.name for f in safetensor_files]}")
            
        # Check for config.json
        config_file = checkpoint_path / "config.json"
        if config_file.exists():
            print("✓ config.json found")
            try:
                with open(config_file) as f:
                    config_data = json.load(f)
                    print(f"Model type from config: {config_data.get('model_type', 'Unknown')}")
                    print(f"Architectures: {config_data.get('architectures', 'Unknown')}")
            except Exception as e:
                print(f"Error reading config.json: {e}")
        else:
            print("✗ config.json not found")
            
    else:
        print(f"Path {checkpoint_path} is not a directory")


def suggest_fixes(checkpoint_config, base_config):
    """Suggest potential fixes for the mismatch"""
    print(f"\n=== Suggested Fixes ===")
    
    if not checkpoint_config or not base_config:
        print("Cannot compare configs - one or both failed to load")
        return
    
    checkpoint_hidden = getattr(checkpoint_config, 'hidden_size', None)
    base_hidden = getattr(base_config, 'hidden_size', None)
    
    if checkpoint_hidden and base_hidden and checkpoint_hidden != base_hidden:
        print(f"❌ Hidden size mismatch: checkpoint={checkpoint_hidden}, base={base_hidden}")
        
        # Determine which model sizes these correspond to
        size_mapping = {
            1280: "Qwen2.5-VL-3B-Instruct",
            2048: "Qwen2.5-VL-7B-Instruct", 
            3584: "Qwen2.5-VL-72B-Instruct"
        }
        
        checkpoint_model = size_mapping.get(checkpoint_hidden, f"Unknown ({checkpoint_hidden})")
        base_model = size_mapping.get(base_hidden, f"Unknown ({base_hidden})")
        
        print(f"Checkpoint appears to be from: {checkpoint_model}")
        print(f"You're trying to load into: {base_model}")
        
        print(f"\n🔧 Solutions:")
        print(f"1. Use the correct base model for inference:")
        print(f"   --base_model '{size_mapping.get(checkpoint_hidden, 'Qwen/Qwen2.5-VL-7B-Instruct')}'")
        print(f"   --model_path '{checkpoint_path}'")
        
        print(f"\n2. Or update your inference script:")
        print(f"   MODEL_PATH='{checkpoint_path}'")
        print(f"   BASE_MODEL='{size_mapping.get(checkpoint_hidden, 'Qwen/Qwen2.5-VL-7B-Instruct')}'")
        
        print(f"\n3. In inference.py, the error suggests you have:")
        print(f"   - Checkpoint hidden_size: {checkpoint_hidden} (from {checkpoint_model})")
        print(f"   - Current model hidden_size: {base_hidden} (from {base_model})")
        print(f"   Fix: Change --base_model to match the checkpoint size")
        
    else:
        print("✓ Hidden sizes match or couldn't be determined")
        print("The error might be due to other architectural differences")
        print("Check the full error traceback for more specific layer mismatches")


def main():
    args = parse_args()
    
    print("=== Model Checkpoint Compatibility Checker ===")
    
    # Check checkpoint directory
    check_checkpoint_files(args.checkpoint_path)
    
    # Analyze checkpoint model
    print(f"\n{'='*50}")
    checkpoint_config = get_model_info(args.checkpoint_path)
    
    # Analyze base model
    print(f"\n{'='*50}")
    base_config = get_model_info(args.base_model)
    
    # Suggest fixes
    suggest_fixes(checkpoint_config, base_config)
    
    print(f"\n{'='*50}")
    print("Summary:")
    print("The error 'size mismatch for bias: copying a param with shape torch.Size([2048]) from checkpoint, the shape in current model is torch.Size([1280])' indicates:")
    print("- Your checkpoint is from a 7B model (hidden_size=2048)")
    print("- You're trying to load it into a 3B model (hidden_size=1280)")
    print("- Solution: Use the correct base model size that matches your checkpoint")


if __name__ == "__main__":
    main()