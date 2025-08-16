#!/usr/bin/env python3
"""
Debug script specifically for 3B model checkpoint issues
Investigates why a 3B checkpoint might show 7B characteristics
"""

import torch
import argparse
from transformers import Qwen2VLForConditionalGeneration, AutoConfig
from pathlib import Path
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Debug 3B checkpoint loading issues")
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        required=True,
        help="Path to your 3B checkpoint directory"
    )
    return parser.parse_args()


def analyze_checkpoint_structure(checkpoint_path):
    """Analyze the checkpoint directory structure and files"""
    checkpoint_path = Path(checkpoint_path)
    
    print(f"=== Analyzing Checkpoint: {checkpoint_path} ===")
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint path does not exist: {checkpoint_path}")
        return False
        
    if not checkpoint_path.is_dir():
        print(f"❌ Checkpoint path is not a directory: {checkpoint_path}")
        return False
    
    print("📁 Directory contents:")
    files = list(checkpoint_path.iterdir())
    for file in sorted(files):
        size = file.stat().st_size if file.is_file() else "DIR"
        print(f"  {file.name:<30} {size}")
    
    return True


def check_config_file(checkpoint_path):
    """Check the config.json file in detail"""
    config_path = Path(checkpoint_path) / "config.json"
    
    print(f"\n=== Config File Analysis ===")
    
    if not config_path.exists():
        print("❌ config.json not found")
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("✅ config.json loaded successfully")
        
        # Key parameters to check
        important_keys = [
            'model_type', 'architectures', 'hidden_size', 'intermediate_size',
            'num_attention_heads', 'num_hidden_layers', 'vocab_size'
        ]
        
        print("🔍 Key configuration parameters:")
        for key in important_keys:
            if key in config:
                print(f"  {key}: {config[key]}")
        
        # Check for vision config
        if 'vision_config' in config:
            print("🎥 Vision configuration found:")
            vision_config = config['vision_config']
            vision_keys = ['hidden_size', 'image_size', 'patch_size']
            for key in vision_keys:
                if key in vision_config:
                    print(f"  vision_{key}: {vision_config[key]}")
        
        return config
        
    except Exception as e:
        print(f"❌ Error loading config.json: {e}")
        return None


def check_model_weights(checkpoint_path):
    """Check the actual model weights to see what size they are"""
    checkpoint_path = Path(checkpoint_path)
    
    print(f"\n=== Model Weights Analysis ===")
    
    # Look for model weight files
    weight_files = []
    weight_files.extend(list(checkpoint_path.glob("*.safetensors")))
    weight_files.extend(list(checkpoint_path.glob("*.bin")))
    weight_files.extend(list(checkpoint_path.glob("pytorch_model*.pt")))
    
    if not weight_files:
        print("❌ No weight files found")
        return
    
    print(f"📦 Found {len(weight_files)} weight files:")
    for wf in weight_files:
        print(f"  {wf.name}")
    
    # Try to load and inspect the first weight file
    try:
        weight_file = weight_files[0]
        print(f"\n🔍 Inspecting: {weight_file.name}")
        
        if weight_file.suffix == '.safetensors':
            from safetensors import safe_open
            with safe_open(weight_file, framework="pt", device="cpu") as f:
                keys = f.keys()
        else:
            # PyTorch file
            checkpoint = torch.load(weight_file, map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                keys = checkpoint['state_dict'].keys()
            elif isinstance(checkpoint, dict):
                keys = checkpoint.keys()
            else:
                print("❌ Unexpected checkpoint format")
                return
        
        # Look for telltale signs of model size
        key_list = list(keys)
        print(f"📊 Found {len(key_list)} parameters")
        
        # Look for specific layer patterns that indicate size
        bias_layers = [k for k in key_list if 'bias' in k and ('mlp' in k or 'ffn' in k)]
        if bias_layers:
            print(f"\n🔍 Found MLP/FFN bias layers (these show the error):")
            for layer in bias_layers[:3]:  # Show first 3
                print(f"  {layer}")
        
        # Try to find the problematic layer
        problematic_layers = [k for k in key_list if 'bias' in k and any(dim in str(k) for dim in ['2048', '1280'])]
        if problematic_layers:
            print(f"\n⚠️  Potential problematic layers:")
            for layer in problematic_layers:
                print(f"  {layer}")
        
        # Check if this looks like a LoRA checkpoint
        lora_keys = [k for k in key_list if 'lora' in k.lower()]
        if lora_keys:
            print(f"\n🎯 LoRA layers detected ({len(lora_keys)} keys)")
            print("This might be a LoRA checkpoint that needs merging!")
            
        adapter_keys = [k for k in key_list if 'adapter' in k.lower()]
        if adapter_keys:
            print(f"\n🔧 Adapter layers detected ({len(adapter_keys)} keys)")
            
    except Exception as e:
        print(f"❌ Error inspecting weights: {e}")


def check_training_args(checkpoint_path):
    """Check for training arguments that might give clues"""
    checkpoint_path = Path(checkpoint_path)
    
    print(f"\n=== Training Arguments Analysis ===")
    
    # Look for training args file
    training_args_file = checkpoint_path / "training_args.bin"
    if training_args_file.exists():
        try:
            training_args = torch.load(training_args_file, map_location='cpu')
            print("✅ training_args.bin found")
            
            # Key parameters to check
            if hasattr(training_args, 'model_name_or_path'):
                print(f"🏷️  Original model: {training_args.model_name_or_path}")
            if hasattr(training_args, 'output_dir'):
                print(f"📁 Output dir: {training_args.output_dir}")
                
        except Exception as e:
            print(f"❌ Error loading training_args.bin: {e}")
    else:
        print("❌ training_args.bin not found")


def suggest_solutions(checkpoint_path):
    """Suggest potential solutions based on findings"""
    print(f"\n=== Potential Solutions ===")
    
    print("Based on the analysis, here are potential fixes:")
    
    print("\n1. 🔍 Verify checkpoint integrity:")
    print(f"   python debug_checkpoint.py --checkpoint_path {checkpoint_path}")
    
    print("\n2. 🔄 Try loading with explicit config:")
    print("   In your inference script, add:")
    print("   config = AutoConfig.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')")
    print("   model = Qwen2VLForConditionalGeneration.from_pretrained(checkpoint_path, config=config)")
    
    print("\n3. 🎯 If this is a LoRA checkpoint, merge it first:")
    print("   python src/merge_lora_weights.py --model_id Qwen/Qwen2.5-VL-3B-Instruct --adapter_model_id {checkpoint_path} --output_path merged_model")
    
    print("\n4. 🏗️  Try loading without strict state dict:")
    print("   model.load_state_dict(checkpoint, strict=False)")
    
    print("\n5. 📂 Check if you're pointing to the right checkpoint:")
    print("   - Make sure it's the final checkpoint, not an intermediate one")
    print("   - Check if there are multiple checkpoint directories")


def main():
    args = parse_args()
    
    print("=== 3B Model Checkpoint Debugger ===")
    print(f"Investigating: {args.checkpoint_path}")
    print("Expected: 3B model (hidden_size=1280)")
    print("Actual error: Found 2048 (7B size) in checkpoint")
    print()
    
    # Step 1: Check directory structure
    if not analyze_checkpoint_structure(args.checkpoint_path):
        return
    
    # Step 2: Check config file
    config = check_config_file(args.checkpoint_path)
    
    # Step 3: Check model weights
    check_model_weights(args.checkpoint_path)
    
    # Step 4: Check training arguments
    check_training_args(args.checkpoint_path)
    
    # Step 5: Suggest solutions
    suggest_solutions(args.checkpoint_path)
    
    print(f"\n{'='*60}")
    print("🎯 SUMMARY:")
    if config and config.get('hidden_size') == 1280:
        print("✅ Config shows 3B model (hidden_size=1280) - config is correct")
        print("❌ But weights show 7B characteristics (bias shape 2048)")
        print("🔍 This suggests:")
        print("   - Weights might be corrupted")
        print("   - This might be a LoRA checkpoint that needs merging")
        print("   - Wrong checkpoint directory selected")
    else:
        print("❌ Config doesn't match expected 3B model")
        print("🔍 This suggests the checkpoint is actually from a different model size")


if __name__ == "__main__":
    main()