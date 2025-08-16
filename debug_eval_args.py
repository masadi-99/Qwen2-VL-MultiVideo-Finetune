#!/usr/bin/env python3
"""
Debug script to test evaluation argument parsing
"""

import sys
from transformers import HfArgumentParser
from src.params import DataArguments, ModelArguments, TrainingArguments

def test_eval_args():
    # Simulate the arguments from the training script
    test_args = [
        "--model_id", "Qwen/Qwen2.5-VL-3B-Instruct",
        "--data_path", "/tmp/train.json",
        "--eval_path", "/tmp/eval.json", 
        "--image_folder", "/",
        "--eval_image_folder", "/",
        "--do_eval",
        "--evaluation_strategy", "steps",
        "--eval_steps", "50",
        "--eval_accumulation_steps", "1",
        "--per_device_eval_batch_size", "1",
        "--eval_delay", "0",
        "--output_dir", "/tmp/output",
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "2",
        "--num_train_epochs", "1",
        "--logging_steps", "10",
        "--save_strategy", "steps",
        "--save_steps", "50",
        "--bf16", "True",
        "--tf32", "True",
        "--report_to", "wandb"
    ]
    
    # Override sys.argv for testing
    original_argv = sys.argv
    sys.argv = ["debug_script"] + test_args
    
    try:
        parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
        print("✅ Arguments parsed successfully!")
        print(f"do_eval: {training_args.do_eval}")
        print(f"evaluation_strategy: {training_args.evaluation_strategy}")
        print(f"save_strategy: {training_args.save_strategy}")
        print(f"eval_steps: {training_args.eval_steps}")
        print(f"save_steps: {training_args.save_steps}")
        print(f"eval_path: {data_args.eval_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error parsing arguments: {str(e)}")
        return False
        
    finally:
        sys.argv = original_argv

if __name__ == "__main__":
    test_eval_args()