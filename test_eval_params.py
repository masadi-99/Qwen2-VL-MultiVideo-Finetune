#!/usr/bin/env python3
"""
Test which evaluation parameters are actually supported
"""

import inspect
from transformers import TrainingArguments as HFTrainingArguments

def check_evaluation_parameters():
    """Check what evaluation parameters exist in HFTrainingArguments"""
    
    # Get all parameters of HFTrainingArguments
    sig = inspect.signature(HFTrainingArguments.__init__)
    params = list(sig.parameters.keys())
    
    print("=== HFTrainingArguments Parameters ===")
    eval_params = [p for p in params if 'eval' in p.lower()]
    
    print("\nEvaluation-related parameters:")
    for param in sorted(eval_params):
        print(f"  {param}")
    
    # Check specific parameters we're trying to use
    test_params = [
        'do_eval', 'evaluation_strategy', 'eval_strategy', 
        'eval_steps', 'per_device_eval_batch_size',
        'load_best_model_at_end', 'metric_for_best_model'
    ]
    
    print("\nTesting specific parameters:")
    for param in test_params:
        exists = param in params
        print(f"  {param:<25} {'✅' if exists else '❌'}")
    
    # Test creating an instance with eval parameters
    print("\n=== Testing Parameter Creation ===")
    try:
        args = HFTrainingArguments(
            output_dir="/tmp/test",
            do_eval=True,
            evaluation_strategy="steps", 
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1
        )
        print("✅ HFTrainingArguments with eval params: SUCCESS")
        print(f"   do_eval: {args.do_eval}")
        print(f"   evaluation_strategy: {args.evaluation_strategy}")
        print(f"   eval_steps: {args.eval_steps}")
        
    except Exception as e:
        print(f"❌ HFTrainingArguments with eval params: FAILED")
        print(f"   Error: {str(e)}")
    
    # Test with load_best_model_at_end
    try:
        args = HFTrainingArguments(
            output_dir="/tmp/test",
            do_eval=True,
            evaluation_strategy="steps", 
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss"
        )
        print("✅ HFTrainingArguments with best model: SUCCESS")
        
    except Exception as e:
        print(f"❌ HFTrainingArguments with best model: FAILED")
        print(f"   Error: {str(e)}")

if __name__ == "__main__":
    check_evaluation_parameters()