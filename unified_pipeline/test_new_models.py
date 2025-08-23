#!/usr/bin/env python3
"""
Test script for new model compatibility
Tests all newly added models with honest evaluation.
"""

import sys
import os
from pathlib import Path
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add the project directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_model_loading(config_path):
    """Test loading a model from config file"""
    print(f"\n{'='*60}")
    print(f"Testing model: {config_path}")
    print('='*60)
    
    try:
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_name = config['model_name']
        print(f"Model: {model_name}")
        
        # Test model loading
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=config.get('trust_remote_code', True)
        )
        
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=getattr(torch, config.get('torch_dtype', 'float16')),
            device_map=config.get('device_map', 'auto'),
            trust_remote_code=config.get('trust_remote_code', True)
        )
        
        # Get model info
        num_layers = getattr(model.config, 'num_hidden_layers', 'unknown')
        num_heads = getattr(model.config, 'num_attention_heads', 'unknown') 
        hidden_size = getattr(model.config, 'hidden_size', 'unknown')
        
        print(f"✅ Model loaded successfully!")
        print(f"   Architecture: {num_layers} layers, {num_heads} heads, {hidden_size} hidden size")
        print(f"   Generation capable: {hasattr(model, 'generate')}")
        print(f"   Device: {next(model.parameters()).device}")
        
        # Test simple generation
        print("\nTesting text generation...")
        test_prompt = "The doctor was"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        if hasattr(model, 'generate'):
            # Move inputs to same device as model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=inputs["input_ids"].shape[1] + 20,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = generated_text[len(test_prompt):].strip()
            print(f"   Prompt: '{test_prompt}'")
            print(f"   Generated: '{completion}'")
            print("✅ Text generation working!")
        else:
            print("❌ Model does not support text generation")
            return False
        
        # Clean up GPU memory
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test all new model configurations"""
    
    print("🔬 Testing New Model Compatibility")
    print("=" * 60)
    
    # List of new model configs to test
    new_models = [
        "configs/models/qwen2.5-1.5b-instruct.yaml",
        "configs/models/qwen2.5-3b-instruct.yaml", 
        # Note: Llama models require HuggingFace authentication
        # "configs/models/llama-3.2-1b-instruct.yaml",
        # "configs/models/llama-3.2-3b-instruct.yaml",
    ]
    
    results = {}
    
    for model_config in new_models:
        config_path = Path(__file__).parent / model_config
        if config_path.exists():
            model_name = config_path.stem
            results[model_name] = test_model_loading(config_path)
        else:
            print(f"❌ Config file not found: {config_path}")
            results[model_config] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("COMPATIBILITY TEST RESULTS")
    print('='*60)
    
    working_models = []
    failed_models = []
    
    for model_name, success in results.items():
        status = "✅ WORKING" if success else "❌ FAILED"
        print(f"{model_name:30} {status}")
        
        if success:
            working_models.append(model_name)
        else:
            failed_models.append(model_name)
    
    print(f"\n📊 Summary:")
    print(f"   Working models: {len(working_models)}")
    print(f"   Failed models:  {len(failed_models)}")
    
    if working_models:
        print(f"\n✅ Successfully compatible models:")
        for model in working_models:
            print(f"   - {model}")
    
    if failed_models:
        print(f"\n❌ Models requiring fixes:")
        for model in failed_models:
            print(f"   - {model}")
    
    return len(failed_models) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)