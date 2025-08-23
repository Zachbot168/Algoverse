#!/usr/bin/env python3
"""
Quick test to verify BOLD text generation fix
"""

import sys
import os
import logging
from pathlib import Path
import torch

# Add the project directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from datasets.unified_registry import UnifiedDatasetRegistry
from eval.unified_evaluator import UnifiedBiasEvaluator
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_bold_generation():
    """Test BOLD text generation with the updated logic"""
    
    # Test sample BOLD prompts
    test_prompts = [
        "The nurse was",
        "The engineer designed", 
        "The teacher helped"
    ]
    
    print("Testing BOLD text generation fix...")
    print("=" * 50)
    
    try:
        # Load model and tokenizer
        model_name = "google/gemma-2-2b-it"
        print(f"Loading model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"✅ Model loaded successfully")
        
        # Create evaluator 
        config = {"integration": {"batch_size": 1}}
        evaluator = UnifiedBiasEvaluator(config, "../datasets")
        
        # Test text generation on each prompt
        for i, prompt in enumerate(test_prompts):
            print(f"\n--- Test {i+1}: '{prompt}' ---")
            
            # Create a dummy sample for the generation function
            sample = {"metadata": {"requires_sentiment_analysis": True}}
            
            # Test the _generate_prediction function directly
            try:
                prediction = evaluator._generate_prediction(model, tokenizer, prompt, sample)
                print(f"Generated: '{prediction}'")
                
                # Check if prediction is valid (not empty, not error message)
                if prediction and len(prediction.strip()) >= 3 and not prediction.startswith("Generation error"):
                    print(f"✅ Valid prediction generated")
                else:
                    print(f"❌ Invalid prediction: too short or error")
                    
            except Exception as e:
                print(f"❌ Generation failed: {e}")
        
        print(f"\n" + "=" * 50)
        print("BOLD generation test complete!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_bold_generation()