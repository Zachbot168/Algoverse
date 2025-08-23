#!/usr/bin/env python3

import sys
import os
import logging
from pathlib import Path

# Add the project directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from datasets.unified_registry import UnifiedDatasetRegistry
from utils.model_compatibility import create_model_variants
from eval.unified_evaluator import UnifiedEvaluator

def debug_bold_scores():
    """Debug BOLD scoring for different model variants"""
    
    logging.basicConfig(level=logging.INFO)
    
    # Initialize registry
    registry = UnifiedDatasetRegistry("../datasets")
    
    # Load BOLD dataset
    bold_loader = registry.load_dataset("BOLD")
    
    # Load a small sample for testing
    bold_samples = bold_loader.load_data(sample_size=5)  # Very small sample
    print(f"\nLoaded {len(bold_samples)} BOLD samples")
    
    # Create model variants
    model_config = {
        "model_name": "google/gemma-2-2b-it",
        "device_map": "auto",
        "torch_dtype": "float16"
    }
    
    model_variants = create_model_variants(model_config)
    
    # Test each model variant
    for variant_name, (model, tokenizer, metadata) in model_variants.items():
        print(f"\n{'='*50}")
        print(f"Testing {variant_name.upper()} model variant")
        print(f"{'='*50}")
        
        try:
            # Create evaluator for this variant
            evaluator = UnifiedEvaluator(model, tokenizer, {})
            
            # Evaluate BOLD on this variant
            print(f"Running BOLD evaluation on {variant_name}...")
            results = evaluator.evaluate_dataset("BOLD", bold_samples)
            
            print(f"\nResults for {variant_name}:")
            if "detailed_metrics" in results:
                metrics = results["detailed_metrics"]
                bold_bias = metrics.get("bold_sentiment_bias", "NOT_FOUND")
                print(f"  bold_sentiment_bias: {bold_bias}")
                print(f"  bold_toxicity_score: {metrics.get('bold_toxicity_score', 'NOT_FOUND')}")
                print(f"  bold_num_prompts: {metrics.get('bold_num_prompts', 'NOT_FOUND')}")
            else:
                print(f"  No detailed_metrics found in results: {results.keys()}")
                
        except Exception as e:
            print(f"Error evaluating {variant_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*50}")
    print("BOLD Debug Complete")
    print(f"{'='*50}")

if __name__ == "__main__":
    debug_bold_scores()