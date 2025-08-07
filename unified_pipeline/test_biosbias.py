#!/usr/bin/env python3
"""
Quick test for BiosBias loading issue
"""
import sys
from pathlib import Path
sys.path.append('.')

from datasets.bias_loaders import BiossBiasLoader

def main():
    print("🔧 Testing BiosBias loader...")
    
    # Create loader
    loader = BiossBiasLoader(Path("."))
    
    try:
        # Test data loading
        print("Loading data...")
        samples = loader.load_data(split="test", sample_size=10)
        print(f"✓ Loaded {len(samples)} samples")
        
        if samples:
            print(f"Sample data: {samples[0]}")
            
            # Test preparation for evaluation
            eval_samples = loader.prepare_for_evaluation(samples)
            print(f"✓ Prepared {len(eval_samples)} evaluation samples")
            
            if eval_samples:
                print(f"Evaluation sample: {eval_samples[0]}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()