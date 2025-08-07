#!/usr/bin/env python3
"""
Test BiosBias loading specifically to identify the issue.
"""
import sys
from pathlib import Path
sys.path.append('.')

def test_biosbias_loading():
    try:
        # Test registry loading
        from datasets import UnifiedDatasetRegistry
        registry = UnifiedDatasetRegistry("/workspace/Algoverse")
        
        print("✓ Registry initialized successfully")
        print(f"Available datasets: {list(registry.dataset_loaders.keys())}")
        print(f"BiosBias in registry: {'BiosBias' in registry.dataset_loaders}")
        
        # Test direct loader instantiation
        from datasets.bias_loaders import BiossBiasLoader
        loader = BiossBiasLoader(Path("."))
        print("✓ Direct BiosBias loader instantiation successful")
        
        # Test registry load_dataset
        print("\n--- Testing registry.load_dataset ---")
        loaded_loader = registry.load_dataset("BiosBias", {})
        print("✓ Registry load_dataset successful")
        
        # Test data loading
        print("\n--- Testing data loading ---")
        data = loaded_loader.load_data(split="test", sample_size=3)
        print(f"✓ Data loading successful: {len(data)} samples")
        if data:
            print(f"Sample data: {data[0]}")
        
        # Test evaluation preparation
        print("\n--- Testing evaluation preparation ---")
        eval_samples = loaded_loader.prepare_for_evaluation(data)
        print(f"✓ Evaluation preparation successful: {len(eval_samples)} samples")
        if eval_samples:
            print(f"Sample eval data: {eval_samples[0]}")
            
    except Exception as e:
        print(f"❌ Error during BiosBias loading test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_biosbias_loading()