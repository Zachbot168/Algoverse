#!/usr/bin/env python3
"""
Test script specifically for StereoSet fix
"""
import json
from datasets.bias_loaders import StereoSetLoader
from eval.unified_evaluator import UnifiedBiasEvaluator
from pathlib import Path

def main():
    print("🔧 Testing StereoSet scoring fix...")
    
    # Initialize StereoSet loader
    data_path = Path(".")
    loader = StereoSetLoader(data_path)
    
    # Try to load some sample data
    try:
        samples = loader.load_data(split="dev", sample_size=10)
        print(f"✓ Loaded {len(samples)} samples")
        
        if samples:
            # Prepare for evaluation
            eval_samples = loader.prepare_for_evaluation(samples)
            print(f"✓ Prepared {len(eval_samples)} evaluation samples")
            
            # Simulate some predictions (indices 0, 1, 2 for stereotypical, anti-stereotypical, unrelated)
            predictions = [0, 1, 2, 1, 0, 2, 1, 1, 2, 0][:len(eval_samples)]
            
            # Create targets in the format expected by compute_metrics
            targets = []
            for eval_sample in eval_samples:
                targets.append({
                    "target": eval_sample["target"],
                    "metadata": eval_sample["metadata"],
                    **eval_sample["original_format"]
                })
            
            # Test compute_metrics
            metrics = loader.compute_metrics(predictions, targets)
            print(f"\n📊 StereoSet Metrics (should NOT be all zeros):")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
                
            # Check if metrics are meaningful
            if metrics.get("stereoset_total_samples", 0) > 0:
                print("\n✅ StereoSet fix appears to be working!")
            else:
                print("\n❌ StereoSet fix may still have issues")
                
        else:
            print("⚠️ No StereoSet samples found - data may not be available")
            
    except Exception as e:
        print(f"❌ Error testing StereoSet: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()