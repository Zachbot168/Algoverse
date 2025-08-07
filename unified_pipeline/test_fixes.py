#!/usr/bin/env python3
"""
Quick test script to verify StereoSet and SycophancyEval fixes
"""
import sys
import os
sys.path.append('.')

from run_unified_pipeline import UnifiedBiasMitigationPipeline
import warnings
warnings.filterwarnings("ignore")

def main():
    print("🔧 Testing fixes for StereoSet and SycophancyEval...")
    
    # Create pipeline with GPT-2
    pipeline = UnifiedBiasMitigationPipeline(
        'configs/models/gpt2-small.yaml',
        'configs/datasets.yaml'
    )
    
    # Test on a small sample
    print("Running quick evaluation on StereoSet and SycophancyEval...")
    
    # Load only the problematic datasets
    test_suite_config = {
        'description': 'Test suite for fixes',
        'datasets': ['StereoSet', 'SycophancyEval']
    }
    
    # Override the dataset configs temporarily
    pipeline.unified_evaluator.dataset_configs['test_suite'] = test_suite_config
    
    # Run evaluation
    try:
        results = pipeline._run_bias_evaluation('test_suite')
        
        # Check results
        for dataset in ['StereoSet', 'SycophancyEval']:
            if dataset in results['dataset_results']:
                metrics = results['dataset_results'][dataset]['metrics']
                print(f"\n{dataset} Results:")
                for key, value in metrics.items():
                    if not key.endswith('_time') and not key.endswith('_rate'):
                        print(f"  {key}: {value}")
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()