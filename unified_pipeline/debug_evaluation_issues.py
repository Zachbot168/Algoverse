#!/usr/bin/env python3
"""
Debug the fundamental evaluation issues where models aren't understanding scenarios
"""
import sys
import json
from pathlib import Path
sys.path.append('.')

def debug_winobias():
    """Debug WinoBias complete failure"""
    print("🔍 Debugging WinoBias...")
    
    from datasets.bias_loaders import WinoBiasLoader
    loader = WinoBiasLoader(Path("."))
    
    # Test data loading
    samples = loader.load_data(split="test", sample_size=3)
    print(f"✓ Loaded {len(samples)} samples")
    if samples:
        print(f"Sample: {samples[0]}")
    
    # Test evaluation preparation
    eval_samples = loader.prepare_for_evaluation(samples)
    print(f"✓ Prepared {len(eval_samples)} evaluation samples")
    if eval_samples:
        print(f"Eval sample: {eval_samples[0]}")
    
    # Test with mock predictions to see compute_metrics
    mock_predictions = ["developer", "nurse", "CEO"]
    mock_targets = eval_samples
    
    metrics = loader.compute_metrics(mock_predictions, mock_targets)
    print(f"Mock metrics: {metrics}")

def debug_biosbias():
    """Debug BiosBias 0 samples issue"""
    print("\n🔍 Debugging BiosBias...")
    
    from datasets.bias_loaders import BiossBiasLoader  
    loader = BiossBiasLoader(Path("."))
    
    # Test data loading
    samples = loader.load_data(split="test", sample_size=3)
    print(f"✓ Loaded {len(samples)} samples")
    if samples:
        print(f"Sample: {samples[0]}")
    else:
        print("❌ No samples loaded!")
        
def debug_seat():
    """Debug SEAT suspicious uniformity"""
    print("\n🔍 Debugging SEAT...")
    
    from datasets.bias_loaders import SEATLoader
    loader = SEATLoader(Path("."))
    
    try:
        samples = loader.load_data(split="test", sample_size=3)
        print(f"✓ Loaded {len(samples)} samples")
        if samples:
            print(f"Sample: {samples[0]}")
        
        # Test with mock predictions
        mock_predictions = ["This test shows bias patterns", "No bias detected here", "Strong gender associations"]
        mock_targets = samples
        
        metrics = loader.compute_metrics(mock_predictions, mock_targets)
        print(f"Mock metrics: {metrics}")
        
    except Exception as e:
        print(f"❌ SEAT error: {e}")

def debug_model_responses():
    """Debug what the model is actually producing"""
    print("\n🔍 Debugging model response patterns...")
    
    # Load recent results to see what model actually generated
    try:
        results_file = Path("unified_pipeline_runs/20250807_035249/evaluation/baseline/evaluation_results.json")
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
            
            # Check WinoBias predictions
            if "WinoBias" in results["dataset_results"]:
                winobias_preds = results["dataset_results"]["WinoBias"]["predictions"][:5]
                print(f"WinoBias predictions: {winobias_preds}")
                
            # Check BiosBias
            if "BiosBias" in results["dataset_results"]:
                biosbias_preds = results["dataset_results"]["BiosBias"]["predictions"][:5] 
                print(f"BiosBias predictions: {biosbias_preds}")
                
            # Check SEAT
            if "SEAT" in results["dataset_results"]:
                seat_preds = results["dataset_results"]["SEAT"]["predictions"][:3]
                print(f"SEAT predictions: {seat_preds}")
                
    except Exception as e:
        print(f"❌ Error loading results: {e}")

if __name__ == "__main__":
    debug_winobias()
    debug_biosbias() 
    debug_seat()
    debug_model_responses()