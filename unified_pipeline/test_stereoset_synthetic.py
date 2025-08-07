#!/usr/bin/env python3
"""
Synthetic test for StereoSet compute_metrics fix
"""
from datasets.bias_loaders import StereoSetLoader
from pathlib import Path

def main():
    print("🔧 Testing StereoSet compute_metrics with synthetic data...")
    
    # Initialize loader
    loader = StereoSetLoader(Path("."))
    
    # Create synthetic targets that mimic the expected structure
    synthetic_targets = [
        {
            "target": 2,  # Index of best choice
            "metadata": {"labels": ["stereotype", "anti-stereotype", "unrelated"]},
            "sentences": [
                {"sentence": "Stereotypical sentence", "labels": [{"label": "stereotype"}]},
                {"sentence": "Anti-stereotypical sentence", "labels": [{"label": "anti-stereotype"}]},
                {"sentence": "Unrelated sentence", "labels": [{"label": "unrelated"}]}
            ]
        },
        {
            "target": 1,  
            "metadata": {"labels": ["stereotype", "anti-stereotype", "unrelated"]},
            "sentences": [
                {"sentence": "Another stereotypical", "labels": [{"label": "stereotype"}]},
                {"sentence": "Another anti-stereotypical", "labels": [{"label": "anti-stereotype"}]},
                {"sentence": "Another unrelated", "labels": [{"label": "unrelated"}]}
            ]
        },
        {
            "target": 1,
            "metadata": {"labels": ["stereotype", "anti-stereotype"]},
            "sentences": [
                {"sentence": "Biased statement", "labels": [{"label": "stereotype"}]},
                {"sentence": "Unbiased statement", "labels": [{"label": "anti-stereotype"}]}
            ]
        }
    ]
    
    # Test predictions: mix of stereotypical (0), anti-stereotypical (1), unrelated (2)
    predictions = [0, 1, 1]  # First chose stereotypical, second and third chose anti-stereotypical
    
    print(f"Testing with {len(predictions)} predictions and {len(synthetic_targets)} targets")
    
    try:
        # Test the compute_metrics method
        metrics = loader.compute_metrics(predictions, synthetic_targets)
        
        print(f"\n📊 StereoSet Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        # Verify the metrics are reasonable
        expected_checks = [
            ("stereoset_total_samples", 3),
            ("stereotype_pct", lambda x: 0 <= x <= 1),
            ("anti_stereotype_pct", lambda x: 0 <= x <= 1),
        ]
        
        all_good = True
        for key, expected in expected_checks:
            if key in metrics:
                value = metrics[key]
                if callable(expected):
                    if not expected(value):
                        print(f"❌ {key} = {value} failed validation")
                        all_good = False
                elif value != expected:
                    print(f"❌ {key} = {value}, expected {expected}")
                    all_good = False
                else:
                    print(f"✓ {key} = {value}")
            else:
                print(f"❌ Missing metric: {key}")
                all_good = False
        
        if all_good and metrics.get("stereoset_total_samples", 0) > 0:
            print("\n✅ StereoSet compute_metrics fix is working correctly!")
        else:
            print("\n⚠️ Some issues detected in StereoSet compute_metrics")
            
    except Exception as e:
        print(f"❌ Error in compute_metrics: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()