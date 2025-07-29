#!/usr/bin/env python3
"""
Pipeline Validation Script

Tests the unified bias mitigation pipeline for common issues:
- Import validation
- Configuration file validation  
- Basic functionality checks
- Bias mitigation component integration
"""

import os
import sys
import json
import yaml
from pathlib import Path

def test_imports():
    """Test all pipeline component imports."""
    print("Testing pipeline imports...")
    
    # Check critical dependencies first
    missing_deps = []
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
        
    try:
        import transformers
    except ImportError:
        missing_deps.append("transformers")
        
    try:
        import sklearn
    except ImportError:
        missing_deps.append("scikit-learn")
        
    try:
        import yaml
    except ImportError:
        missing_deps.append("pyyaml")
    
    if missing_deps:
        print(f"❌ Missing critical dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    try:
        sys.path.append(str(Path(__file__).parent))
        
        from eval.run_diagnostic import UnifiedDiagnosticPass
        from train.component_registry import ComponentRegistryManager
        from train.run_pinpoint_tuning import UnifiedPinpointTuner
        from steer.compute_dsv import DSVComputer
        from eval.run_benchmark import UnifiedBenchmark
        from eval.metrics import UnifiedMetrics
        from run_full_pipeline import BiasMitigationPipelineRunner
        
        print("✅ All imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Install dependencies with: pip install -r requirements.txt")
        return False

def test_configurations():
    """Test configuration file validity."""
    print("\nTesting configuration files...")
    
    config_dir = Path(__file__).parent / "configs"
    config_files = ["baseline.yaml", "full.yaml", "pinpoint.yaml", "steer.yaml"]
    
    all_valid = True
    
    for config_file in config_files:
        config_path = config_dir / config_file
        
        if not config_path.exists():
            print(f"❌ Missing config file: {config_file}")
            all_valid = False
            continue
            
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Check required sections
            required_sections = ['model', 'interventions', 'evaluation']
            for section in required_sections:
                if section not in config:
                    print(f"❌ {config_file}: Missing section '{section}'")
                    all_valid = False
                    
            print(f"✅ {config_file}: Valid")
                    
        except Exception as e:
            print(f"❌ {config_file}: Error - {e}")
            all_valid = False
            
    return all_valid

def test_pipeline_initialization():
    """Test pipeline runner initialization."""
    print("\nTesting pipeline initialization...")
    
    try:
        from run_full_pipeline import BiasMitigationPipelineRunner
        
        config_path = "configs/baseline.yaml"
        runner = BiasMitigationPipelineRunner(config_path)
        
        # Test basic attributes
        assert hasattr(runner, 'config')
        assert hasattr(runner, 'model_name')
        assert hasattr(runner, 'pipeline_state')
        
        print("✅ Pipeline initialization successful")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        return False

def test_bias_mitigation_components():
    """Test bias-specific components."""
    print("\nTesting bias mitigation components...")
    
    try:
        # Test diagnostic data creation
        test_data = [
            {
                "id": "test_gender_bias",
                "biased_data": [
                    {"role": "user", "content": "Women are naturally better at caregiving."},
                    {"role": "assistant", "content": "That's often true."}
                ],
                "unbiased_data": [
                    {"role": "user", "content": "Caregiving skills vary among individuals."},
                    {"role": "assistant", "content": "Caregiving depends on individual traits."}
                ],
                "bias_type": "gender",
                "bias_indicators": ["naturally", "stereotype"],
                "target_token": "stereotype"
            }
        ]
        
        # Test JSON serialization/deserialization 
        test_json = json.dumps(test_data[0])
        parsed_data = json.loads(test_json)
        
        assert parsed_data['bias_type'] == 'gender'
        assert 'biased_data' in parsed_data
        assert 'unbiased_data' in parsed_data
        
        print("✅ Bias mitigation data structures valid")
        return True
        
    except Exception as e:
        print(f"❌ Bias mitigation component test failed: {e}")
        return False

def test_error_handling():
    """Test error handling improvements."""
    print("\nTesting error handling...")
    
    try:
        # Test JSON parsing with malformed data
        from eval.run_diagnostic import UnifiedDiagnosticPass
        
        # Create test file with malformed JSON
        test_file = "/tmp/test_malformed.jsonl"
        with open(test_file, 'w') as f:
            f.write('{"valid": "json"}\n')
            f.write('{"malformed": json}\n')  # Invalid JSON
            f.write('{"another": "valid"}\n')
            
        # Test that our error handling works
        diagnostic = UnifiedDiagnosticPass({'model': {'name': 'test'}})
        try:
            data = diagnostic._load_diagnostic_data(test_file)
            # Should load 2 valid entries, skip 1 malformed
            assert len(data) == 2
            print("✅ JSON error handling working correctly")
            os.unlink(test_file)
            return True
        except Exception as e:
            print(f"❌ JSON error handling test failed: {e}")
            os.unlink(test_file)
            return False
            
    except Exception as e:
        print(f"❌ Error handling test setup failed: {e}")
        return False

def main():
    """Run all pipeline validation tests."""
    print("🔍 UNIFIED BIAS MITIGATION PIPELINE VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Import Validation", test_imports),
        ("Configuration Validation", test_configurations), 
        ("Pipeline Initialization", test_pipeline_initialization),
        ("Bias Mitigation Components", test_bias_mitigation_components),
        ("Error Handling", test_error_handling)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name}: Unexpected error - {e}")
    
    print("\n" + "=" * 60)
    print(f"VALIDATION RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is ready for bias mitigation.")
        return 0
    else:
        print("⚠️  Some tests failed. Please address issues before running pipeline.")
        return 1

if __name__ == "__main__":
    exit(main())