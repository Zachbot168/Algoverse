#!/usr/bin/env python3
"""
Installation Test for Algoverse FIRM Framework
Verifies that all core components can be imported and basic functionality works.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_core_imports():
    """Test that core modules can be imported."""
    print("🔍 Testing core imports...")
    
    try:
        # Test dataset loaders
        from datasets.base_loader import BaseDatasetLoader, BiasType, EvaluationMode
        from datasets.bias_loaders import CrowsPairsLoader, WinoBiasLoader
        print("✅ Dataset loaders imported successfully")
        
        # Test evaluation components
        from eval.unified_evaluator import UnifiedBiasEvaluator
        from eval.real_bias_evaluator import RealBiasEvaluator
        print("✅ Evaluation components imported successfully")
        
        # Test causal analysis
        from causal_analysis.bias_circuit_tracer import BiasCircuitTracer
        from causal_analysis.real_circuit_identification import RealCircuitIdentifier
        print("✅ Causal analysis components imported successfully")
        
        # Test steering components
        from steer.das_wrapper import BiasActivationDetector
        from steer.real_steering_vectors import RealSteeringVectorComputer
        print("✅ Steering components imported successfully")
        
        # Test training components
        from train.component_registry import ComponentRegistryManager
        print("✅ Training components imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_dependencies():
    """Test that key dependencies are available."""
    print("\n🔍 Testing dependencies...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
        
        import scipy
        print(f"✅ SciPy {scipy.__version__}")
        
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
        
        # Test CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available - using CPU mode")
        
        return True
        
    except ImportError as e:
        print(f"❌ Dependency missing: {e}")
        return False

def test_dataset_availability():
    """Test dataset availability."""
    print("\n🔍 Testing dataset availability...")
    
    datasets_dir = Path(__file__).parent.parent / "datasets"
    
    if not datasets_dir.exists():
        print("❌ Datasets directory not found")
        return False
    
    expected_datasets = [
        "crows-pairs",
        "winobias", 
        "winogender",
        "bbq",
        "stereoset",
        "truthfulqa"
    ]
    
    available_count = 0
    for dataset in expected_datasets:
        dataset_path = datasets_dir / dataset
        if dataset_path.exists():
            print(f"✅ {dataset} found")
            available_count += 1
        else:
            print(f"⚠️  {dataset} not found")
    
    print(f"\n📊 Datasets: {available_count}/{len(expected_datasets)} available")
    
    if available_count >= len(expected_datasets) // 2:
        print("✅ Sufficient datasets for testing")
        return True
    else:
        print("⚠️  Limited datasets - run ./enhanced_pull_datasets.sh")
        return False

def test_configs():
    """Test configuration files."""
    print("\n🔍 Testing configuration files...")
    
    configs_dir = Path(__file__).parent / "configs"
    
    # Check model configs
    models_dir = configs_dir / "models"
    if models_dir.exists():
        model_configs = list(models_dir.glob("*.yaml"))
        print(f"✅ Found {len(model_configs)} model configurations")
    else:
        print("❌ Model configurations directory not found")
        return False
    
    # Check dataset config
    dataset_config = configs_dir / "datasets.yaml"
    if dataset_config.exists():
        print("✅ Dataset configuration found")
    else:
        print("❌ Dataset configuration not found")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality."""
    print("\n🔍 Testing basic functionality...")
    
    try:
        # Test BiasType enum
        from datasets.base_loader import BiasType
        assert BiasType.GENDER.value == "gender"
        print("✅ BiasType enum working")
        
        # Test ComponentRegistry
        from train.component_registry import ComponentRegistryManager
        registry_manager = ComponentRegistryManager()
        print("✅ ComponentRegistry instantiation working")
        
        # Test dataset validation
        from datasets.data_validator import DatasetValidator
        validator = DatasetValidator("../datasets")
        print("✅ DatasetValidator instantiation working")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Algoverse Installation Test")
    print("=" * 40)
    
    tests = [
        ("Core Imports", test_core_imports),
        ("Dependencies", test_dependencies), 
        ("Dataset Availability", test_dataset_availability),
        ("Configuration Files", test_configs),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        if test_func():
            passed += 1
    
    print(f"\n📊 Test Summary")
    print("=" * 40)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Installation is working correctly.")
        print("\nNext steps:")
        print("1. Run dataset download: cd .. && ./enhanced_pull_datasets.sh")
        print("2. Authenticate with HuggingFace: huggingface-cli login")
        print("3. Run quick evaluation: python run_unified_pipeline.py --model-config configs/models/gemma-2-2b-it.yaml --suite quick_evaluation")
        return 0
    elif passed >= total * 0.8:
        print("✅ Most tests passed! Installation is mostly working.")
        print("⚠️  Address any warnings above before running evaluations.")
        return 0
    else:
        print("❌ Multiple test failures. Please check your installation.")
        print("Try running: python setup.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())