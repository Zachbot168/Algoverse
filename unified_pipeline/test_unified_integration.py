#!/usr/bin/env python3
"""
Test Script for Unified Dataset Integration

Tests the comprehensive dataset integration system to ensure all datasets
are properly loaded and can be evaluated while preserving their unique
characteristics.
"""

import os
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import components
from datasets import UnifiedDatasetRegistry
from eval.unified_evaluator import UnifiedBiasEvaluator
import yaml


def test_registry_initialization():
    """Test unified dataset registry initialization."""
    print("=" * 50)
    print("TESTING: Dataset Registry Initialization")
    print("=" * 50)
    
    try:
        base_path = "/workspace/Algoverse"
        registry = UnifiedDatasetRegistry(base_path)
        
        print(f"✓ Registry initialized successfully")
        print(f"  Total datasets: {len(registry.get_available_datasets())}")
        print(f"  Working datasets: {len(registry.get_working_datasets())}")
        print(f"  High priority: {len(registry.get_high_priority_datasets())}")
        print(f"  Medium priority: {len(registry.get_medium_priority_datasets())}")
        print(f"  Low priority: {len(registry.get_low_priority_datasets())}")
        
        return True, registry
        
    except Exception as e:
        print(f"✗ Registry initialization failed: {e}")
        traceback.print_exc()
        return False, None


def test_dataset_availability(registry):
    """Test dataset availability validation."""
    print("\n" + "=" * 50)
    print("TESTING: Dataset Availability Validation")
    print("=" * 50)
    
    try:
        availability = registry.validate_dataset_availability()
        
        available_count = sum(availability.values())
        total_count = len(availability)
        
        print(f"Dataset Availability: {available_count}/{total_count}")
        print("\nDetailed Status:")
        
        for dataset, available in availability.items():
            status = "✓ Available" if available else "✗ Missing"
            priority = "Working" if dataset in registry.WORKING_DATASETS else \
                      "High" if dataset in registry.HIGH_PRIORITY else \
                      "Medium" if dataset in registry.MEDIUM_PRIORITY else \
                      "Low" if dataset in registry.LOW_PRIORITY else "Unknown"
            
            print(f"  {dataset:<15} {status:<12} ({priority} Priority)")
        
        return True, availability
        
    except Exception as e:
        print(f"✗ Dataset availability check failed: {e}")
        traceback.print_exc()
        return False, {}


def test_individual_dataset_loading(registry, availability):
    """Test loading individual datasets."""
    print("\n" + "=" * 50)
    print("TESTING: Individual Dataset Loading")
    print("=" * 50)
    
    # Test loading available datasets
    available_datasets = [name for name, avail in availability.items() if avail]
    
    if not available_datasets:
        print("No datasets available for testing")
        return False, {}
    
    loading_results = {}
    
    for dataset_name in available_datasets:
        print(f"\n--- Testing {dataset_name} ---")
        
        try:
            # Load dataset
            loader = registry.load_dataset(dataset_name)
            
            # Get metadata
            metadata = loader.get_metadata()
            print(f"✓ Loaded successfully")
            print(f"  Size: {metadata.size}")
            print(f"  Bias types: {[bt.value for bt in metadata.bias_types]}")
            print(f"  Evaluation mode: {metadata.evaluation_mode.value}")
            print(f"  Requires generation: {metadata.requires_generation}")
            
            # Try loading a small sample
            try:
                samples = loader.load_data(sample_size=5)
                print(f"  Sample data: {len(samples)} samples loaded")
                
                # Try preparing for evaluation
                prepared = loader.prepare_for_evaluation(samples[:3])
                print(f"  Evaluation prep: {len(prepared)} samples prepared")
                
                loading_results[dataset_name] = {
                    'status': 'success',
                    'metadata': {
                        'size': metadata.size,
                        'bias_types': [bt.value for bt in metadata.bias_types],
                        'evaluation_mode': metadata.evaluation_mode.value,
                        'requires_generation': metadata.requires_generation
                    },
                    'sample_count': len(samples),
                    'prepared_count': len(prepared)
                }
                
            except Exception as e:
                print(f"  ⚠ Data loading failed: {e}")
                loading_results[dataset_name] = {
                    'status': 'metadata_only',
                    'error': str(e)
                }
        
        except Exception as e:
            print(f"✗ Failed to load {dataset_name}: {e}")
            loading_results[dataset_name] = {
                'status': 'failed',
                'error': str(e)
            }
    
    successful_loads = sum(1 for r in loading_results.values() if r['status'] == 'success')
    total_attempts = len(loading_results)
    
    print(f"\nLoading Summary: {successful_loads}/{total_attempts} datasets loaded successfully")
    
    return successful_loads > 0, loading_results


def test_evaluation_suite_configurations():
    """Test evaluation suite configurations."""
    print("\n" + "=" * 50)
    print("TESTING: Evaluation Suite Configurations")
    print("=" * 50)
    
    try:
        # Load dataset configuration
        config_path = os.path.join(os.path.dirname(__file__), "configs/datasets.yaml")
        if not os.path.exists(config_path):
            print(f"✗ Dataset config not found: {config_path}")
            return False, {}
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        evaluation_suites = config.get('evaluation_suites', {})
        
        print(f"Found {len(evaluation_suites)} evaluation suites:")
        
        suite_results = {}
        
        for suite_name, suite_config in evaluation_suites.items():
            datasets = suite_config.get('datasets', [])
            expected_samples = suite_config.get('expected_samples', 'Unknown')
            description = suite_config.get('description', '')
            
            print(f"\n{suite_name}:")
            print(f"  Description: {description}")
            print(f"  Datasets: {len(datasets)}")
            print(f"  Expected samples: {expected_samples}")
            print(f"  Dataset list: {', '.join(datasets)}")
            
            suite_results[suite_name] = {
                'dataset_count': len(datasets),
                'datasets': datasets,
                'expected_samples': expected_samples
            }
        
        return True, suite_results
        
    except Exception as e:
        print(f"✗ Suite configuration test failed: {e}")
        traceback.print_exc()
        return False, {}


def test_unified_evaluator_initialization():
    """Test unified evaluator initialization."""
    print("\n" + "=" * 50)
    print("TESTING: Unified Evaluator Initialization")
    print("=" * 50)
    
    try:
        # Load dataset configuration
        config_path = os.path.join(os.path.dirname(__file__), "configs/datasets.yaml")
        base_path = "/workspace/Algoverse"
        
        if not os.path.exists(config_path):
            print(f"✗ Dataset config not found: {config_path}")
            return False, None
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize evaluator
        evaluator = UnifiedBiasEvaluator(config, base_path)
        
        print(f"✓ Unified evaluator initialized successfully")
        print(f"  Registry: {evaluator.registry}")
        print(f"  Dataset configs: {len(evaluator.dataset_configs)}")
        print(f"  Evaluation suites: {len(evaluator.evaluation_suites)}")
        
        return True, evaluator
        
    except Exception as e:
        print(f"✗ Unified evaluator initialization failed: {e}")
        traceback.print_exc()
        return False, None


def test_evaluation_data_preparation(evaluator):
    """Test evaluation data preparation."""
    print("\n" + "=" * 50)
    print("TESTING: Evaluation Data Preparation")
    print("=" * 50)
    
    try:
        # Test with quick evaluation suite
        print("Preparing data for 'quick_evaluation' suite...")
        
        prepared_data = evaluator.prepare_evaluation_data(suite_name="quick_evaluation")
        
        if prepared_data:
            print(f"✓ Data preparation successful")
            print(f"  Datasets prepared: {len(prepared_data)}")
            
            total_samples = sum(len(samples) for samples in prepared_data.values())
            print(f"  Total samples: {total_samples}")
            
            print("\nDataset breakdown:")
            for dataset_name, samples in prepared_data.items():
                print(f"  {dataset_name}: {len(samples)} samples")
                
                # Show sample structure
                if samples:
                    sample = samples[0]
                    print(f"    Sample keys: {list(sample.keys())}")
                    print(f"    Bias type: {sample.get('bias_type', 'Unknown')}")
                    print(f"    Evaluation mode: {sample.get('evaluation_mode', 'Unknown')}")
            
            return True, prepared_data
        else:
            print("✗ No data prepared")
            return False, {}
            
    except Exception as e:
        print(f"✗ Data preparation failed: {e}")
        traceback.print_exc()
        return False, {}


def generate_test_report(results):
    """Generate comprehensive test report."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE TEST REPORT")
    print("=" * 60)
    
    # Summarize test results
    total_tests = len(results)
    passed_tests = sum(1 for success, _ in results.values() if success)
    
    print(f"Overall Test Results: {passed_tests}/{total_tests} tests passed")
    
    # Detailed results
    for test_name, (success, data) in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"\n{test_name}: {status}")
        
        if test_name == "dataset_availability" and data:
            available = sum(data.values())
            total = len(data)
            print(f"  Available datasets: {available}/{total}")
            
        elif test_name == "dataset_loading" and data:
            successful = sum(1 for r in data.values() if r.get('status') == 'success')
            total = len(data)
            print(f"  Successfully loaded: {successful}/{total}")
            
        elif test_name == "suite_configs" and data:
            print(f"  Evaluation suites configured: {len(data)}")
            
        elif test_name == "data_preparation" and data:
            datasets = len(data)
            total_samples = sum(len(samples) for samples in data.values())
            print(f"  Datasets prepared: {datasets}")
            print(f"  Total samples: {total_samples}")
    
    # Integration status summary
    print(f"\n" + "=" * 40)
    print("INTEGRATION STATUS SUMMARY")
    print("=" * 40)
    
    if "registry_init" in results and results["registry_init"][0]:
        registry = results["registry_init"][1]
        
        working = registry.get_working_datasets()
        high_priority = registry.get_high_priority_datasets()
        medium_priority = registry.get_medium_priority_datasets()
        low_priority = registry.get_low_priority_datasets()
        
        print(f"✅ Working Datasets ({len(working)}):")
        for dataset in working:
            print(f"   - {dataset}")
        
        print(f"\n🔥 High Priority Pending ({len(high_priority)}):")
        for dataset in high_priority:
            if dataset not in working:
                print(f"   - {dataset}")
        
        print(f"\n🟡 Medium Priority Pending ({len(medium_priority)}):")
        for dataset in medium_priority:
            print(f"   - {dataset}")
        
        print(f"\n🟢 Low Priority Pending ({len(low_priority)}):")
        for dataset in low_priority:
            print(f"   - {dataset}")
    
    # Save test report
    report_data = {
        'timestamp': str(datetime.now()),
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'test_results': {name: {'success': success, 'data_summary': str(data)[:200]} 
                        for name, (success, data) in results.items()}
    }
    
    report_file = "test_results.json"
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)
    
    print(f"\n📊 Test report saved to: {report_file}")
    
    return passed_tests == total_tests


def main():
    """Run comprehensive integration tests."""
    print("UNIFIED DATASET INTEGRATION TEST SUITE")
    print("=" * 60)
    
    # Import datetime here to avoid issues
    from datetime import datetime
    
    # Test results storage
    results = {}
    
    # Test 1: Registry Initialization
    success, registry = test_registry_initialization()
    results["registry_init"] = (success, registry)
    
    if not success:
        print("❌ Cannot continue tests - registry initialization failed")
        return False
    
    # Test 2: Dataset Availability
    success, availability = test_dataset_availability(registry)
    results["dataset_availability"] = (success, availability)
    
    # Test 3: Individual Dataset Loading
    success, loading_results = test_individual_dataset_loading(registry, availability)
    results["dataset_loading"] = (success, loading_results)
    
    # Test 4: Evaluation Suite Configurations
    success, suite_configs = test_evaluation_suite_configurations()
    results["suite_configs"] = (success, suite_configs)
    
    # Test 5: Unified Evaluator Initialization
    success, evaluator = test_unified_evaluator_initialization()
    results["evaluator_init"] = (success, evaluator)
    
    if success and evaluator:
        # Test 6: Data Preparation
        success, prepared_data = test_evaluation_data_preparation(evaluator)
        results["data_preparation"] = (success, prepared_data)
    else:
        results["data_preparation"] = (False, {})
    
    # Generate comprehensive report
    all_passed = generate_test_report(results)
    
    if all_passed:
        print(f"\n🎉 ALL TESTS PASSED! Unified dataset integration is working correctly.")
    else:
        print(f"\n⚠️  Some tests failed. Check the details above for issues to resolve.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)