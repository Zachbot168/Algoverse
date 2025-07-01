#!/usr/bin/env python3
"""
Simple runner script for BERT bias testing.
Usage examples:
  python run_bert_bias_tests.py                    # Run all tests on all models
  python run_bert_bias_tests.py --quick            # Run StereoSet only
  python run_bert_bias_tests.py --crows            # Run CrowS-Pairs only
  python run_bert_bias_tests.py --models bert-base-uncased roberta-base  # Test specific models
"""

import sys
import os
import argparse

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_bert_bias import BERTBiasTester


def quick_test():
    """Run a quick bias test (StereoSet only)."""
    print("Running QUICK bias test (StereoSet only)...")
    
    tester = BERTBiasTester(
        persistent_dir="./quick_bias_results",
        data_dir="./data"
    )
    
    # Run only StereoSet for faster testing
    results = tester.run_full_evaluation(tests_to_run=["stereoset"])
    return results


def crows_test(models=None):
    """Run CrowS-Pairs testing only."""
    print("Running CrowS-Pairs bias testing...")
    
    tester = BERTBiasTester(
        persistent_dir="./crows_only_results", 
        data_dir="./data"
    )
    
    # Filter models if specified
    if models:
        original_models = tester.models_to_test.copy()
        tester.models_to_test = {
            k: v for k, v in original_models.items() 
            if k in models
        }
    
    # Run only CrowS-Pairs
    results = tester.run_full_evaluation(tests_to_run=["crows"])
    return results


def full_test(models=None):
    """Run full bias testing suite."""
    print("Running FULL bias testing suite...")
    
    tester = BERTBiasTester(
        persistent_dir="./full_bias_results", 
        data_dir="./data"
    )
    
    # Filter models if specified
    if models:
        original_models = tester.models_to_test.copy()
        tester.models_to_test = {
            k: v for k, v in original_models.items() 
            if k in models
        }
    
    # Run all tests
    results = tester.run_full_evaluation(tests_to_run=["stereoset", "crows", "seat"])
    return results


def seat_test(models=None):
    """Run SEAT testing only."""
    print("Running SEAT bias testing...")
    
    tester = BERTBiasTester(
        persistent_dir="./seat_only_results", 
        data_dir="./data"
    )
    
    # Filter models if specified
    if models:
        original_models = tester.models_to_test.copy()
        tester.models_to_test = {
            k: v for k, v in original_models.items() 
            if k in models
        }
    
    # Run only SEAT
    results = tester.run_full_evaluation(tests_to_run=["seat"])
    return results

def main():
    parser = argparse.ArgumentParser(description="BERT Bias Testing Runner")
    parser.add_argument(
        "--quick", 
        action="store_true",
        help="Run quick test (StereoSet only)"
    )
    parser.add_argument(
        "--crows", 
        action="store_true",
        help="Run CrowS-Pairs test only"
    )
    parser.add_argument(
        "--seat", 
        action="store_true",
        help="Run SEAT test only"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["bert-base-uncased", "roberta-base", "google/electra-small-discriminator", "bert-large-uncased"],
        help="Specific models to test"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        results = quick_test()
    elif args.crows:
        results = crows_test(models=args.models)
    elif args.seat:
        results = seat_test(models=args.models)
    else:
        results = full_test(models=args.models)
    
    print(f"\nTesting completed! Results for {len(results)} models.")


if __name__ == "__main__":
    main()