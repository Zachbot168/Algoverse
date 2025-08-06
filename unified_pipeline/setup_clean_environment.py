#!/usr/bin/env python3
"""
Clean Environment Setup for Bias Evaluation Pipeline

This script configures the environment to suppress PyTorch compilation warnings
and other verbose outputs that can clutter the pipeline execution logs.
"""

import os
import warnings
import logging
import torch
import sys


def setup_clean_environment():
    """Configure environment for clean pipeline execution."""
    
    print("🧹 Setting up clean environment for bias evaluation pipeline...")
    
    # Suppress all Python warnings
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning) 
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    # Configure logging to reduce verbosity
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('torch').setLevel(logging.ERROR)
    logging.getLogger('datasets').setLevel(logging.ERROR)
    logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
    logging.getLogger().setLevel(logging.WARNING)
    
    # Set environment variables to disable verbose outputs
    env_vars = {
        # PyTorch compilation optimizations (major source of warnings)
        'TORCH_DYNAMO_DISABLE': '1',
        'TORCH_COMPILE_DEBUG': '0',
        
        # Tokenizers parallelism warnings
        'TOKENIZERS_PARALLELISM': 'false',
        
        # Transformers library verbosity
        'TRANSFORMERS_VERBOSITY': 'error',
        'TRANSFORMERS_NO_ADVISORY_WARNINGS': 'true',
        
        # Hugging Face Hub progress bars  
        'HF_HUB_DISABLE_PROGRESS_BARS': 'true',
        
        # Python warnings
        'PYTHONWARNINGS': 'ignore',
        
        # CUDA warnings
        'CUDA_LAUNCH_BLOCKING': '0',
        
        # Additional suppression
        'TF_CPP_MIN_LOG_LEVEL': '3',  # TensorFlow if present
        'OMP_NUM_THREADS': '1',  # Reduce threading warnings
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value
        
    # Apply torch-specific configurations
    if torch.cuda.is_available():
        # Disable CUDA warnings about memory optimization
        torch.cuda.empty_cache()
        
    print("✅ Clean environment configured:")
    print("   - PyTorch compilation warnings suppressed")
    print("   - Transformers verbosity reduced to errors only")
    print("   - Progress bars and tokenizer warnings disabled")
    print("   - All Python warnings filtered")
    
    return True


def verify_clean_setup():
    """Verify that warning suppression is working."""
    print("\n🔍 Verifying clean environment setup...")
    
    # Test warning suppression
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.warn("Test warning", UserWarning)
        if len(w) == 0:
            print("✅ Warning suppression working correctly")
        else:
            print("⚠️  Some warnings may still be visible")
    
    # Check environment variables
    critical_vars = ['TORCH_DYNAMO_DISABLE', 'TRANSFORMERS_VERBOSITY', 'TOKENIZERS_PARALLELISM']
    all_set = True
    for var in critical_vars:
        if var not in os.environ:
            print(f"⚠️  {var} not set")
            all_set = False
    
    if all_set:
        print("✅ All critical environment variables configured")
    
    print("\n🚀 Environment ready for clean pipeline execution!")
    return True


def main():
    """Main function to set up clean environment."""
    try:
        setup_clean_environment()
        verify_clean_setup()
        
        print("\n" + "="*60)
        print("🎯 CLEAN ENVIRONMENT SETUP COMPLETE")
        print("="*60)
        print("You can now run the bias evaluation pipeline with minimal warnings:")
        print("python run_unified_pipeline.py --model-config configs/gemma_2b.yaml")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Failed to set up clean environment: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()