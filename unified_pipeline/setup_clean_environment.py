#!/usr/bin/env python3
"""
Clean Environment Setup Module

Sets up a clean environment for the unified pipeline to run without
warnings, compilation issues, or other verbose output that can interfere
with the evaluation process.
"""

import os
import sys
import warnings
import logging

def setup_clean_environment():
    """
    Set up clean environment variables and suppress verbose outputs.
    
    This function should be called at the very start of pipeline execution
    to ensure consistent, clean execution environment.
    """
    
    # Disable torch dynamo compilation
    os.environ['TORCH_DYNAMO_DISABLE'] = '1'
    os.environ['TORCH_COMPILE_DEBUG'] = '0'
    
    # Disable tokenizer parallelism to avoid warnings
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Suppress transformers verbose outputs
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    
    # Disable HuggingFace progress bars
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = 'true'
    
    # Suppress Python warnings
    os.environ['PYTHONWARNINGS'] = 'ignore'
    
    # Disable wandb if not explicitly enabled
    if 'WANDB_DISABLED' not in os.environ:
        os.environ['WANDB_DISABLED'] = 'true'
    
    # Set logging level to reduce verbosity
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('datasets').setLevel(logging.ERROR)
    
    # Suppress all warnings
    warnings.filterwarnings('ignore')
    
    # Suppress specific transformers warnings
    import transformers
    transformers.logging.set_verbosity_error()
    
    print("✓ Clean environment configured")