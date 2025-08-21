#!/usr/bin/env python3
"""
Model Variant Loader for Unified Pipeline

Handles loading different model variants:
- baseline: Original model without modifications
- fairsteer: Model with FairSteer steering vectors applied
- sycophancy: Model with pinpoint tuning fine-tuning
- firm: Model with combined FIRM interventions
"""

import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent))
from steer.das_wrapper import DynamicActivationSteering
from train.component_registry import ComponentRegistryManager

warnings.filterwarnings('ignore')


class ModelVariantLoader:
    """
    Loads different variants of models with appropriate interventions applied.
    """
    
    def __init__(self, base_model: AutoModelForCausalLM, tokenizer: AutoTokenizer, config: Dict[str, Any]):
        """
        Initialize the model variant loader.
        
        Args:
            base_model: The base model to modify
            tokenizer: The tokenizer for the model
            config: Configuration dictionary containing variant settings
        """
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.config = config
        self.model_name = config.get('model_name', '')
        self.model_variant = config.get('model_variant', 'baseline')
    
    def load_variant_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load the specified model variant.
        
        Returns:
            Tuple of (modified_model, tokenizer)
        """
        if self.model_variant == 'baseline':
            return self.base_model, self.tokenizer
        elif self.model_variant == 'fairsteer':
            return self._load_fairsteer_variant()
        elif self.model_variant == 'sycophancy':
            return self._load_sycophancy_variant()
        elif self.model_variant == 'firm':
            return self._load_firm_variant()
        else:
            raise ValueError(f"Unknown model variant: {self.model_variant}")
    
    def _load_fairsteer_variant(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load FairSteer variant with steering vectors."""
        print("   🎯 Applying FairSteer steering vectors...")
        
        # Look for existing steering vectors
        fairsteer_path = self._find_fairsteer_path()
        if fairsteer_path is None:
            print("   ⚠️  No FairSteer vectors found, using baseline model")
            return self.base_model, self.tokenizer
        
        try:
            with open(fairsteer_path, 'rb') as f:
                fairsteer_data = pickle.load(f)
            
            steering_vectors = fairsteer_data.get('steering_vectors', {})
            optimal_layer = fairsteer_data.get('optimal_layer', 15)
            
            if not steering_vectors:
                print("   ⚠️  Empty steering vectors, using baseline model")
                return self.base_model, self.tokenizer
            
            # Apply steering vectors using DAS wrapper
            das_model = DynamicActivationSteering(
                model=self.base_model,
                steering_vectors=steering_vectors,
                optimal_layer=optimal_layer,
                intervention_strength=self.config.get('fairsteer', {}).get('intervention_strength', 1.0)
            )
            
            print(f"   ✓ FairSteer applied at layer {optimal_layer}")
            return das_model, self.tokenizer
            
        except Exception as e:
            print(f"   ⚠️  Failed to load FairSteer variant: {e}")
            return self.base_model, self.tokenizer
    
    def _load_sycophancy_variant(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load sycophancy variant with pinpoint tuning."""
        print("   🎯 Applying sycophancy pinpoint tuning...")
        
        # Look for sycophancy fine-tuned model
        sycophancy_path = self._find_sycophancy_path()
        if sycophancy_path is None:
            print("   ⚠️  No sycophancy model found, using baseline")
            return self.base_model, self.tokenizer
        
        try:
            # Load the fine-tuned model
            sycophancy_model = AutoModelForCausalLM.from_pretrained(
                sycophancy_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                attn_implementation="eager"
            )
            
            print(f"   ✓ Sycophancy model loaded from {sycophancy_path}")
            return sycophancy_model, self.tokenizer
            
        except Exception as e:
            print(f"   ⚠️  Failed to load sycophancy variant: {e}")
            return self.base_model, self.tokenizer
    
    def _load_firm_variant(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load FIRM variant with combined interventions."""
        print("   🎯 Applying FIRM multi-component interventions...")
        
        # FIRM combines multiple techniques
        model = self.base_model
        
        try:
            # 1. Apply FairSteer if available
            fairsteer_path = self._find_fairsteer_path()
            if fairsteer_path:
                print("   📍 Applying FairSteer component...")
                with open(fairsteer_path, 'rb') as f:
                    fairsteer_data = pickle.load(f)
                
                steering_vectors = fairsteer_data.get('steering_vectors', {})
                optimal_layer = fairsteer_data.get('optimal_layer', 15)
                
                if steering_vectors:
                    model = DynamicActivationSteering(
                        model=model,
                        steering_vectors=steering_vectors,
                        optimal_layer=optimal_layer,
                        intervention_strength=self.config.get('firm', {}).get('fairsteer_strength', 0.8)
                    )
            
            # 2. Apply additional FIRM components if available
            # (This would include other FIRM techniques when implemented)
            
            print("   ✓ FIRM interventions applied")
            return model, self.tokenizer
            
        except Exception as e:
            print(f"   ⚠️  Failed to load FIRM variant: {e}")
            return self.base_model, self.tokenizer
    
    def _find_fairsteer_path(self) -> Optional[str]:
        """Find FairSteer steering vectors file."""
        # Check multiple possible locations
        possible_paths = [
            f"fairsteer_{self.model_name.split('/')[-1].lower()}.pkl",
            f"fairsteer_{self.model_name.split('/')[-1].lower().replace('-', '_')}.pkl",
            "fairsteer_gemma2b.pkl",  # Legacy path
            f"steering_vectors/{self.model_name.split('/')[-1].lower()}.pkl",
            f"steer/steering_vectors_{self.model_name.split('/')[-1].lower()}.pkl"
        ]
        
        # Check current directory and parent directories
        for base_path in [Path.cwd(), Path.cwd().parent, Path(__file__).parent.parent]:
            for possible_path in possible_paths:
                full_path = base_path / possible_path
                if full_path.exists():
                    return str(full_path)
        
        return None
    
    def _find_sycophancy_path(self) -> Optional[str]:
        """Find sycophancy fine-tuned model directory."""
        # Check for sycophancy pipeline runs
        possible_dirs = [
            "sycophancy_pipeline_runs",
            "unified_pipeline/sycophancy_pipeline_runs",
            f"sycophancy_pipeline_runs/sycophancy_{self.model_name.split('/')[-1].lower()}_*/training",
            f"pipeline_runs/*/training"  # Legacy structure
        ]
        
        for base_path in [Path.cwd(), Path.cwd().parent, Path(__file__).parent.parent]:
            for possible_dir in possible_dirs:
                full_path = base_path / possible_dir
                if full_path.exists():
                    # Find most recent training directory
                    if "*" in str(possible_dir):
                        matching_dirs = list(base_path.glob(possible_dir))
                        if matching_dirs:
                            # Sort by modification time, get most recent
                            latest_dir = max(matching_dirs, key=lambda x: x.stat().st_mtime)
                            return str(latest_dir)
                    else:
                        return str(full_path)
        
        return None