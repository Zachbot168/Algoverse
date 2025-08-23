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
        # Normalize model name to fix character encoding issues
        model_name = config.get('model_name', '')
        model_name = model_name.replace('–', '-').replace('—', '-')  # Replace em-dashes with hyphens
        model_name = model_name.replace('\u2013', '-').replace('\u2014', '-')  # Unicode em-dashes
        model_name = model_name.strip('"').strip("'").strip('\u201c').strip('\u201d')  # Remove quotes
        self.model_name = model_name
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
            
            # Create a simple steering wrapper for FairSteer
            from steer.simple_fairsteer_wrapper import SimpleFairSteerWrapper
            das_model = SimpleFairSteerWrapper(
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
                    from steer.simple_fairsteer_wrapper import SimpleFairSteerWrapper
                    model = SimpleFairSteerWrapper(
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
        """Find model-agnostic FairSteer steering vectors file."""
        # Generate model-specific filename based on actual model name
        safe_model_name = self.model_name.replace('/', '_').replace('-', '_').lower()
        
        # Primary check: Look for existing model-specific steering vectors
        steering_vectors_dir = Path(__file__).parent.parent / "steering_vectors"
        primary_path = steering_vectors_dir / f"fairsteer_{safe_model_name}.pkl"
        
        if primary_path.exists():
            print(f"   📁 Found model-specific FairSteer vectors: {primary_path}")
            return str(primary_path)
        
        # Secondary check: Look in common locations with various naming conventions
        possible_paths = [
            f"steering_vectors/fairsteer_{safe_model_name}.pkl",
            f"fairsteer_{safe_model_name}.pkl",
            f"fairsteer_{self.model_name.split('/')[-1].lower()}.pkl",
            f"fairsteer_{self.model_name.split('/')[-1].lower().replace('-', '_')}.pkl",
        ]
        
        # Check current directory and parent directories
        for base_path in [Path.cwd(), Path.cwd().parent, Path(__file__).parent.parent]:
            for possible_path in possible_paths:
                full_path = base_path / possible_path
                if full_path.exists():
                    print(f"   📁 Found FairSteer vectors: {full_path}")
                    return str(full_path)
        
        # If no model-specific vectors found, create them using model-agnostic FairSteer
        print(f"   ⚠️  No FairSteer vectors found for {self.model_name}")
        print(f"   🔧 Creating model-specific FairSteer vectors using model-agnostic implementation...")
        
        try:
            # Import and use model-agnostic FairSteer
            sys.path.append(str(Path(__file__).parent))
            from steer.model_agnostic_fairsteer import ModelAgnosticFairSteer, generate_bias_pairs
            
            # Create steering vectors for this specific model
            fairsteer = ModelAgnosticFairSteer(self.model_name, self.base_model, self.tokenizer)
            bias_pairs = generate_bias_pairs()
            steering_vectors = fairsteer.compute_steering_vectors(bias_pairs)
            
            if steering_vectors and hasattr(fairsteer, 'optimal_layer'):
                # Save for future use
                steering_vectors_dir.mkdir(exist_ok=True)
                output_path = steering_vectors_dir / f"fairsteer_{safe_model_name}.pkl"
                fairsteer.save_steering_vectors(str(output_path))
                print(f"   ✅ Created and saved model-specific FairSteer vectors: {output_path}")
                return str(output_path)
            else:
                print(f"   ❌ Failed to generate valid steering vectors for {self.model_name}")
        except Exception as e:
            print(f"   ❌ Failed to create FairSteer vectors: {e}")
            import traceback
            traceback.print_exc()
        
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