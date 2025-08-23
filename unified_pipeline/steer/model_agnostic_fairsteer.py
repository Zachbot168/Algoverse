#!/usr/bin/env python3
"""
Model-Agnostic FairSteer Implementation

Creates steering vectors for any supported model architecture:
- Qwen 2.5 (1.5B, 3B)
- Llama 3.2 (1B, 3B)
- Gemma 2 (2B)
- Ministral (3B)

Based on the original FairSteer methodology but made architecture-independent.
"""

import os
import sys
import torch
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import warnings
from tqdm import tqdm

# Add unified pipeline to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.pytorch_compilation_fix import apply_pytorch_compilation_fixes, disable_model_compilation

warnings.filterwarnings('ignore')
apply_pytorch_compilation_fixes()

@dataclass
class ModelConfig:
    """Configuration for different model architectures."""
    model_name: str
    architecture: str
    num_layers: int
    num_heads: int
    hidden_size: int
    optimal_layer_range: Tuple[int, int]
    layer_name_pattern: str  # How layers are named in the model
    
    def get_optimal_layers(self) -> List[int]:
        """Get the optimal layers for steering vector computation."""
        start, end = self.optimal_layer_range
        return list(range(start, min(end + 1, self.num_layers)))

# Model architecture configurations
MODEL_CONFIGS = {
    "Qwen/Qwen2.5-3B-Instruct": ModelConfig(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        architecture="qwen2",
        num_layers=36,
        num_heads=16,
        hidden_size=2048,
        optimal_layer_range=(24, 30),  # Upper-middle layers
        layer_name_pattern="model.layers.{}.self_attn"
    ),
    "Qwen/Qwen2.5-1.5B-Instruct": ModelConfig(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        architecture="qwen2",
        num_layers=28,
        num_heads=12,
        hidden_size=1536,
        optimal_layer_range=(18, 24),
        layer_name_pattern="model.layers.{}.self_attn"
    ),
    "meta-llama/Llama-3.2-3B-Instruct": ModelConfig(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        architecture="llama",
        num_layers=28,
        num_heads=24,
        hidden_size=3072,
        optimal_layer_range=(18, 24),
        layer_name_pattern="model.layers.{}.self_attn"
    ),
    "meta-llama/Llama-3.2-1B-Instruct": ModelConfig(
        model_name="meta-llama/Llama-3.2-1B-Instruct", 
        architecture="llama",
        num_layers=16,
        num_heads=32,
        hidden_size=2048,
        optimal_layer_range=(10, 14),
        layer_name_pattern="model.layers.{}.self_attn"
    ),
    "google/gemma-2-2b-it": ModelConfig(
        model_name="google/gemma-2-2b-it",
        architecture="gemma",
        num_layers=26,
        num_heads=8,
        hidden_size=2304,
        optimal_layer_range=(16, 22),
        layer_name_pattern="model.layers.{}.self_attn"
    ),
    "ministral/Ministral-3b-instruct": ModelConfig(
        model_name="ministral/Ministral-3b-instruct",
        architecture="ministral",
        num_layers=14,
        num_heads=32,
        hidden_size=3072,
        optimal_layer_range=(9, 12),
        layer_name_pattern="model.layers.{}.self_attn"
    )
}

class ModelAgnosticFairSteer:
    """
    Model-agnostic FairSteer implementation that works across different architectures.
    """
    
    def __init__(self, model_name: str, model=None, tokenizer=None, device: str = "auto"):
        """
        Initialize model-agnostic FairSteer.
        
        Args:
            model_name: HuggingFace model name
            model: Pre-loaded model (optional)
            tokenizer: Pre-loaded tokenizer (optional)
            device: Device to use
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        
        # Get model configuration
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Model {model_name} not supported. Supported models: {list(MODEL_CONFIGS.keys())}")
        
        self.config = MODEL_CONFIGS[model_name]
        
        # Load model and tokenizer if not provided
        if model is None or tokenizer is None:
            self.model, self.tokenizer = self._load_model()
        else:
            self.model, self.tokenizer = model, tokenizer
            
        # Ensure model is on correct device and compilation is disabled
        self.model = disable_model_compilation(self.model)
        
        # FairSteer state
        self.steering_vectors = {}
        self.bias_classifiers = {}
        self.optimal_layer = None
        
        print(f"✅ Model-agnostic FairSteer initialized for {model_name}")
        print(f"   Architecture: {self.config.architecture}")
        print(f"   Layers: {self.config.num_layers}, Heads: {self.config.num_heads}")
        print(f"   Optimal layer range: {self.config.optimal_layer_range}")
    
    def _setup_device(self, device: str) -> str:
        """Setup device for computation."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        return device
    
    def _load_model(self):
        """Load model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading {self.model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            attn_implementation="eager"
        )
        
        return model, tokenizer
    
    def compute_steering_vectors(self, bias_pairs: List[Tuple[str, str]], 
                                num_layers_to_compute: int = 5) -> Dict[int, torch.Tensor]:
        """
        Compute steering vectors for bias mitigation.
        
        Args:
            bias_pairs: List of (biased_text, neutral_text) pairs
            num_layers_to_compute: Number of layers to compute vectors for
            
        Returns:
            Dictionary mapping layer indices to steering vectors
        """
        print(f"Computing steering vectors for {self.model_name}...")
        print(f"Using {len(bias_pairs)} bias pairs")
        
        # Get optimal layers to compute
        optimal_layers = self.config.get_optimal_layers()
        layers_to_use = optimal_layers[:num_layers_to_compute]
        
        print(f"Computing for layers: {layers_to_use}")
        
        steering_vectors = {}
        
        for layer_idx in tqdm(layers_to_use, desc="Computing steering vectors"):
            # Get activations for biased and neutral texts
            biased_activations = []
            neutral_activations = []
            
            for biased_text, neutral_text in bias_pairs[:50]:  # Use subset for speed
                # Get biased activations
                biased_acts = self._get_layer_activations(biased_text, layer_idx)
                if biased_acts is not None:
                    biased_activations.append(biased_acts.cpu())
                
                # Get neutral activations
                neutral_acts = self._get_layer_activations(neutral_text, layer_idx)
                if neutral_acts is not None:
                    neutral_activations.append(neutral_acts.cpu())
            
            if biased_activations and neutral_activations:
                # Compute steering vector as difference of means
                biased_mean = torch.stack(biased_activations).mean(dim=0)
                neutral_mean = torch.stack(neutral_activations).mean(dim=0)
                
                steering_vector = biased_mean - neutral_mean
                steering_vectors[layer_idx] = steering_vector
                
                print(f"✓ Layer {layer_idx}: steering vector shape {steering_vector.shape}")
            else:
                print(f"⚠ Layer {layer_idx}: insufficient activations")
        
        # Find optimal layer (highest magnitude steering vector)
        if steering_vectors:
            self.optimal_layer = max(steering_vectors.keys(), 
                                   key=lambda k: torch.norm(steering_vectors[k]).item())
            print(f"🎯 Optimal layer identified: {self.optimal_layer}")
        
        self.steering_vectors = steering_vectors
        return steering_vectors
    
    def _get_layer_activations(self, text: str, layer_idx: int) -> Optional[torch.Tensor]:
        """Get activations from a specific layer for given text."""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Hook to capture activations
            activations = {}
            
            def hook_fn(module, input, output):
                # Get the hidden states (usually first element of output)
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                # Average pool over sequence length
                pooled = hidden_states.mean(dim=1)  # [batch, hidden_size]
                activations['hidden'] = pooled.detach()
            
            # Register hook on the appropriate layer
            layer_module = self._get_layer_module(layer_idx)
            if layer_module is not None:
                handle = layer_module.register_forward_hook(hook_fn)
                
                # Forward pass
                with torch.no_grad():
                    _ = self.model(**inputs)
                
                handle.remove()
                
                return activations.get('hidden')
            
        except Exception as e:
            print(f"Error getting activations for layer {layer_idx}: {e}")
            return None
    
    def _get_layer_module(self, layer_idx: int):
        """Get the module for a specific layer."""
        try:
            # Navigate to the layer based on architecture
            if self.config.architecture in ["qwen2", "llama", "mistral", "ministral"]:
                return self.model.model.layers[layer_idx]
            elif self.config.architecture == "gemma":
                return self.model.model.layers[layer_idx]
            else:
                print(f"Unknown architecture: {self.config.architecture}")
                return None
        except Exception as e:
            print(f"Error accessing layer {layer_idx}: {e}")
            return None
    
    def save_steering_vectors(self, output_path: str):
        """Save steering vectors to file."""
        output_data = {
            'model_name': self.model_name,
            'architecture': self.config.architecture,
            'steering_vectors': {k: v.cpu().numpy() for k, v in self.steering_vectors.items()},
            'optimal_layer': self.optimal_layer,
            'model_config': {
                'num_layers': self.config.num_layers,
                'num_heads': self.config.num_heads,
                'hidden_size': self.config.hidden_size,
                'optimal_layer_range': self.config.optimal_layer_range
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f)
        
        print(f"✅ Steering vectors saved to: {output_path}")
        
        # Also save metadata as JSON for inspection
        metadata_path = output_path.replace('.pkl', '_metadata.json')
        metadata = {k: v for k, v in output_data.items() if k != 'steering_vectors'}
        metadata['steering_vector_layers'] = list(self.steering_vectors.keys())
        metadata['steering_vector_shapes'] = {
            k: list(v.shape) for k, v in self.steering_vectors.items()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Metadata saved to: {metadata_path}")
        
        return output_path

def generate_bias_pairs() -> List[Tuple[str, str]]:
    """Generate bias pairs for steering vector computation."""
    bias_pairs = [
        # Gender bias
        ("She works as a nurse", "They work as a nurse"),
        ("He is an engineer", "They are an engineer"),
        ("The woman is caring", "The person is caring"),
        ("The man is strong", "The person is strong"),
        ("Girls are good at reading", "Students are good at reading"),
        ("Boys are good at math", "Students are good at math"),
        
        # Racial bias
        ("The Black person was", "The person was"),
        ("Asian students are smart", "Students are smart"),
        ("White families live in", "Families live in"),
        
        # Religious bias  
        ("Muslims are", "People are"),
        ("Christians believe", "People believe"),
        ("Jewish families", "Families"),
        
        # Professional bias
        ("The female CEO", "The CEO"),
        ("Male doctors", "Doctors"),
        ("Women teachers", "Teachers"),
        ("Men in construction", "People in construction"),
    ]
    
    return bias_pairs

def create_steering_vectors_for_all_models(output_dir: str = "steering_vectors"):
    """Create steering vectors for all supported models."""
    os.makedirs(output_dir, exist_ok=True)
    
    bias_pairs = generate_bias_pairs()
    results = {}
    
    for model_name in MODEL_CONFIGS.keys():
        print(f"\n{'='*60}")
        print(f"Creating FairSteer vectors for: {model_name}")
        print(f"{'='*60}")
        
        try:
            # Initialize FairSteer for this model
            fairsteer = ModelAgnosticFairSteer(model_name)
            
            # Compute steering vectors
            steering_vectors = fairsteer.compute_steering_vectors(bias_pairs)
            
            if steering_vectors:
                # Save steering vectors
                safe_model_name = model_name.replace('/', '_').replace('-', '_').lower()
                output_path = f"{output_dir}/fairsteer_{safe_model_name}.pkl"
                fairsteer.save_steering_vectors(output_path)
                
                results[model_name] = {
                    'success': True,
                    'output_path': output_path,
                    'optimal_layer': fairsteer.optimal_layer,
                    'num_vectors': len(steering_vectors)
                }
                
                print(f"✅ Successfully created FairSteer vectors for {model_name}")
            else:
                results[model_name] = {
                    'success': False,
                    'error': 'No steering vectors computed'
                }
                print(f"❌ Failed to create steering vectors for {model_name}")
                
        except Exception as e:
            results[model_name] = {
                'success': False,
                'error': str(e)
            }
            print(f"❌ Error with {model_name}: {e}")
    
    # Save overall results
    with open(f"{output_dir}/creation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ FairSteer vector creation complete!")
    print(f"Results saved to: {output_dir}/creation_results.json")
    
    return results


if __name__ == "__main__":
    # Create steering vectors for all models
    results = create_steering_vectors_for_all_models()
    
    # Print summary
    print(f"\n{'='*60}")
    print("FAIRSTEER CREATION SUMMARY")
    print(f"{'='*60}")
    
    successful = [k for k, v in results.items() if v['success']]
    failed = [k for k, v in results.items() if not v['success']]
    
    print(f"✅ Successful: {len(successful)}/{len(results)}")
    for model in successful:
        print(f"   - {model}")
    
    if failed:
        print(f"❌ Failed: {len(failed)}")
        for model in failed:
            print(f"   - {model}: {results[model]['error']}")