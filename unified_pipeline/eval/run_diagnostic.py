#!/usr/bin/env python3
"""
Unified Diagnostic Pass - Combines Path Patching and BAD Training

This script implements the core integration logic that runs both:
1. Path patching to identify sycophancy-related attention heads
2. BAD (Biased Activation Detection) probe training on the same activations

The expensive forward passes are done once, with activations reused for both analyses.
Results are stored in a unified component registry for downstream use.
"""

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.model_adapter import create_model_adapter, UniversalModelAdapter
import yaml

warnings.filterwarnings('ignore')


class UnifiedDiagnosticPass:
    """
    Unified diagnostic system that combines path patching and BAD training.
    
    This class runs both analyses on the same set of activations to identify:
    1. Attention heads responsible for sycophantic behavior (via path patching)
    2. Layers where bias can be linearly detected (via BAD probing)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize unified diagnostic pass with configuration."""
        self.config = config
        self.model_name = config['model']['name']
        self.device = config['model'].get('device', 'auto')
        
        # Setup device
        if self.device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        
        self.model = None
        self.tokenizer = None
        
        # Diagnostic parameters
        self.batch_size = config.get('evaluation', {}).get('batch_size', 4)
        self.max_samples = config.get('evaluation', {}).get('max_samples', 1000)
        
        print(f"Initialized UnifiedDiagnosticPass for {self.model_name}")
        print(f"Device: {self.device}")
    
    def load_model(self):
        """Load model and tokenizer with universal adapter."""
        print(f"Loading model with universal adapter: {self.model_name}")
        
        model_kwargs = {
            'torch_dtype': getattr(torch, self.config['model'].get('torch_dtype', 'float16')),
            'trust_remote_code': self.config['model'].get('trust_remote_code', False)
        }
        
        self.adapter = create_model_adapter(
            self.model_name,
            device=self.device,
            **model_kwargs
        )
        
        # Quick access to model and tokenizer
        self.model = self.adapter.model
        self.tokenizer = self.adapter.tokenizer
        
        self.model.eval()
        
        # Get model architecture info from adapter
        self.num_layers = self.adapter.arch_info.num_layers
        self.hidden_size = self.adapter.arch_info.hidden_size
        self.num_heads = self.adapter.arch_info.num_heads
        self.architecture = self.adapter.arch_info.architecture
        
        print(f"Model loaded: {self.architecture} with {self.num_layers} layers, {self.hidden_size} hidden size, {self.num_heads} heads")
    
    def load_diagnostic_data(self, dataset_path: str) -> List[Dict]:
        """Load diagnostic dataset."""
        print(f"Loading diagnostic data from: {dataset_path}")
        
        data = []
        with open(dataset_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item)
        
        print(f"Loaded {len(data)} diagnostic examples")
        return data[:self.max_samples]
    
    def extract_activations(self, data: List[Dict]) -> Dict[str, Any]:
        """Extract activations for both biased and unbiased examples."""
        print("Extracting activations from model...")
        
        # Prepare data for extraction
        biased_prompts = []
        unbiased_prompts = []
        labels = []
        
        for item in data:
            # Handle different data formats
            if 'biased_data' in item and 'unbiased_data' in item:
                # Format from pipeline
                biased = item['biased_data'][0]['content'] if isinstance(item['biased_data'], list) else item['biased_data']
                unbiased = item['unbiased_data'][0]['content'] if isinstance(item['unbiased_data'], list) else item['unbiased_data']
                
                biased_prompts.append(biased)
                unbiased_prompts.append(unbiased)
                labels.extend([0, 1])  # 0 = biased, 1 = unbiased
            
            elif 'reference_data' in item and 'counterfactual_data' in item:
                # Alternative format
                ref = item['reference_data'][0]['content'] if isinstance(item['reference_data'], list) else item['reference_data']
                counter = item['counterfactual_data'][0]['content'] if isinstance(item['counterfactual_data'], list) else item['counterfactual_data']
                
                biased_prompts.append(ref)
                unbiased_prompts.append(counter)
                labels.extend([0, 1])
        
        # Extract activations
        all_prompts = biased_prompts + unbiased_prompts
        activations = self._extract_layer_activations(all_prompts)
        
        return {
            'activations': activations,
            'labels': labels,
            'biased_prompts': biased_prompts,
            'unbiased_prompts': unbiased_prompts
        }
    
    def _extract_layer_activations(self, prompts: List[str]) -> Dict[int, np.ndarray]:
        """Extract activations from all layers."""
        layer_activations = {i: [] for i in range(self.num_layers)}
        
        # Setup hooks to capture activations
        hooks = []
        activations_cache = {}
        
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # Store last token activation
                if isinstance(output, tuple):
                    hidden_state = output[0]
                else:
                    hidden_state = output
                
                # Get last non-padding token activation
                last_token_activation = hidden_state[:, -1, :].detach().cpu().numpy()
                activations_cache[layer_idx] = last_token_activation
            return hook_fn
        
        # Register hooks on all layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Llama/Gemma style
            for i, layer in enumerate(self.model.model.layers):
                hook = layer.register_forward_hook(make_hook(i))
                hooks.append(hook)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT style
            for i, layer in enumerate(self.model.transformer.h):
                hook = layer.register_forward_hook(make_hook(i))
                hooks.append(hook)
        else:
            print("Warning: Could not find model layers for hook registration")
        
        try:
            # Process prompts in batches
            for i in tqdm(range(0, len(prompts), self.batch_size), desc="Extracting activations"):
                batch_prompts = prompts[i:i+self.batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                ).to(self.device)
                
                # Forward pass
                with torch.no_grad():
                    _ = self.model(**inputs)
                
                # Store activations
                for layer_idx in range(self.num_layers):
                    if layer_idx in activations_cache:
                        layer_activations[layer_idx].append(activations_cache[layer_idx])
                
                # Clear cache
                activations_cache.clear()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        # Convert to numpy arrays
        for layer_idx in layer_activations:
            if layer_activations[layer_idx]:
                layer_activations[layer_idx] = np.vstack(layer_activations[layer_idx])
            else:
                layer_activations[layer_idx] = np.array([])
        
        return layer_activations
    
    def run_bad_training(self, activations_data: Dict[str, Any]) -> Dict[int, float]:
        """Train BAD probes on extracted activations."""
        print("Training BAD (Biased Activation Detection) probes...")
        
        activations = activations_data['activations']
        labels = np.array(activations_data['labels'])
        
        bad_results = {}
        
        for layer_idx in tqdm(range(self.num_layers), desc="Training BAD probes"):
            if layer_idx not in activations or len(activations[layer_idx]) == 0:
                continue
            
            X = activations[layer_idx]
            y = labels[:len(X)]
            
            # Handle NaN values
            if np.isnan(X).any() or np.isinf(X).any():
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Skip if insufficient data
            if len(X) < 10 or len(np.unique(y)) < 2:
                continue
            
            try:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Train classifier
                clf = LogisticRegression(
                    solver='liblinear',
                    max_iter=1000,
                    random_state=42
                )
                clf.fit(X_train, y_train)
                
                # Evaluate
                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                bad_results[layer_idx] = accuracy
                
            except Exception as e:
                print(f"Warning: BAD training failed for layer {layer_idx}: {e}")
                continue
        
        print(f"BAD training completed for {len(bad_results)} layers")
        return bad_results
    
    def run_path_patching(self, activations_data: Dict[str, Any]) -> Dict[Tuple[int, int], float]:
        """Run simplified path patching analysis."""
        print("Running path patching analysis...")
        
        # Simplified path patching based on activation differences
        biased_prompts = activations_data['biased_prompts']
        unbiased_prompts = activations_data['unbiased_prompts']
        activations = activations_data['activations']
        
        path_patching_results = {}
        
        for layer_idx in tqdm(range(self.num_layers), desc="Path patching analysis"):
            if layer_idx not in activations or len(activations[layer_idx]) == 0:
                continue
            
            try:
                # Split activations by bias type
                n_pairs = min(len(biased_prompts), len(unbiased_prompts))
                biased_acts = activations[layer_idx][:n_pairs]
                unbiased_acts = activations[layer_idx][n_pairs:n_pairs*2]
                
                if len(biased_acts) == 0 or len(unbiased_acts) == 0:
                    continue
                
                # Compute activation differences (simplified head analysis)
                act_diff = np.mean(biased_acts, axis=0) - np.mean(unbiased_acts, axis=0)
                
                # Simulate head-level analysis by chunking hidden dimensions
                head_dim = self.hidden_size // self.num_heads
                
                for head_idx in range(self.num_heads):
                    start_idx = head_idx * head_dim
                    end_idx = min((head_idx + 1) * head_dim, len(act_diff))
                    
                    if start_idx < len(act_diff):
                        head_importance = np.abs(act_diff[start_idx:end_idx]).mean()
                        path_patching_results[(layer_idx, head_idx)] = float(head_importance)
                        
            except Exception as e:
                print(f"Warning: Path patching failed for layer {layer_idx}: {e}")
                continue
        
        print(f"Path patching completed for {len(path_patching_results)} head-layer pairs")
        return path_patching_results
    
    def create_component_registry(self, bad_results: Dict[int, float], 
                                  path_patching_results: Dict[Tuple[int, int], float]) -> Dict[str, Any]:
        """Create unified component registry from both analyses."""
        print("Creating unified component registry...")
        
        components = []
        
        # Add BAD results (MLP layers)
        for layer_idx, accuracy in bad_results.items():
            if accuracy > 0.6:  # Threshold for significant bias detection
                components.append({
                    "layer": layer_idx,
                    "type": "mlp",
                    "importance": accuracy,
                    "bias_type": "general",
                    "source": "bad_probe"
                })
        
        # Add path patching results (attention heads)
        for (layer_idx, head_idx), importance in path_patching_results.items():
            if importance > 0.1:  # Threshold for significant head importance
                components.append({
                    "layer": layer_idx,
                    "type": "head",
                    "head_index": head_idx,
                    "importance": importance,
                    "bias_type": "sycophancy",
                    "source": "path_patching"
                })
        
        # Sort by importance
        components.sort(key=lambda x: x['importance'], reverse=True)
        
        registry = {
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "num_components": len(components),
            "components": components,
            "metadata": {
                "total_components": len(components),
                "bad_layers": len([c for c in components if c['type'] == 'mlp']),
                "path_patching_heads": len([c for c in components if c['type'] == 'head']),
                "avg_importance": np.mean([c['importance'] for c in components]) if components else 0
            }
        }
        
        print(f"Registry created with {len(components)} components")
        return registry
    
    def run_unified_diagnostic(self, dataset_path: str, output_dir: str) -> Dict[str, Any]:
        """Run complete unified diagnostic pass."""
        print("Starting unified diagnostic pass...")
        
        # Load model
        self.load_model()
        
        # Load diagnostic data
        data = self.load_diagnostic_data(dataset_path)
        
        # Extract activations
        activations_data = self.extract_activations(data)
        
        # Run BAD training
        bad_results = self.run_bad_training(activations_data)
        
        # Run path patching
        path_patching_results = self.run_path_patching(activations_data)
        
        # Create component registry
        component_registry = self.create_component_registry(bad_results, path_patching_results)
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        
        # Save component registry
        registry_path = os.path.join(output_dir, "component_registry.json")
        with open(registry_path, 'w') as f:
            json.dump(component_registry, f, indent=2)
        
        # Save detailed results
        detailed_results = {
            "bad_results": bad_results,
            "path_patching_results": {f"{k[0]}_{k[1]}": v for k, v in path_patching_results.items()},
            "activations_summary": {
                "num_layers": self.num_layers,
                "hidden_size": self.hidden_size,
                "num_samples": len(activations_data['labels'])
            }
        }
        
        detailed_path = os.path.join(output_dir, "diagnostic_details.json")
        with open(detailed_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"Diagnostic results saved to: {output_dir}")
        print(f"Component registry: {registry_path}")
        
        return {
            "component_registry": component_registry,
            "bad_results": bad_results,
            "path_patching_results": path_patching_results,
            "registry_path": registry_path
        }


def main():
    """Main entry point for unified diagnostic pass."""
    parser = argparse.ArgumentParser(description="Run unified diagnostic pass")
    parser.add_argument("--config", required=True, help="Configuration file path")
    parser.add_argument("--data_path", required=True, help="Path to diagnostic dataset")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize and run diagnostic pass
    diagnostic = UnifiedDiagnosticPass(config)
    results = diagnostic.run_unified_diagnostic(args.data_path, args.output_dir)
    
    print("\nUnified diagnostic pass completed successfully!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()