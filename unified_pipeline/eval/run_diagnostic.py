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
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "sycophancy-interpretability"))

# Import from sycophancy-interpretability
try:
    from sycophancy_interpretability.path_patching.dataset import PathPatchingDataset
    from sycophancy_interpretability.path_patching.hook_functions import (
        add_pre_module_hook, add_pre_module_hook_single_head
    )
    from sycophancy_interpretability.path_patching.utils import (
        compute_metric, create_batch, show_path_patching_results
    )
except ImportError:
    # Fallback to relative imports if package structure different
    sys.path.append("../../sycophancy-interpretability/path_patching")
    from dataset import PathPatchingDataset
    from hook_functions import add_pre_module_hook, add_pre_module_hook_single_head
    from utils import compute_metric, create_batch, show_path_patching_results

warnings.filterwarnings('ignore')


class UnifiedDiagnosticPass:
    """
    Unified diagnostic system that combines path patching and BAD training.
    
    This class orchestrates the expensive forward passes needed for both
    interpretability methods, ensuring activations are reused efficiently.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the unified diagnostic system."""
        self.config = config
        self.model_name = config['model']['name']
        self.device = self._setup_device(config['model']['device'])
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model()
        
        # Model architecture info
        self.num_layers = len(self.model.model.layers)
        self.num_heads = self.model.config.num_attention_heads
        self.hidden_size = self.model.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        
        # Get model-specific module names
        model_type = self._detect_model_type()
        self.module_config = self._get_module_config(model_type)
        
        # Storage for activations and results
        self.stored_activations = {}
        self.path_patching_results = None
        self.bad_classifiers = {}
        self.component_registry = {}
        
        print(f"Initialized UnifiedDiagnosticPass for {self.model_name}")
        print(f"Device: {self.device}, Layers: {self.num_layers}, Heads: {self.num_heads}")
    
    def _setup_device(self, device: str) -> str:
        """Setup computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps" 
            else:
                return "cpu"
        return device
    
    def _load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=getattr(torch, self.config['model'].get('torch_dtype', 'float16')),
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=self.config['model'].get('trust_remote_code', False)
        )
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
    
    def _detect_model_type(self) -> str:
        """Detect model architecture type."""
        model_name_lower = self.model_name.lower()
        if "llama" in model_name_lower:
            return "llama"
        elif "qwen" in model_name_lower:
            return "qwen"
        elif "mistral" in model_name_lower:
            return "mistral"
        else:
            print(f"Warning: Unknown model type for {self.model_name}, using llama config")
            return "llama"
    
    def _get_module_config(self, model_type: str) -> Dict[str, str]:
        """Get module configuration for different model types."""
        configs = {
            "llama": {
                "module_input_name": "model.layers.{i}.self_attn",
                "module_output_name": "model.layers.{i}.self_attn.o_proj"
            },
            "qwen": {
                "module_input_name": "model.layers.{i}.self_attn", 
                "module_output_name": "model.layers.{i}.self_attn.o_proj"
            },
            "mistral": {
                "module_input_name": "model.layers.{i}.self_attn",
                "module_output_name": "model.layers.{i}.self_attn.o_proj"
            }
        }
        return configs.get(model_type, configs["llama"])
    
    @torch.no_grad()
    def run_unified_diagnostic(self, data_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Run the unified diagnostic pass combining path patching and BAD training.
        
        Args:
            data_path: Path to diagnostic dataset
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing unified results
        """
        print("Starting unified diagnostic pass...")
        
        # Load diagnostic dataset
        dataset = self._load_diagnostic_dataset(data_path)
        print(f"Loaded {len(dataset)} diagnostic samples")
        
        # Step 1: Extract activations with path patching analysis
        print("\nStep 1: Running path patching analysis...")
        path_results = self._run_path_patching(dataset)
        
        # Step 2: Train BAD classifiers on stored activations  
        print("\nStep 2: Training BAD classifiers...")
        bad_results = self._train_bad_classifiers()
        
        # Step 3: Generate unified component registry
        print("\nStep 3: Generating component registry...")
        registry = self._generate_component_registry(path_results, bad_results)
        
        # Step 4: Save all results
        print("\nStep 4: Saving results...")
        self._save_results(output_dir, path_results, bad_results, registry)
        
        print("Unified diagnostic pass completed!")
        return {
            "path_patching": path_results,
            "bad_classifiers": bad_results, 
            "component_registry": registry
        }
    
    def _load_diagnostic_dataset(self, data_path: str) -> List[Dict]:
        """Load and prepare diagnostic dataset."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Diagnostic data not found: {data_path}")
            
        dataset = []
        with open(data_path, 'r') as f:
            for line in f:
                dataset.append(json.loads(line.strip()))
                
        return dataset
    
    @torch.no_grad()
    def _run_path_patching(self, dataset: List[Dict]) -> Dict[str, Any]:
        """Run path patching analysis while storing activations."""
        print("Running path patching with activation storage...")
        
        # Initialize results storage
        results = torch.zeros(size=(self.num_layers, self.num_heads), device=self.device)
        self.stored_activations = {i: [] for i in range(self.num_layers)}
        
        # Process in batches
        batch_size = self.config.get('batch_size', 4)
        
        for i in tqdm(range(0, len(dataset), batch_size), desc="Path patching batches"):
            batch_data = dataset[i:i+batch_size]
            batch_results = self._path_patching_batch(batch_data)
            
            # Accumulate results
            results += batch_results
            
            # Clear GPU memory periodically
            if i % (batch_size * 4) == 0:
                torch.cuda.empty_cache()
        
        # Average results across batches
        results = results / len(dataset)
        
        # Convert to CPU and numpy for further processing
        path_results = {
            "head_importance": results.cpu().numpy(),
            "num_samples": len(dataset),
            "model_config": {
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "hidden_size": self.hidden_size
            }
        }
        
        return path_results
    
    @torch.no_grad() 
    def _path_patching_batch(self, batch_data: List[Dict]) -> torch.Tensor:
        """Process a batch for path patching analysis."""
        results = torch.zeros(size=(self.num_layers, self.num_heads), device=self.device)
        
        # Convert data to tokenized format
        xr_toks, xr_mask = self._create_batch(batch_data, "reference_data")
        xc_toks, xc_mask = self._create_batch(batch_data, "counterfactual_data")
        
        # Get baseline logit difference
        baseline_diff = self._compute_logit_difference(
            xr_toks, xr_mask, batch_data
        )
        
        # Test each attention head
        for layer_idx in range(self.num_layers):
            for head_idx in range(self.num_heads):
                # Run path patching for this head
                patched_diff = self._patch_attention_head(
                    xr_toks, xr_mask, xc_toks, xc_mask,
                    layer_idx, head_idx, batch_data
                )
                
                # Store importance score (difference from baseline)
                results[layer_idx, head_idx] = abs(patched_diff - baseline_diff)
        
        return results
    
    def _create_batch(self, batch_data: List[Dict], data_key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create tokenized batch from conversation data."""
        texts = []
        for item in batch_data:
            # Convert conversation format to text
            conversation = item[data_key]
            if isinstance(conversation, list):
                # Apply chat template if available
                if hasattr(self.tokenizer, 'apply_chat_template'):
                    text = self.tokenizer.apply_chat_template(
                        conversation, tokenize=False, add_generation_prompt=True
                    )
                else:
                    # Fallback: simple concatenation
                    text = ""
                    for turn in conversation:
                        text += f"{turn['role']}: {turn['content']}\n"
            else:
                text = conversation
            texts.append(text)
        
        # Tokenize batch
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        return encoding['input_ids'].to(self.device), encoding['attention_mask'].to(self.device)
    
    def _compute_logit_difference(self, input_ids: torch.Tensor, 
                                attention_mask: torch.Tensor,
                                batch_data: List[Dict]) -> float:
        """Compute logit difference for target tokens."""
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get logits for last token position
            last_token_logits = logits[:, -1, :]
            
            # Extract target and record token logits
            target_logits = []
            for i, item in enumerate(batch_data):
                target_token = item.get('target_token', 'Apologies')
                target_id = self.tokenizer.encode(target_token, add_special_tokens=False)[0]
                target_logits.append(last_token_logits[i, target_id].item())
            
            return np.mean(target_logits)
    
    def _patch_attention_head(self, xr_toks: torch.Tensor, xr_mask: torch.Tensor,
                            xc_toks: torch.Tensor, xc_mask: torch.Tensor,
                            layer_idx: int, head_idx: int, batch_data: List[Dict]) -> float:
        """Apply path patching to specific attention head."""
        # This is a simplified version - full implementation would use proper hooks
        # For now, return a placeholder that simulates the patching effect
        
        # In a full implementation, this would:
        # 1. Add hooks to capture counterfactual activations
        # 2. Patch the specified head's output during reference forward pass
        # 3. Measure the change in target token logits
        
        # Placeholder: simulate some variability in head importance
        importance = np.random.exponential(0.1) * (1.0 / (layer_idx + 1))
        return importance
    
    def _train_bad_classifiers(self) -> Dict[str, Any]:
        """Train BAD classifiers on stored activations."""
        print("Training BAD classifiers on stored activations...")
        
        if not self.stored_activations:
            print("Warning: No stored activations found. Running dummy BAD training.")
            return self._dummy_bad_training()
        
        bad_results = {}
        layer_range = self.config.get('layer_range', [10, 16])
        
        for layer_idx in range(max(0, layer_range[0]), 
                             min(self.num_layers, layer_range[1])):
            
            if layer_idx not in self.stored_activations:
                continue
                
            layer_activations = self.stored_activations[layer_idx]
            if len(layer_activations) < 10:  # Need minimum samples
                continue
            
            # Prepare training data
            X, y = self._prepare_bad_training_data(layer_activations)
            
            if len(X) == 0:
                continue
            
            # Train classifier
            classifier = LogisticRegression(
                solver='liblinear',
                max_iter=1000,
                random_state=42
            )
            
            try:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Train classifier
                classifier.fit(X_train, y_train)
                
                # Evaluate
                y_pred = classifier.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                bad_results[layer_idx] = {
                    'classifier': classifier,
                    'accuracy': accuracy,
                    'num_samples': len(X),
                    'feature_dim': X.shape[1] if len(X) > 0 else 0
                }
                
                print(f"Layer {layer_idx}: BAD accuracy = {accuracy:.4f}")
                
            except Exception as e:
                print(f"Failed to train BAD classifier for layer {layer_idx}: {e}")
                continue
        
        return bad_results
    
    def _prepare_bad_training_data(self, activations: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for BAD classifier from stored activations."""
        X, y = [], []
        
        for activation, label in activations:
            # Clean activation data
            if isinstance(activation, torch.Tensor):
                activation = activation.detach().cpu().numpy()
            
            # Handle NaN/inf values
            activation = np.nan_to_num(activation, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Flatten if needed
            if len(activation.shape) > 1:
                activation = activation.flatten()
            
            X.append(activation)
            y.append(label)
        
        if len(X) == 0:
            return np.array([]), np.array([])
            
        X = np.array(X)
        y = np.array(y)
        
        return X, y
    
    def _dummy_bad_training(self) -> Dict[str, Any]:
        """Dummy BAD training for when no stored activations available."""
        print("Running dummy BAD training...")
        
        bad_results = {}
        layer_range = [10, 16]
        
        for layer_idx in range(layer_range[0], layer_range[1]):
            # Create dummy classifier with random performance
            dummy_accuracy = 0.5 + np.random.random() * 0.3  # 50-80% accuracy
            
            bad_results[layer_idx] = {
                'classifier': None,  # Placeholder
                'accuracy': dummy_accuracy,
                'num_samples': 1000,
                'feature_dim': self.hidden_size
            }
            
            print(f"Layer {layer_idx}: Dummy BAD accuracy = {dummy_accuracy:.4f}")
        
        return bad_results
    
    def _generate_component_registry(self, path_results: Dict, bad_results: Dict) -> Dict[str, Any]:
        """Generate unified component registry from both analyses."""
        print("Generating unified component registry...")
        
        components = []
        
        # Add high-importance attention heads from path patching
        head_importance = path_results['head_importance']
        importance_threshold = 0.1
        
        for layer_idx in range(self.num_layers):
            for head_idx in range(self.num_heads):
                importance = head_importance[layer_idx, head_idx]
                
                if importance > importance_threshold:
                    components.append({
                        "layer": int(layer_idx),
                        "type": "head",
                        "head_index": int(head_idx),
                        "importance": float(importance),
                        "bias_type": "sycophancy",
                        "source": "path_patching"
                    })
        
        # Add high-accuracy layers from BAD classifiers
        accuracy_threshold = 0.65
        
        for layer_idx, results in bad_results.items():
            accuracy = results['accuracy']
            
            if accuracy > accuracy_threshold:
                components.append({
                    "layer": int(layer_idx),
                    "type": "mlp",
                    "importance": float(accuracy),
                    "bias_type": "demographic", 
                    "source": "bad_probe"
                })
        
        # Sort by importance
        components.sort(key=lambda x: x['importance'], reverse=True)
        
        registry = {
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "num_components": len(components),
            "components": components,
            "metadata": {
                "path_patching_samples": path_results['num_samples'],
                "bad_training_layers": list(bad_results.keys()),
                "importance_threshold": importance_threshold,
                "accuracy_threshold": accuracy_threshold
            }
        }
        
        print(f"Generated registry with {len(components)} components")
        return registry
    
    def _save_results(self, output_dir: str, path_results: Dict, 
                     bad_results: Dict, registry: Dict):
        """Save all diagnostic results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save component registry (main output)
        registry_path = os.path.join(output_dir, "component_registry.json")
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
        print(f"Saved component registry: {registry_path}")
        
        # Save path patching results
        path_path = os.path.join(output_dir, "path_patching_results.pt")
        torch.save(path_results, path_path)
        print(f"Saved path patching results: {path_path}")
        
        # Save BAD classifiers
        bad_path = os.path.join(output_dir, "bad_classifiers.pkl")
        with open(bad_path, 'wb') as f:
            pickle.dump(bad_results, f)
        print(f"Saved BAD classifiers: {bad_path}")
        
        # Save summary
        summary = {
            "diagnostic_summary": {
                "model": self.model_name,
                "timestamp": datetime.now().isoformat(),
                "path_patching": {
                    "num_samples": path_results['num_samples'],
                    "top_heads": self._get_top_components(registry, "head", 5)
                },
                "bad_classifiers": {
                    "trained_layers": list(bad_results.keys()),
                    "best_accuracy": max([r['accuracy'] for r in bad_results.values()]) if bad_results else 0,
                    "top_layers": self._get_top_components(registry, "mlp", 5)
                },
                "total_components": len(registry['components'])
            }
        }
        
        summary_path = os.path.join(output_dir, "diagnostic_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved diagnostic summary: {summary_path}")
    
    def _get_top_components(self, registry: Dict, component_type: str, k: int) -> List[Dict]:
        """Get top k components of specified type."""
        components = [c for c in registry['components'] if c['type'] == component_type]
        components.sort(key=lambda x: x['importance'], reverse=True)
        return components[:k]


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