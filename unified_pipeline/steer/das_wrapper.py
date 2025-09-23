#!/usr/bin/env python3
"""
Dynamic Activation Steering (DAS) Wrapper

Implements runtime activation steering using the unified pipeline approach:
1. BAD (Biased Activation Detection) probes check for bias
2. DSV (Debiasing Steering Vectors) are applied when bias is detected
3. Compatible with HuggingFace generate() via forward hooks

This system provides inference-time debiasing that works with both:
- Original models (baseline steering)
- Fine-tuned models from pinpoint tuning (residual steering)
"""

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from train.component_registry import ComponentRegistryManager

warnings.filterwarnings('ignore')


class BiasActivationDetector:
    """
    BAD (Biased Activation Detection) component that predicts bias from activations.
    """
    
    def __init__(self, classifiers: Dict[int, LogisticRegression]):
        """
        Initialize BAD detector with trained classifiers.
        
        Args:
            classifiers: Dictionary mapping layer indices to trained classifiers
        """
        self.classifiers = classifiers
        self.optimal_layer = self._find_optimal_layer()
        
    def _find_optimal_layer(self) -> Optional[int]:
        """Find the layer with the best classifier performance."""
        if not self.classifiers:
            return None
            
        # Assume classifiers have an accuracy attribute or use default
        best_layer = None
        best_score = 0.0
        
        for layer_idx, classifier in self.classifiers.items():
            # Try to get stored accuracy, fallback to baseline score
            # TODO: Implement proper classifier evaluation instead of fallback
            score = getattr(classifier, 'accuracy_', 0.5)  # Baseline random performance
            if score > best_score:
                best_score = score
                best_layer = layer_idx
        
        return best_layer
    
    def predict_bias(self, activations: torch.Tensor, layer_idx: Optional[int] = None) -> float:
        """
        Predict bias probability from layer activations.
        
        Args:
            activations: Layer activations tensor [batch_size, seq_len, hidden_size]
            layer_idx: Specific layer to use (defaults to optimal layer)
            
        Returns:
            Bias probability (0.0 = unbiased, 1.0 = biased)
        """
        if layer_idx is None:
            layer_idx = self.optimal_layer
            
        if layer_idx not in self.classifiers:
            return 0.0  # No classifier available
        
        classifier = self.classifiers[layer_idx]
        
        # Extract features from activations (use last token)
        if len(activations.shape) == 3:
            # [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size]
            features = activations[:, -1, :].detach().cpu().numpy()
        else:
            features = activations.detach().cpu().numpy()
        
        # Handle batch dimension
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Clean features
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        try:
            # Get bias probability
            bias_probs = classifier.predict_proba(features)
            # Assume class 1 is biased, class 0 is unbiased
            if bias_probs.shape[1] > 1:
                return float(bias_probs[0, 1])  # Probability of bias
            else:
                return float(bias_probs[0, 0])  # Single class prediction
        except Exception as e:
            print(f"Warning: BAD prediction failed for layer {layer_idx}: {e}")
            return 0.0


class DebiasingSteering:
    """
    DSV (Debiasing Steering Vector) component that applies corrections to activations.
    """
    
    def __init__(self, steering_vectors: Dict[str, torch.Tensor], 
                 magnitude_scale: float = 1.0):
        """
        Initialize debiasing steering component.
        
        Args:
            steering_vectors: Dictionary mapping bias types to steering vectors
            magnitude_scale: Global scaling factor for steering magnitude
        """
        self.steering_vectors = steering_vectors
        self.magnitude_scale = magnitude_scale
        
    def apply_steering(self, activations: torch.Tensor, 
                      bias_type: str = "general",
                      confidence: float = 1.0) -> torch.Tensor:
        """
        Apply debiasing steering to activations.
        
        Args:
            activations: Layer activations to correct
            bias_type: Type of bias to correct ('general', 'gender', 'race', etc.)
            confidence: Confidence in bias detection (0.0-1.0)
            
        Returns:
            Corrected activations
        """
        if bias_type not in self.steering_vectors:
            bias_type = "general"  # Fallback to general steering
            
        if bias_type not in self.steering_vectors:
            return activations  # No steering vector available
        
        steering_vector = self.steering_vectors[bias_type]
        
        # Ensure steering vector is on same device
        if steering_vector.device != activations.device:
            steering_vector = steering_vector.to(activations.device)
        
        # Scale steering by confidence and magnitude
        scaled_steering = steering_vector * confidence * self.magnitude_scale
        
        # Apply steering (broadcast across batch and sequence dimensions)
        if len(activations.shape) == 3:  # [batch, seq, hidden]
            # Expand steering to match activations shape
            expanded_steering = scaled_steering.unsqueeze(0).unsqueeze(0)
            corrected = activations + expanded_steering
        elif len(activations.shape) == 2:  # [batch, hidden]
            expanded_steering = scaled_steering.unsqueeze(0)  
            corrected = activations + expanded_steering
        else:
            corrected = activations + scaled_steering
        
        return corrected


class DynamicActivationSteering:
    """
    Main DAS (Dynamic Activation Steering) wrapper that combines BAD and DSV.
    
    This class wraps a HuggingFace model and provides real-time bias detection
    and correction during inference.
    """
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                 bad_detector: BiasActivationDetector,
                 debiasing_steering: DebiasingSteering,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize DAS wrapper.
        
        Args:
            model: HuggingFace model to wrap
            tokenizer: Associated tokenizer
            bad_detector: BAD component for bias detection
            debiasing_steering: DSV component for bias correction
            config: DAS configuration parameters
        """
        self.model = model
        self.tokenizer = tokenizer
        self.bad_detector = bad_detector
        self.debiasing_steering = debiasing_steering
        
        # Configuration
        self.config = config or {}
        self.threshold = self.config.get('threshold', 0.5)
        self.apply_scaling = self.config.get('apply_scaling', True)
        self.max_intervention = self.config.get('max_intervention', 2.0)
        
        # Hook management
        self.hooks = []
        self.hook_layers = self._determine_hook_layers()
        self.is_active = False
        
        # Statistics tracking
        self.stats = {
            'total_tokens': 0,
            'bias_detected': 0,
            'interventions_applied': 0,
            'avg_bias_confidence': 0.0
        }
        
        print(f"Initialized DAS wrapper with {len(self.hook_layers)} hook layers")
        print(f"Bias threshold: {self.threshold}, Max intervention: {self.max_intervention}")
    
    def _determine_hook_layers(self) -> List[int]:
        """Determine which layers should have steering hooks."""
        hook_config = self.config.get('hook_layers', 'auto')
        
        if hook_config == 'auto':
            # Use layers where we have BAD classifiers
            return list(self.bad_detector.classifiers.keys())
        elif isinstance(hook_config, list):
            return hook_config
        elif isinstance(hook_config, int):
            return [hook_config]
        else:
            # Default to middle layers
            num_layers = len(self.model.model.layers)
            return list(range(num_layers // 2, num_layers * 3 // 4))
    
    def _create_steering_hook(self, layer_idx: int) -> Callable:
        """Create a forward hook for the specified layer."""
        
        def steering_hook(module, input, output):
            """Forward hook that applies dynamic steering."""
            if not self.is_active:
                return output
            
            # Get activations (output of the layer)
            activations = output[0] if isinstance(output, tuple) else output
            
            # Detect bias using BAD
            bias_prob = self.bad_detector.predict_bias(activations, layer_idx)
            
            # Update statistics
            self.stats['total_tokens'] += activations.size(0) * activations.size(1)
            
            if bias_prob > self.threshold:
                self.stats['bias_detected'] += activations.size(0) * activations.size(1)
                
                # Calculate intervention strength
                if self.apply_scaling:
                    confidence = min(bias_prob, self.max_intervention)
                else:
                    confidence = 1.0 if bias_prob > self.threshold else 0.0
                
                # Apply steering
                corrected_activations = self.debiasing_steering.apply_steering(
                    activations, bias_type="general", confidence=confidence
                )
                
                self.stats['interventions_applied'] += activations.size(0) * activations.size(1)
                self.stats['avg_bias_confidence'] = (
                    (self.stats['avg_bias_confidence'] * (self.stats['bias_detected'] - 1) + bias_prob) 
                    / self.stats['bias_detected']
                )
                
                # Return corrected output in same format as input
                if isinstance(output, tuple):
                    return (corrected_activations,) + output[1:]
                else:
                    return corrected_activations
            
            return output
        
        return steering_hook
    
    def activate_steering(self) -> None:
        """Activate dynamic steering by installing forward hooks."""
        if self.is_active:
            return
        
        print("Activating dynamic steering...")
        
        # Install hooks on specified layers
        for layer_idx in self.hook_layers:
            if layer_idx < len(self.model.model.layers):
                layer = self.model.model.layers[layer_idx]
                hook = layer.register_forward_hook(self._create_steering_hook(layer_idx))
                self.hooks.append(hook)
        
        self.is_active = True
        print(f"Installed {len(self.hooks)} steering hooks")
    
    def deactivate_steering(self) -> None:
        """Deactivate dynamic steering by removing forward hooks."""
        if not self.is_active:
            return
        
        print("Deactivating dynamic steering...")
        
        # Remove all hooks
        for hook in self.hooks:
            hook.remove()
        
        self.hooks.clear()
        self.is_active = False
        print("Removed all steering hooks")
    
    def generate(self, *args, **kwargs) -> Any:
        """
        Generate text with dynamic steering active.
        
        This method wraps the model's generate() method while ensuring
        steering is active during generation.
        """
        was_active = self.is_active
        
        try:
            # Activate steering if not already active
            if not was_active:
                self.activate_steering()
            
            # Call original generate method
            result = self.model.generate(*args, **kwargs)
            
            return result
            
        finally:
            # Restore original state
            if not was_active:
                self.deactivate_steering()
    
    def forward(self, *args, **kwargs) -> Any:
        """
        Forward pass with dynamic steering.
        
        This method wraps the model's forward() method.
        """
        was_active = self.is_active
        
        try:
            if not was_active:
                self.activate_steering()
            
            result = self.model(*args, **kwargs)
            return result
            
        finally:
            if not was_active:
                self.deactivate_steering()
    
    def __call__(self, *args, **kwargs) -> Any:
        """Make the wrapper callable like the original model."""
        return self.forward(*args, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get steering statistics."""
        total_tokens = max(self.stats['total_tokens'], 1)  # Avoid division by zero
        
        return {
            'total_tokens_processed': self.stats['total_tokens'],
            'bias_detected_tokens': self.stats['bias_detected'],
            'interventions_applied_tokens': self.stats['interventions_applied'],
            'bias_detection_rate': self.stats['bias_detected'] / total_tokens,
            'intervention_rate': self.stats['interventions_applied'] / total_tokens,
            'avg_bias_confidence': self.stats['avg_bias_confidence'],
            'is_active': self.is_active,
            'num_hooks': len(self.hooks)
        }
    
    def reset_stats(self) -> None:
        """Reset steering statistics."""
        self.stats = {
            'total_tokens': 0,
            'bias_detected': 0,
            'interventions_applied': 0,
            'avg_bias_confidence': 0.0
        }
    
    def save_steering_config(self, output_path: str) -> None:
        """Save current steering configuration."""
        config = {
            'das_config': self.config,
            'hook_layers': self.hook_layers,
            'threshold': self.threshold,
            'max_intervention': self.max_intervention,
            'stats': self.get_stats()
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Saved steering configuration to: {output_path}")


def load_bad_classifiers(bad_path: str) -> Dict[int, LogisticRegression]:
    """Load BAD classifiers from file."""
    if not os.path.exists(bad_path):
        print(f"Warning: BAD classifiers not found at {bad_path}")
        return {}
    
    with open(bad_path, 'rb') as f:
        bad_results = pickle.load(f)
    
    classifiers = {}
    for layer_idx, results in bad_results.items():
        if 'classifier' in results and results['classifier'] is not None:
            classifier = results['classifier']
            # Store accuracy for optimal layer selection
            classifier.accuracy_ = results.get('accuracy', 0.5)
            classifiers[int(layer_idx)] = classifier
    
    return classifiers


def load_steering_vectors(dsv_path: str) -> Dict[str, torch.Tensor]:
    """Load steering vectors from file."""
    if not os.path.exists(dsv_path):
        raise FileNotFoundError(
            f"Steering vectors not found at {dsv_path}. "
            f"Please ensure steering vectors have been computed and saved using "
            f"real_steering_vectors.py before attempting to load them. "
            f"No fake or random vectors will be generated."
        )
    
    with open(dsv_path, 'rb') as f:
        steering_data = pickle.load(f)
    
    # Convert numpy arrays to torch tensors if needed
    steering_vectors = {}
    for bias_type, vector in steering_data.items():
        if isinstance(vector, np.ndarray):
            vector = torch.from_numpy(vector).float()
        steering_vectors[bias_type] = vector
    
    return steering_vectors


def create_das_wrapper(model: AutoModelForCausalLM, 
                      tokenizer: AutoTokenizer,
                      diagnostic_dir: str,
                      config: Dict[str, Any]) -> DynamicActivationSteering:
    """
    Create a DAS wrapper from diagnostic results.
    
    Args:
        model: HuggingFace model to wrap
        tokenizer: Associated tokenizer
        diagnostic_dir: Directory containing diagnostic results
        config: DAS configuration
        
    Returns:
        DAS wrapper ready for inference
    """
    print(f"Creating DAS wrapper from diagnostics in: {diagnostic_dir}")
    
    # Load BAD classifiers
    bad_path = os.path.join(diagnostic_dir, "bad_classifiers.pkl")
    classifiers = load_bad_classifiers(bad_path)
    bad_detector = BiasActivationDetector(classifiers)
    
    # Load steering vectors
    dsv_path = os.path.join(diagnostic_dir, "steering_vectors.pkl")
    steering_vectors = load_steering_vectors(dsv_path)
    steering_config = config.get('steering', {}).get('dsv', {})
    magnitude_scale = steering_config.get('magnitude_scale', 1.0)
    debiasing_steering = DebiasingSteering(steering_vectors, magnitude_scale)
    
    # Create DAS wrapper
    das_config = config.get('steering', {}).get('das', {})
    wrapper = DynamicActivationSteering(
        model=model,
        tokenizer=tokenizer,
        bad_detector=bad_detector,
        debiasing_steering=debiasing_steering,
        config=das_config
    )
    
    return wrapper


def main():
    """Demo usage of DAS wrapper."""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Test DAS wrapper")
    parser.add_argument("--config", required=True, help="Configuration file")
    parser.add_argument("--diagnostic_dir", required=True, help="Diagnostic results directory")
    parser.add_argument("--prompt", default="What do you think about diversity in tech?", help="Test prompt")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model and tokenizer
    model_name = config['model']['name']
    print(f"Loading model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create DAS wrapper
    das_wrapper = create_das_wrapper(model, tokenizer, args.diagnostic_dir, config)
    
    # Test generation
    print(f"\nTesting with prompt: {args.prompt}")
    
    # Tokenize input
    inputs = tokenizer(args.prompt, return_tensors="pt")
    
    # Generate without steering
    print("\nGenerating without steering...")
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    original_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Original: {original_text}")
    
    # Generate with steering
    print("\nGenerating with steering...")
    with torch.no_grad():
        steered_outputs = das_wrapper.generate(
            inputs.input_ids,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    steered_text = tokenizer.decode(steered_outputs[0], skip_special_tokens=True)
    print(f"Steered: {steered_text}")
    
    # Print statistics
    print("\nSteering Statistics:")
    stats = das_wrapper.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()