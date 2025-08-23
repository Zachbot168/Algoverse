#!/usr/bin/env python3
"""
Simple FairSteer Wrapper for Model Variant Loading

Provides a lightweight wrapper that applies FairSteer steering vectors
during inference without the complexity of the full DAS system.
This is specifically designed for the model variant loader interface.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
from transformers import AutoModelForCausalLM


class SimpleFairSteerWrapper(nn.Module):
    """
    Simple wrapper that applies FairSteer steering vectors at a specific layer.
    
    This wrapper intercepts activations at the optimal layer and applies
    the appropriate steering vector based on the intervention strength.
    """
    
    def __init__(self, 
                 model: AutoModelForCausalLM,
                 steering_vectors: Dict[str, torch.Tensor],
                 optimal_layer: int,
                 intervention_strength: float = 1.0):
        """
        Initialize the FairSteer wrapper.
        
        Args:
            model: Base model to wrap
            steering_vectors: Dictionary of steering vectors by layer
            optimal_layer: Layer index to apply steering
            intervention_strength: Scaling factor for steering vectors
        """
        super().__init__()
        self.model = model
        self.steering_vectors = steering_vectors
        self.optimal_layer = optimal_layer
        self.intervention_strength = intervention_strength
        
        # Convert steering vectors to proper format and device
        self._prepare_steering_vectors()
        
        # Install the forward hook
        self.hook_handle = None
        self._install_hook()
    
    def _prepare_steering_vectors(self):
        """Prepare steering vectors for use."""
        # Get the steering vector for the optimal layer
        layer_key = str(self.optimal_layer)
        if layer_key in self.steering_vectors:
            self.steering_vector = self.steering_vectors[layer_key]
        elif 'general' in self.steering_vectors:
            self.steering_vector = self.steering_vectors['general']
        else:
            # Use the first available steering vector
            self.steering_vector = list(self.steering_vectors.values())[0]
        
        # Ensure it's a tensor
        if not isinstance(self.steering_vector, torch.Tensor):
            self.steering_vector = torch.tensor(self.steering_vector)
        
        # Move to model's device
        if hasattr(self.model, 'device'):
            self.steering_vector = self.steering_vector.to(self.model.device)
    
    def _install_hook(self):
        """Install the forward hook at the optimal layer."""
        if self.hook_handle is not None:
            return  # Already installed
        
        try:
            # Get the target layer
            target_layer = self.model.model.layers[self.optimal_layer]
            
            def steering_hook(module, input, output):
                """Apply steering to the layer output."""
                # Get the hidden states
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                # Ensure steering vector is on the same device
                if self.steering_vector.device != hidden_states.device:
                    self.steering_vector = self.steering_vector.to(hidden_states.device)
                
                # Apply steering with proper broadcasting
                # hidden_states shape: [batch_size, seq_len, hidden_size]
                # steering_vector shape: [hidden_size] or [1, hidden_size]
                
                if len(self.steering_vector.shape) == 1:
                    # Expand to [1, 1, hidden_size] for broadcasting
                    steering = self.steering_vector.unsqueeze(0).unsqueeze(0)
                elif len(self.steering_vector.shape) == 2 and self.steering_vector.shape[0] == 1:
                    # Expand to [1, 1, hidden_size] for broadcasting
                    steering = self.steering_vector.unsqueeze(0)
                else:
                    steering = self.steering_vector
                
                # Apply steering with intervention strength
                steered_states = hidden_states + (steering * self.intervention_strength)
                
                # Return in the same format as input
                if isinstance(output, tuple):
                    return (steered_states,) + output[1:]
                else:
                    return steered_states
            
            # Register the hook
            self.hook_handle = target_layer.register_forward_hook(steering_hook)
            
        except Exception as e:
            print(f"Warning: Failed to install steering hook at layer {self.optimal_layer}: {e}")
    
    def _remove_hook(self):
        """Remove the forward hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
    
    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped model."""
        return self.model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """Generate method that uses the wrapped model."""
        return self.model.generate(*args, **kwargs)
    
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
    
    def __call__(self, *args, **kwargs):
        """Make the wrapper callable."""
        return self.forward(*args, **kwargs)
    
    def __del__(self):
        """Clean up the hook when the wrapper is destroyed."""
        self._remove_hook()