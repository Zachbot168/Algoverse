#!/usr/bin/env python3
"""
Unified Pipeline Training Utilities

Common utility functions shared across the training components
of the unified pipeline.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# Custom JSON encoder to handle numpy/torch types  
class FIRMJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy/torch types."""
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # torch tensor
            return obj.item()
        elif hasattr(obj, 'tolist'):  # torch tensor
            return obj.tolist()
        return super(FIRMJSONEncoder, self).default(obj)

warnings.filterwarnings('ignore')


def load_component_registry(registry_path: str) -> Dict[str, Any]:
    """
    Load component registry from file.
    
    Args:
        registry_path: Path to component registry JSON file
        
    Returns:
        Registry dictionary
    """
    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"Component registry not found: {registry_path}")
    
    with open(registry_path, 'r') as f:
        return json.load(f)


def save_component_registry(registry: Dict[str, Any], output_path: str) -> None:
    """
    Save component registry to file.
    
    Args:
        registry: Registry dictionary
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(registry, f, indent=2, cls=FIRMJSONEncoder)


def get_model_layer_names(model: nn.Module, model_type: str = "auto") -> List[str]:
    """
    Get layer names for a specific model architecture.
    
    Args:
        model: PyTorch model
        model_type: Model type ('llama', 'qwen', 'mistral', 'auto')
        
    Returns:
        List of layer names
    """
    if model_type == "auto":
        model_type = detect_model_type(model)
    
    layer_names = []
    
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        num_layers = len(model.model.layers)
        
        for i in range(num_layers):
            # Attention layers
            layer_names.extend([
                f"model.layers.{i}.self_attn.q_proj",
                f"model.layers.{i}.self_attn.k_proj", 
                f"model.layers.{i}.self_attn.v_proj",
                f"model.layers.{i}.self_attn.o_proj"
            ])
            
            # MLP layers
            if model_type in ['llama', 'mistral']:
                layer_names.extend([
                    f"model.layers.{i}.mlp.gate_proj",
                    f"model.layers.{i}.mlp.up_proj",
                    f"model.layers.{i}.mlp.down_proj"
                ])
            elif model_type == 'qwen':
                layer_names.extend([
                    f"model.layers.{i}.mlp.w1",
                    f"model.layers.{i}.mlp.w2",
                    f"model.layers.{i}.mlp.c_proj"
                ])
    
    return layer_names


def detect_model_type(model: nn.Module) -> str:
    """
    Detect model architecture type from the model object.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model type string
    """
    model_name = model.__class__.__name__.lower()
    
    if 'llama' in model_name:
        return 'llama'
    elif 'qwen' in model_name:
        return 'qwen'
    elif 'mistral' in model_name:
        return 'mistral'
    else:
        return 'unknown'


def filter_target_modules(all_modules: List[str], 
                         component_layers: List[int],
                         component_types: List[str]) -> List[str]:
    """
    Filter target modules based on component registry.
    
    Args:
        all_modules: All possible module names
        component_layers: List of layer indices to target
        component_types: List of component types ('head', 'mlp')
        
    Returns:
        Filtered list of module names
    """
    filtered_modules = []
    
    for module in all_modules:
        # Extract layer index from module name
        parts = module.split('.')
        
        try:
            layer_idx = int(parts[2])  # Assumes format: model.layers.{i}.{component}
        except (IndexError, ValueError):
            continue
        
        # Check if this layer is in our target list
        if layer_idx not in component_layers:
            continue
        
        # Check component type
        if 'head' in component_types and 'self_attn' in module:
            filtered_modules.append(module)
        elif 'mlp' in component_types and 'mlp' in module:
            filtered_modules.append(module)
    
    return filtered_modules


def create_lora_target_mapping(registry: Dict[str, Any], 
                              base_modules: List[str]) -> Dict[str, List[str]]:
    """
    Create mapping from component types to specific LoRA target modules.
    
    Args:
        registry: Component registry
        base_modules: Base module types (e.g., ['q_proj', 'v_proj'])
        
    Returns:
        Dictionary mapping component types to target modules
    """
    components = registry.get('components', [])
    
    # Group components by layer and type
    layer_components = {}
    for comp in components:
        layer_idx = comp['layer']
        comp_type = comp['type']
        
        if layer_idx not in layer_components:
            layer_components[layer_idx] = {'head': False, 'mlp': False}
        
        layer_components[layer_idx][comp_type] = True
    
    # Generate target modules
    target_mapping = {'head': [], 'mlp': []}
    
    for layer_idx, layer_comps in layer_components.items():
        if layer_comps['head']:
            for base_module in base_modules:
                if base_module in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    target_mapping['head'].append(f"model.layers.{layer_idx}.self_attn.{base_module}")
        
        if layer_comps['mlp']:
            mlp_modules = ['gate_proj', 'up_proj', 'down_proj']  # Default to LLaMA style
            for mlp_module in mlp_modules:
                target_mapping['mlp'].append(f"model.layers.{layer_idx}.mlp.{mlp_module}")
    
    return target_mapping


def compute_parameter_efficiency(model: nn.Module, 
                               target_modules: List[str]) -> Dict[str, Any]:
    """
    Compute parameter efficiency statistics for LoRA targeting.
    
    Args:
        model: PyTorch model
        target_modules: List of target module names
        
    Returns:
        Dictionary with efficiency statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    targeted_params = 0
    
    for name, param in model.named_parameters():
        if any(target in name for target in target_modules):
            targeted_params += param.numel()
    
    efficiency_stats = {
        'total_parameters': total_params,
        'targeted_parameters': targeted_params,
        'targeting_ratio': targeted_params / total_params if total_params > 0 else 0.0,
        'num_target_modules': len(target_modules),
        'efficiency_score': targeted_params / len(target_modules) if target_modules else 0.0
    }
    
    return efficiency_stats


def validate_component_registry(registry: Dict[str, Any]) -> List[str]:
    """
    Validate component registry format and content.
    
    Args:
        registry: Component registry dictionary
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required top-level fields
    required_fields = ['model_name', 'timestamp', 'components']
    for field in required_fields:
        if field not in registry:
            errors.append(f"Missing required field: {field}")
    
    # Check components format
    if 'components' in registry:
        components = registry['components']
        
        if not isinstance(components, list):
            errors.append("Components must be a list")
        else:
            for i, comp in enumerate(components):
                if not isinstance(comp, dict):
                    errors.append(f"Component {i} must be a dictionary")
                    continue
                
                # Check required component fields
                comp_required = ['layer', 'type', 'importance', 'source']
                for field in comp_required:
                    if field not in comp:
                        errors.append(f"Component {i} missing required field: {field}")
                
                # Validate field values
                if 'type' in comp and comp['type'] not in ['head', 'mlp']:
                    errors.append(f"Component {i} has invalid type: {comp['type']}")
                
                if 'importance' in comp and not isinstance(comp['importance'], (int, float)):
                    errors.append(f"Component {i} importance must be numeric")
                
                if 'layer' in comp and not isinstance(comp['layer'], int):
                    errors.append(f"Component {i} layer must be integer")
    
    return errors


def merge_training_configs(base_config: Dict[str, Any], 
                          override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge training configurations with override support.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_training_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def setup_training_environment(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup training environment based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Environment setup information
    """
    import torch
    
    # Device setup
    device = config.get('device', 'auto')
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    # Memory optimization
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    # Set random seeds for reproducibility
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    env_info = {
        'device': device,
        'seed': seed,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': torch.backends.mps.is_available(),
        'num_gpus': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    return env_info


def log_training_metrics(metrics: Dict[str, Any], 
                        log_file: str,
                        step: Optional[int] = None) -> None:
    """
    Log training metrics to file.
    
    Args:
        metrics: Dictionary of metrics to log
        log_file: Path to log file
        step: Optional training step number
    """
    import datetime
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    log_entry = {
        'timestamp': datetime.datetime.now().isoformat(),
        'step': step,
        'metrics': metrics
    }
    
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


def create_symlinks_for_data(config: Dict[str, Any], target_dir: str) -> Dict[str, str]:
    """
    Create symlinks for data directories.
    
    Args:
        config: Configuration with data paths
        target_dir: Target directory for symlinks
        
    Returns:
        Dictionary mapping dataset names to symlink paths
    """
    os.makedirs(target_dir, exist_ok=True)
    
    data_config = config.get('data', {})
    symlinks = {}
    
    for key, source_path in data_config.items():
        if key.endswith('_path') and os.path.exists(source_path):
            dataset_name = key.replace('_path', '')
            symlink_path = os.path.join(target_dir, f"{dataset_name}_data")
            
            # Remove existing symlink if present
            if os.path.islink(symlink_path):
                os.unlink(symlink_path)
            
            try:
                os.symlink(os.path.abspath(source_path), symlink_path)
                symlinks[dataset_name] = symlink_path
            except OSError as e:
                print(f"Warning: Could not create symlink for {dataset_name}: {e}")
    
    return symlinks


def estimate_training_time(num_samples: int, 
                          batch_size: int,
                          num_epochs: int,
                          model_size: int,
                          device: str = 'cuda') -> Dict[str, float]:
    """
    Estimate training time based on configuration.
    
    Args:
        num_samples: Number of training samples
        batch_size: Training batch size
        num_epochs: Number of training epochs
        model_size: Model size in parameters
        device: Training device
        
    Returns:
        Dictionary with time estimates
    """
    # Rough estimates based on empirical data
    steps_per_epoch = num_samples // batch_size
    total_steps = steps_per_epoch * num_epochs
    
    # Time per step estimates (seconds)
    if device == 'cuda':
        # GPU estimates
        if model_size < 1e9:  # < 1B parameters
            time_per_step = 0.5
        elif model_size < 7e9:  # < 7B parameters
            time_per_step = 2.0
        else:  # >= 7B parameters
            time_per_step = 5.0
    else:
        # CPU estimates (much slower)
        if model_size < 1e9:
            time_per_step = 10.0
        else:
            time_per_step = 60.0
    
    estimated_seconds = total_steps * time_per_step
    
    return {
        'steps_per_epoch': steps_per_epoch,
        'total_steps': total_steps,
        'estimated_seconds': estimated_seconds,
        'estimated_minutes': estimated_seconds / 60,
        'estimated_hours': estimated_seconds / 3600,
        'time_per_step': time_per_step
    }


def cleanup_training_artifacts(output_dir: str, keep_final: bool = True) -> None:
    """
    Clean up training artifacts to save space.
    
    Args:
        output_dir: Training output directory
        keep_final: Whether to keep final model checkpoints
    """
    if not os.path.exists(output_dir):
        return
    
    # Remove intermediate checkpoints
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        
        if os.path.isdir(item_path) and item.startswith('checkpoint-'):
            if not keep_final or item != 'checkpoint-final':
                import shutil
                shutil.rmtree(item_path)
                print(f"Removed checkpoint: {item_path}")
    
    # Remove large temporary files
    temp_patterns = ['*.tmp', '*.cache', 'training_state.json']
    for pattern in temp_patterns:
        import glob
        for temp_file in glob.glob(os.path.join(output_dir, pattern)):
            os.remove(temp_file)
            print(f"Removed temp file: {temp_file}")


if __name__ == "__main__":
    # Demo usage
    print("Unified Pipeline Training Utilities")
    print("This module provides shared utilities for training components.")
    
    # Example registry validation
    example_registry = {
        'model_name': 'test-model',
        'timestamp': '2024-01-01T00:00:00Z',
        'components': [
            {
                'layer': 10,
                'type': 'head',
                'head_index': 5,
                'importance': 0.85,
                'bias_type': 'sycophancy',
                'source': 'path_patching'
            }
        ]
    }
    
    errors = validate_component_registry(example_registry)
    if errors:
        print(f"Registry validation errors: {errors}")
    else:
        print("Example registry is valid")
    
    # Example environment setup
    example_config = {'device': 'auto', 'seed': 42}
    env_info = setup_training_environment(example_config)
    print(f"Environment setup: {env_info}")