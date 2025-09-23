#!/usr/bin/env python3
"""
Component Registry System

Manages the unified registry of important model components identified by
both path patching (sycophancy heads) and BAD probes (bias layers).

This registry is used by:
- Pinpoint Tuning: To select which parameters get LoRA adapters
- DSV Computation: To determine optimal steering layers
- Monitoring: To track component importance over time
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import numpy as np

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


@dataclass
class ComponentInfo:
    """Information about a single model component."""
    layer: int
    type: str  # 'head' or 'mlp'
    importance: float
    bias_type: str  # 'sycophancy', 'demographic', etc.
    source: str  # 'path_patching' or 'bad_probe'
    head_index: Optional[int] = None  # Only for attention heads
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ComponentRegistry:
    """Registry containing all identified important components."""
    model_name: str
    timestamp: str
    num_components: int
    components: List[ComponentInfo]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "num_components": self.num_components,
            "components": [asdict(comp) for comp in self.components],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComponentRegistry':
        """Create registry from dictionary."""
        components = [ComponentInfo(**comp) for comp in data['components']]
        return cls(
            model_name=data['model_name'],
            timestamp=data['timestamp'],
            num_components=data['num_components'],
            components=components,
            metadata=data['metadata']
        )


class ComponentRegistryManager:
    """Manages component registry operations."""
    
    def __init__(self, registry_dir: str = "train"):
        """Initialize registry manager."""
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True)
        self.current_registry: Optional[ComponentRegistry] = None
    
    def create_registry(self, model_name: str, 
                       path_patching_results: Dict[str, Any],
                       bad_results: Dict[str, Any],
                       config: Optional[Dict[str, Any]] = None) -> ComponentRegistry:
        """
        Create a new component registry from diagnostic results.
        
        Args:
            model_name: Name of the model
            path_patching_results: Results from path patching analysis
            bad_results: Results from BAD classifier training
            config: Optional configuration parameters
            
        Returns:
            ComponentRegistry object
        """
        components = []
        config = config or {}
        
        # Extract attention heads from path patching
        if 'head_importance' in path_patching_results:
            head_importance = path_patching_results['head_importance']
            importance_threshold = config.get('head_importance_threshold', 0.1)
            
            num_layers, num_heads = head_importance.shape
            
            for layer_idx in range(num_layers):
                for head_idx in range(num_heads):
                    importance = float(head_importance[layer_idx, head_idx])
                    
                    if importance > importance_threshold:
                        components.append(ComponentInfo(
                            layer=layer_idx,
                            type="head",
                            head_index=head_idx,
                            importance=importance,
                            bias_type="sycophancy",
                            source="path_patching",
                            metadata={
                                "num_samples": path_patching_results.get('num_samples', 0)
                            }
                        ))
        
        # Extract MLP layers from BAD results
        accuracy_threshold = config.get('bad_accuracy_threshold', 0.65)
        
        for layer_idx, results in bad_results.items():
            if isinstance(layer_idx, str):
                layer_idx = int(layer_idx)
                
            accuracy = results.get('accuracy', 0.0)
            
            if accuracy > accuracy_threshold:
                components.append(ComponentInfo(
                    layer=layer_idx,
                    type="mlp",
                    importance=accuracy,
                    bias_type="demographic",
                    source="bad_probe",
                    metadata={
                        "num_samples": results.get('num_samples', 0),
                        "feature_dim": results.get('feature_dim', 0)
                    }
                ))
        
        # Sort components by importance
        components.sort(key=lambda x: x.importance, reverse=True)
        
        # Create registry
        registry = ComponentRegistry(
            model_name=model_name,
            timestamp=datetime.now().isoformat(),
            num_components=len(components),
            components=components,
            metadata={
                "path_patching_samples": path_patching_results.get('num_samples', 0),
                "bad_training_layers": list(bad_results.keys()),
                "thresholds": {
                    "head_importance": importance_threshold,
                    "bad_accuracy": accuracy_threshold
                },
                "config": config
            }
        )
        
        self.current_registry = registry
        return registry
    
    def save_registry(self, registry: ComponentRegistry, 
                     filename: str = "component_registry.json") -> str:
        """Save registry to file."""
        filepath = self.registry_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(registry.to_dict(), f, indent=2, cls=FIRMJSONEncoder)
        
        print(f"Saved component registry to: {filepath}")
        return str(filepath)
    
    def load_registry(self, filename: str = "component_registry.json") -> ComponentRegistry:
        """Load registry from file."""
        filepath = self.registry_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Registry file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        registry = ComponentRegistry.from_dict(data)
        self.current_registry = registry
        return registry
    
    def get_components_for_lora(self, max_components: int = 32,
                              prioritize_heads: bool = True) -> List[ComponentInfo]:
        """
        Get components suitable for LoRA fine-tuning.
        
        Args:
            max_components: Maximum number of components to return
            prioritize_heads: Whether to prioritize attention heads
            
        Returns:
            List of components for LoRA targeting
        """
        if not self.current_registry:
            raise ValueError("No registry loaded. Call load_registry() first.")
        
        components = self.current_registry.components.copy()
        
        if prioritize_heads:
            # Sort: heads first, then by importance
            components.sort(key=lambda x: (x.type != "head", -x.importance))
        else:
            # Sort by importance only
            components.sort(key=lambda x: -x.importance)
        
        return components[:max_components]
    
    def get_steering_layers(self, layer_range: Optional[List[int]] = None) -> List[int]:
        """
        Get layers suitable for steering vector application.
        
        Args:
            layer_range: Optional range [min_layer, max_layer] to constrain results
            
        Returns:
            List of layer indices for steering
        """
        if not self.current_registry:
            raise ValueError("No registry loaded. Call load_registry() first.")
        
        # Get all layers with bias detection capability (from BAD probes)
        steering_layers = []
        
        for component in self.current_registry.components:
            if component.source == "bad_probe" and component.type == "mlp":
                layer_idx = component.layer
                
                # Apply layer range filter if specified
                if layer_range is None or (layer_range[0] <= layer_idx <= layer_range[1]):
                    steering_layers.append(layer_idx)
        
        return sorted(list(set(steering_layers)))
    
    def get_optimal_steering_layer(self) -> Optional[int]:
        """Get the single best layer for steering based on BAD accuracy."""
        if not self.current_registry:
            return None
        
        best_layer = None
        best_accuracy = 0.0
        
        for component in self.current_registry.components:
            if component.source == "bad_probe" and component.importance > best_accuracy:
                best_accuracy = component.importance
                best_layer = component.layer
        
        return best_layer
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the current registry."""
        if not self.current_registry:
            return {}
        
        registry = self.current_registry
        
        # Count components by type and source
        counts = {
            "total": len(registry.components),
            "by_type": {"head": 0, "mlp": 0},
            "by_source": {"path_patching": 0, "bad_probe": 0},
            "by_bias_type": {}
        }
        
        importance_stats = {
            "mean": 0.0,
            "std": 0.0,
            "min": float('inf'),
            "max": float('-inf')
        }
        
        importances = []
        
        for component in registry.components:
            # Count by type
            counts["by_type"][component.type] += 1
            
            # Count by source
            counts["by_source"][component.source] += 1
            
            # Count by bias type
            bias_type = component.bias_type
            counts["by_bias_type"][bias_type] = counts["by_bias_type"].get(bias_type, 0) + 1
            
            # Collect importance values
            importances.append(component.importance)
        
        # Calculate importance statistics
        if importances:
            importances = np.array(importances)
            importance_stats = {
                "mean": float(np.mean(importances)),
                "std": float(np.std(importances)),
                "min": float(np.min(importances)),
                "max": float(np.max(importances))
            }
        
        return {
            "model_name": registry.model_name,
            "timestamp": registry.timestamp,
            "counts": counts,
            "importance_stats": importance_stats,
            "top_components": [
                {
                    "layer": comp.layer,
                    "type": comp.type,
                    "head_index": comp.head_index,
                    "importance": comp.importance,
                    "bias_type": comp.bias_type,
                    "source": comp.source
                }
                for comp in registry.components[:10]  # Top 10
            ]
        }
    
    def filter_components(self, 
                         layer_range: Optional[List[int]] = None,
                         component_types: Optional[List[str]] = None,
                         bias_types: Optional[List[str]] = None,
                         min_importance: Optional[float] = None) -> List[ComponentInfo]:
        """
        Filter components based on various criteria.
        
        Args:
            layer_range: [min_layer, max_layer] to include
            component_types: List of component types to include ('head', 'mlp')
            bias_types: List of bias types to include ('sycophancy', 'demographic')
            min_importance: Minimum importance threshold
            
        Returns:
            Filtered list of components
        """
        if not self.current_registry:
            return []
        
        components = self.current_registry.components
        
        # Apply filters
        if layer_range is not None:
            components = [c for c in components 
                         if layer_range[0] <= c.layer <= layer_range[1]]
        
        if component_types is not None:
            components = [c for c in components if c.type in component_types]
        
        if bias_types is not None:
            components = [c for c in components if c.bias_type in bias_types]
        
        if min_importance is not None:
            components = [c for c in components if c.importance >= min_importance]
        
        return components
    
    def merge_registries(self, other_registry: ComponentRegistry, 
                        strategy: str = "union") -> ComponentRegistry:
        """
        Merge current registry with another registry.
        
        Args:
            other_registry: Registry to merge with
            strategy: Merge strategy ('union', 'intersection', 'update')
            
        Returns:
            New merged registry
        """
        if not self.current_registry:
            return other_registry
        
        current = self.current_registry
        
        if strategy == "union":
            # Combine all components, removing duplicates
            seen = set()
            merged_components = []
            
            for component in current.components + other_registry.components:
                # Create unique key for component
                key = (component.layer, component.type, component.head_index)
                
                if key not in seen:
                    seen.add(key)
                    merged_components.append(component)
                else:
                    # Keep component with higher importance
                    for i, existing in enumerate(merged_components):
                        existing_key = (existing.layer, existing.type, existing.head_index)
                        if existing_key == key and component.importance > existing.importance:
                            merged_components[i] = component
                            break
            
        elif strategy == "intersection":
            # Keep only components present in both registries
            current_keys = {(c.layer, c.type, c.head_index) for c in current.components}
            other_keys = {(c.layer, c.type, c.head_index) for c in other_registry.components}
            common_keys = current_keys & other_keys
            
            merged_components = [c for c in current.components 
                               if (c.layer, c.type, c.head_index) in common_keys]
            
        elif strategy == "update":
            # Update current with other, overwriting duplicates
            other_dict = {(c.layer, c.type, c.head_index): c for c in other_registry.components}
            merged_components = []
            
            for component in current.components:
                key = (component.layer, component.type, component.head_index)
                if key in other_dict:
                    merged_components.append(other_dict[key])  # Use updated version
                    del other_dict[key]
                else:
                    merged_components.append(component)  # Keep original
            
            # Add remaining components from other registry
            merged_components.extend(other_dict.values())
        
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")
        
        # Sort by importance
        merged_components.sort(key=lambda x: x.importance, reverse=True)
        
        # Create new registry
        merged_registry = ComponentRegistry(
            model_name=current.model_name,
            timestamp=datetime.now().isoformat(),
            num_components=len(merged_components),
            components=merged_components,
            metadata={
                "merged_from": [current.timestamp, other_registry.timestamp],
                "merge_strategy": strategy,
                "original_counts": [current.num_components, other_registry.num_components]
            }
        )
        
        self.current_registry = merged_registry
        return merged_registry


def main():
    """Demo usage of ComponentRegistryManager."""
    # Create example registry
    manager = ComponentRegistryManager()
    
    # Example diagnostic results (for demo only - replace with real data)
    example_path_results = {
        "head_importance": np.zeros((12, 8)),  # TODO: Replace with real head importance data
        "num_samples": 1000
    }
    
    example_bad_results = {
        10: {"accuracy": 0.72, "num_samples": 800, "feature_dim": 1024},
        11: {"accuracy": 0.68, "num_samples": 800, "feature_dim": 1024},
        14: {"accuracy": 0.75, "num_samples": 800, "feature_dim": 1024}
    }
    
    # Create registry
    registry = manager.create_registry(
        model_name="example/model",
        path_patching_results=example_path_results,
        bad_results=example_bad_results
    )
    
    # Save registry
    manager.save_registry(registry)
    
    # Print summary
    summary = manager.get_registry_summary()
    print("\nRegistry Summary:")
    print(json.dumps(summary, indent=2))
    
    # Get components for LoRA
    lora_components = manager.get_components_for_lora(max_components=10)
    print(f"\nTop {len(lora_components)} components for LoRA:")
    for comp in lora_components:
        print(f"  Layer {comp.layer}, {comp.type}, importance: {comp.importance:.4f}")
    
    # Get steering layers
    steering_layers = manager.get_steering_layers()
    print(f"\nSteering layers: {steering_layers}")
    print(f"Optimal steering layer: {manager.get_optimal_steering_layer()}")


class FIRMPipeline:
    """Simple wrapper for FIRM pipeline functionality."""
    
    def __init__(self, config_path: str):
        """Initialize FIRM pipeline with model config."""
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        print(f"✅ FIRM Pipeline initialized with config: {config_path}")
    
    def run_pipeline(self, model_name: str, output_dir: str = None) -> Dict[str, Any]:
        """Run FIRM pipeline (placeholder for full implementation)."""
        return {
            "status": "initialized",
            "model_name": model_name,
            "config_path": str(self.config_path)
        }


if __name__ == "__main__":
    main()