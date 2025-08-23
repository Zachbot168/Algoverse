#!/usr/bin/env python3
"""
Multi-Layer Intervention Framework - FIRM Phase 5

Extends FairSteer methodology to support multi-layer steering interventions,
testing robustness when steering is applied to downstream or unrelated layers.
"""

import json
import os
from typing import Dict, List, Tuple, Any, Optional
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

warnings.filterwarnings('ignore')


class MultiLayerSteering:
    """
    FIRM Phase 5: Multi-layer intervention framework for testing
    downstream robustness and joint steering strategies.
    """
    
    def __init__(self, model, tokenizer, config: Dict[str, Any]):
        """
        Initialize multi-layer steering framework.
        
        Args:
            model: HuggingFace model
            tokenizer: Associated tokenizer
            config: Configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.config = config
        
        # Model architecture info
        self.num_layers = len(model.model.layers)
        self.hidden_size = model.config.hidden_size
        
        # Multi-layer steering configuration
        self.steering_config = config.get('firm_config', {}).get('multi_layer_steering', {})
        self.max_concurrent_layers = self.steering_config.get('max_concurrent_layers', 3)
        self.layer_interaction_strength = self.steering_config.get('interaction_strength', 1.0)
        
        # Steering vectors storage
        self.single_layer_vectors: Dict[int, torch.Tensor] = {}
        self.multi_layer_vectors: Dict[str, Dict[int, torch.Tensor]] = {}
        self.intervention_hooks: List[Any] = []
        
        print(f"Initialized MultiLayerSteering for {self.num_layers} layers")
        print(f"Max concurrent layers: {self.max_concurrent_layers}")
    
    def load_single_layer_vectors(self, steering_vectors_path: str) -> None:
        """Load single-layer steering vectors from previous computation."""
        try:
            import pickle
            with open(steering_vectors_path, 'rb') as f:
                data = pickle.load(f)
            
            # Extract single-layer vectors from the data structure
            if isinstance(data, dict) and 'steering_vectors' in data:
                vectors_data = data['steering_vectors']
                
                # Find single-layer vectors (prefer causal_aligned if available)
                if 'causal_aligned' in vectors_data:
                    layer_vectors = vectors_data['causal_aligned']
                elif 'training_aligned' in vectors_data:
                    layer_vectors = vectors_data['training_aligned']
                else:
                    # Take first available strategy
                    layer_vectors = list(vectors_data.values())[0]
                
                # Convert numpy arrays back to tensors
                for layer_idx, vector_np in layer_vectors.items():
                    self.single_layer_vectors[int(layer_idx)] = torch.tensor(
                        vector_np, device=self.device, dtype=torch.float32
                    )
            
            print(f"✅ Loaded {len(self.single_layer_vectors)} single-layer steering vectors")
            print(f"   Loaded layers: {list(self.single_layer_vectors.keys())}")
            
        except Exception as e:
            print(f"❌ Failed to load steering vectors: {e}")
            print("   Multi-layer steering will require vectors to be computed first")
    
    def compute_joint_steering_vectors(self, target_layers: List[int], 
                                     strategy: str = "averaged") -> Dict[int, torch.Tensor]:
        """
        Compute joint steering vectors across multiple layers.
        
        Args:
            target_layers: List of layer indices to combine
            strategy: Strategy for combining vectors ('averaged', 'weighted', 'cascaded')
            
        Returns:
            Dictionary of layer-specific steering vectors for joint application
        """
        print(f"🔧 Computing joint steering vectors for layers {target_layers}")
        print(f"   Strategy: {strategy}")
        
        if not self.single_layer_vectors:
            raise ValueError("No single-layer vectors loaded. Call load_single_layer_vectors() first.")
        
        # Filter to available layers
        available_layers = [layer for layer in target_layers if layer in self.single_layer_vectors]
        if not available_layers:
            raise ValueError(f"None of the target layers {target_layers} have steering vectors available")
        
        if len(available_layers) != len(target_layers):
            print(f"⚠️  Only {len(available_layers)}/{len(target_layers)} layers available: {available_layers}")
        
        joint_vectors = {}
        
        if strategy == "averaged":
            # Simple averaging of steering vectors
            for layer_idx in available_layers:
                # Average with vectors from other layers (weighted by distance)
                combined_vector = torch.zeros_like(self.single_layer_vectors[layer_idx])
                total_weight = 0.0
                
                for other_layer in available_layers:
                    # Weight by inverse distance + small constant
                    distance_weight = 1.0 / (1.0 + abs(layer_idx - other_layer))
                    combined_vector += distance_weight * self.single_layer_vectors[other_layer]
                    total_weight += distance_weight
                
                joint_vectors[layer_idx] = combined_vector / total_weight
                
        elif strategy == "weighted":
            # Weight by layer importance or position
            layer_weights = self._compute_layer_weights(available_layers)
            
            for layer_idx in available_layers:
                # Create weighted combination
                combined_vector = torch.zeros_like(self.single_layer_vectors[layer_idx])
                
                for other_layer in available_layers:
                    weight = layer_weights.get(other_layer, 1.0)
                    if other_layer == layer_idx:
                        weight *= 2.0  # Emphasize self-layer
                    combined_vector += weight * self.single_layer_vectors[other_layer]
                
                # Normalize
                combined_vector = combined_vector / len(available_layers)
                joint_vectors[layer_idx] = combined_vector
                
        elif strategy == "cascaded":
            # Apply steering in cascade from early to late layers
            sorted_layers = sorted(available_layers)
            
            for i, layer_idx in enumerate(sorted_layers):
                # Earlier layers get stronger intervention
                cascade_strength = 1.0 - (i / len(sorted_layers)) * 0.5
                cascaded_vector = cascade_strength * self.single_layer_vectors[layer_idx]
                
                # Add influence from earlier layers
                for j, earlier_layer in enumerate(sorted_layers[:i]):
                    influence_weight = 0.3 * (1.0 - j / max(i, 1))
                    cascaded_vector += influence_weight * self.single_layer_vectors[earlier_layer]
                
                joint_vectors[layer_idx] = cascaded_vector
        
        else:
            raise ValueError(f"Unknown joint steering strategy: {strategy}")
        
        # Store results
        self.multi_layer_vectors[strategy] = joint_vectors
        
        print(f"   ✅ Computed joint vectors for {len(joint_vectors)} layers")
        return joint_vectors
    
    def test_downstream_robustness(self, causal_layers: List[int], 
                                 test_offsets: List[int] = [1, 2, 3]) -> Dict[str, Any]:
        """
        Test FIRM requirement: what happens when steering is applied downstream
        from causally-identified layers?
        
        Args:
            causal_layers: Layers identified as causally important for bias
            test_offsets: Offsets to test downstream (1 = next layer, 2 = +2 layers, etc.)
            
        Returns:
            Downstream robustness test results
        """
        print(f"🧪 TESTING DOWNSTREAM ROBUSTNESS")
        print(f"   Causal layers: {causal_layers}")
        print(f"   Testing offsets: {test_offsets}")
        
        robustness_results = {
            "causal_layers": causal_layers,
            "test_offsets": test_offsets,
            "downstream_effectiveness": {},
            "robustness_scores": {},
            "optimal_downstream_offset": None
        }
        
        # Test steering at various downstream positions
        for offset in test_offsets:
            downstream_layers = [
                layer + offset for layer in causal_layers 
                if layer + offset < self.num_layers
            ]
            
            if not downstream_layers:
                continue
            
            print(f"   🔍 Testing downstream offset +{offset}: layers {downstream_layers}")
            
            # Measure effectiveness at downstream positions
            effectiveness_scores = []
            
            for downstream_layer in downstream_layers:
                if downstream_layer in self.single_layer_vectors:
                    # Get corresponding causal layer
                    causal_layer = downstream_layer - offset
                    if causal_layer in self.single_layer_vectors:
                        
                        # Compare vector effectiveness (simplified metric)
                        downstream_vector = self.single_layer_vectors[downstream_layer]
                        causal_vector = self.single_layer_vectors[causal_layer]
                        
                        # Measure similarity and magnitude preservation
                        similarity = F.cosine_similarity(
                            downstream_vector.unsqueeze(0),
                            causal_vector.unsqueeze(0)
                        ).item()
                        
                        magnitude_ratio = downstream_vector.norm().item() / (causal_vector.norm().item() + 1e-8)
                        
                        # Effectiveness combines similarity and magnitude preservation
                        effectiveness = similarity * min(magnitude_ratio, 1.0 / magnitude_ratio)
                        effectiveness_scores.append(effectiveness)
                        
                        print(f"     Layer {causal_layer} → {downstream_layer}: "
                              f"similarity={similarity:.3f}, mag_ratio={magnitude_ratio:.3f}, "
                              f"effectiveness={effectiveness:.3f}")
            
            if effectiveness_scores:
                avg_effectiveness = np.mean(effectiveness_scores)
                robustness_results["downstream_effectiveness"][f"offset_{offset}"] = {
                    "average_effectiveness": float(avg_effectiveness),
                    "layers_tested": downstream_layers,
                    "individual_scores": [float(s) for s in effectiveness_scores]
                }
                
                # Compute robustness score (higher = more robust to downstream application)
                baseline_effectiveness = 1.0  # Perfect effectiveness at causal layer
                robustness_score = avg_effectiveness / baseline_effectiveness
                robustness_results["robustness_scores"][f"offset_{offset}"] = float(robustness_score)
        
        # Find optimal downstream offset
        if robustness_results["robustness_scores"]:
            best_offset = max(
                robustness_results["robustness_scores"].items(),
                key=lambda x: x[1]
            )[0]
            robustness_results["optimal_downstream_offset"] = int(best_offset.split('_')[1])
        
        print(f"   ✅ Downstream robustness testing complete")
        print(f"   🎯 Optimal downstream offset: {robustness_results['optimal_downstream_offset']}")
        
        return robustness_results
    
    def test_unrelated_layer_steering(self, causal_layers: List[int], 
                                    num_random_layers: int = 3) -> Dict[str, Any]:
        """
        Test FIRM requirement: effectiveness of steering at layers unrelated
        to causally-identified bias circuits.
        
        Args:
            causal_layers: Causally important layers
            num_random_layers: Number of random unrelated layers to test
            
        Returns:
            Unrelated layer steering test results
        """
        print(f"🎲 TESTING UNRELATED LAYER STEERING")
        
        # Select random layers far from causal layers
        all_layers = list(range(self.num_layers))
        unrelated_candidates = []
        
        for layer in all_layers:
            # Consider layer "unrelated" if it's far from any causal layer
            min_distance = min(abs(layer - causal_layer) for causal_layer in causal_layers)
            if min_distance >= 3:  # At least 3 layers away
                unrelated_candidates.append(layer)
        
        # Select random unrelated layers
        np.random.seed(42)  # For reproducibility
        unrelated_layers = np.random.choice(
            unrelated_candidates, 
            size=min(num_random_layers, len(unrelated_candidates)),
            replace=False
        ).tolist()
        
        print(f"   Causal layers: {causal_layers}")
        print(f"   Unrelated layers to test: {unrelated_layers}")
        
        unrelated_results = {
            "causal_layers": causal_layers,
            "unrelated_layers": unrelated_layers,
            "effectiveness_comparison": {},
            "interference_analysis": {},
            "isolation_scores": {}
        }
        
        # Compare effectiveness between causal and unrelated layers
        for unrelated_layer in unrelated_layers:
            if unrelated_layer not in self.single_layer_vectors:
                continue
            
            unrelated_vector = self.single_layer_vectors[unrelated_layer]
            
            # Compare with causal layers
            causal_similarities = []
            causal_magnitudes = []
            
            for causal_layer in causal_layers:
                if causal_layer in self.single_layer_vectors:
                    causal_vector = self.single_layer_vectors[causal_layer]
                    
                    similarity = F.cosine_similarity(
                        unrelated_vector.unsqueeze(0),
                        causal_vector.unsqueeze(0)
                    ).item()
                    
                    mag_ratio = unrelated_vector.norm().item() / (causal_vector.norm().item() + 1e-8)
                    
                    causal_similarities.append(similarity)
                    causal_magnitudes.append(mag_ratio)
            
            if causal_similarities:
                avg_similarity = np.mean(causal_similarities)
                avg_mag_ratio = np.mean(causal_magnitudes)
                
                # Isolation score: lower similarity = better isolation
                isolation_score = 1.0 - abs(avg_similarity)
                
                unrelated_results["effectiveness_comparison"][f"layer_{unrelated_layer}"] = {
                    "similarity_to_causal": float(avg_similarity),
                    "magnitude_ratio": float(avg_mag_ratio),
                    "isolation_score": float(isolation_score)
                }
                
                print(f"   Layer {unrelated_layer}: similarity={avg_similarity:.3f}, "
                      f"isolation={isolation_score:.3f}")
        
        # Analyze overall isolation
        if unrelated_results["effectiveness_comparison"]:
            isolation_scores = [
                data["isolation_score"] 
                for data in unrelated_results["effectiveness_comparison"].values()
            ]
            
            unrelated_results["isolation_analysis"] = {
                "average_isolation": float(np.mean(isolation_scores)),
                "min_isolation": float(np.min(isolation_scores)),
                "max_isolation": float(np.max(isolation_scores)),
                "isolation_consistency": float(1.0 - np.std(isolation_scores))
            }
            
            print(f"   ✅ Average isolation score: {unrelated_results['isolation_analysis']['average_isolation']:.3f}")
        
        return unrelated_results
    
    def apply_multi_layer_intervention(self, input_text: str, 
                                     layer_strategy: str = "joint_averaged",
                                     intervention_strength: float = 1.0) -> str:
        """
        Apply multi-layer steering intervention during text generation.
        
        Args:
            input_text: Input text to process
            layer_strategy: Strategy for multi-layer intervention
            intervention_strength: Strength of intervention
            
        Returns:
            Generated text with multi-layer intervention
        """
        print(f"🎯 Applying multi-layer intervention: {layer_strategy}")
        
        # Select steering vectors based on strategy
        if layer_strategy == "joint_averaged" and "averaged" in self.multi_layer_vectors:
            steering_vectors = self.multi_layer_vectors["averaged"]
        elif layer_strategy == "joint_weighted" and "weighted" in self.multi_layer_vectors:
            steering_vectors = self.multi_layer_vectors["weighted"]
        elif layer_strategy == "joint_cascaded" and "cascaded" in self.multi_layer_vectors:
            steering_vectors = self.multi_layer_vectors["cascaded"]
        else:
            # Fallback to single-layer steering
            if not self.single_layer_vectors:
                raise ValueError("No steering vectors available for intervention")
            
            # Use first few available layers
            available_layers = sorted(list(self.single_layer_vectors.keys()))[:self.max_concurrent_layers]
            steering_vectors = {layer: self.single_layer_vectors[layer] for layer in available_layers}
        
        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        # Set up hooks for multi-layer intervention
        hook_handles = []
        
        def create_steering_hook(layer_idx: int, steering_vector: torch.Tensor):
            def hook_fn(module, input, output):
                # Apply steering to the output
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                # Apply intervention to last token position
                if hidden_states.dim() == 3:  # [batch, seq, hidden]
                    hidden_states[:, -1, :] += intervention_strength * steering_vector
                
                return output
            return hook_fn
        
        # Register hooks for each target layer
        for layer_idx, steering_vector in steering_vectors.items():
            if layer_idx < self.num_layers:
                layer_module = self.model.model.layers[layer_idx]
                hook_handle = layer_module.register_forward_hook(
                    create_steering_hook(layer_idx, steering_vector)
                )
                hook_handles.append(hook_handle)
        
        try:
            # Generate with multi-layer intervention
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )
        
        finally:
            # Remove all hooks
            for hook_handle in hook_handles:
                hook_handle.remove()
        
        print(f"   ✅ Multi-layer intervention complete")
        return generated_text
    
    def _compute_layer_weights(self, layers: List[int]) -> Dict[int, float]:
        """Compute importance weights for layers based on position and other factors."""
        layer_weights = {}
        
        # Simple position-based weighting (middle layers get higher weight)
        for layer_idx in layers:
            position_factor = 1.0 - abs(layer_idx - self.num_layers / 2) / (self.num_layers / 2)
            layer_weights[layer_idx] = max(0.1, position_factor)
        
        return layer_weights
    
    def save_multi_layer_results(self, output_dir: str, 
                               downstream_results: Dict[str, Any] = None,
                               unrelated_results: Dict[str, Any] = None) -> None:
        """Save multi-layer intervention results."""
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            "multi_layer_vectors": {},
            "downstream_robustness": downstream_results or {},
            "unrelated_layer_analysis": unrelated_results or {},
            "configuration": {
                "max_concurrent_layers": self.max_concurrent_layers,
                "layer_interaction_strength": self.layer_interaction_strength,
                "available_strategies": list(self.multi_layer_vectors.keys())
            }
        }
        
        # Convert tensor vectors to numpy for serialization
        for strategy, vectors in self.multi_layer_vectors.items():
            results["multi_layer_vectors"][strategy] = {
                str(layer_idx): vector.detach().cpu().numpy().tolist()
                for layer_idx, vector in vectors.items()
            }
        
        # Save results
        results_path = os.path.join(output_dir, "multi_layer_intervention_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Saved multi-layer intervention results to: {results_path}")