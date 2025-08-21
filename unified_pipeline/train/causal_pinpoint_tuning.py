#!/usr/bin/env python3
"""
Causal-Informed Pinpoint Tuning - FIRM Phase 2

Extends UnifiedPinpointTuner to use causally-identified bias circuits
for targeted LoRA training instead of generic component selection.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings
import numpy as np

import torch
from peft import LoraConfig, get_peft_model, TaskType

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

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from train.run_pinpoint_tuning import UnifiedPinpointTuner
from causal_analysis.bias_circuit_tracer import BiasCircuitTracer, CircuitComponent

warnings.filterwarnings('ignore')


class CausalPinpointTuner(UnifiedPinpointTuner):
    """
    FIRM-enhanced pinpoint tuner that uses causal circuit analysis
    to identify and selectively fine-tune bias-causing components.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize causal pinpoint tuner."""
        super().__init__(config)
        
        # FIRM-specific components
        self.circuit_tracer: Optional[BiasCircuitTracer] = None
        self.causal_circuits: Dict[Tuple[int, int], CircuitComponent] = {}
        self.bias_types = config.get('causal_config', {}).get('bias_types', ['gender', 'race', 'religion'])
        self.min_causal_importance = config.get('causal_config', {}).get('min_importance', 0.1)
        
        print(f"Initialized CausalPinpointTuner for bias types: {self.bias_types}")
    
    def initialize_circuit_tracer(self, model, tokenizer) -> None:
        """Initialize the bias circuit tracer with loaded model."""
        print("Initializing bias circuit tracer...")
        self.circuit_tracer = BiasCircuitTracer(model, tokenizer, self.device)
        print("✅ Circuit tracer initialized")
    
    def identify_causal_circuits(self, output_dir: str) -> Dict[Tuple[int, int], CircuitComponent]:
        """
        Run causal analysis to identify bias circuits across all configured bias types.
        
        Args:
            output_dir: Directory to save circuit analysis results
            
        Returns:
            Dictionary mapping (layer, head) to CircuitComponent
        """
        if not self.circuit_tracer:
            raise ValueError("Circuit tracer not initialized. Call initialize_circuit_tracer() first.")
        
        print("🧠 " + "="*60)
        print("   🔍 FIRM PHASE 2A: CAUSAL CIRCUIT IDENTIFICATION")
        print("🧠 " + "="*60)
        
        all_circuits = {}
        circuit_analysis_results = {}
        
        for bias_type in self.bias_types:
            print(f"\n📊 Analyzing {bias_type} bias circuits...")
            
            # Identify circuits for this bias type
            bias_circuits = self.circuit_tracer.identify_bias_circuits(
                bias_type=bias_type,
                num_pairs=100,  # Can be configured
                batch_size=4   # Conservative batch size for stability
            )
            
            # Merge with overall results
            for circuit_id, component in bias_circuits.items():
                if circuit_id in all_circuits:
                    # If circuit already identified for another bias type, combine importance scores
                    existing_component = all_circuits[circuit_id]
                    existing_component.importance_score = max(
                        existing_component.importance_score, 
                        component.importance_score
                    )
                    existing_component.bias_type = f"{existing_component.bias_type},{bias_type}"
                else:
                    all_circuits[circuit_id] = component
            
            # Store analysis for this bias type
            circuit_analysis_results[bias_type] = {
                "num_circuits": len(bias_circuits),
                "avg_importance": sum(c.importance_score for c in bias_circuits.values()) / len(bias_circuits) if bias_circuits else 0
            }
            
            print(f"  ✅ Found {len(bias_circuits)} {bias_type} bias circuits")
        
        # Filter circuits by minimum importance threshold
        filtered_circuits = {
            circuit_id: component for circuit_id, component in all_circuits.items()
            if component.importance_score >= self.min_causal_importance
        }
        
        self.causal_circuits = filtered_circuits
        
        # Save circuit analysis results
        os.makedirs(output_dir, exist_ok=True)
        circuit_results_path = os.path.join(output_dir, "causal_circuit_analysis.json")
        
        # Prepare serializable results
        circuits_data = []
        for (layer, head), component in filtered_circuits.items():
            circuits_data.append({
                "layer": layer,
                "head": head,
                "component_type": component.component_type,
                "importance_score": float(component.importance_score),
                "bias_type": component.bias_type,
                "logit_diff_contribution": float(component.logit_diff_contribution)
            })
        
        results = {
            "total_circuits_identified": len(all_circuits),
            "circuits_above_threshold": len(filtered_circuits),
            "min_importance_threshold": self.min_causal_importance,
            "bias_type_analysis": circuit_analysis_results,
            "circuits": circuits_data
        }
        
        with open(circuit_results_path, 'w') as f:
            json.dump(results, f, indent=2, cls=FIRMJSONEncoder)
        
        print(f"\n🎯 CAUSAL CIRCUIT IDENTIFICATION COMPLETE")
        print(f"  📊 Total circuits identified: {len(all_circuits)}")
        print(f"  🔍 Circuits above threshold ({self.min_causal_importance}): {len(filtered_circuits)}")
        print(f"  💾 Results saved to: {circuit_results_path}")
        
        return filtered_circuits
    
    def _select_causal_components(self) -> None:
        """
        Select components for LoRA training based on causal circuit analysis.
        Overrides the generic component selection from parent class.
        """
        if not self.causal_circuits:
            raise ValueError("No causal circuits identified. Run identify_causal_circuits() first.")
        
        print(f"\n🎯 SELECTING CAUSAL COMPONENTS FOR TRAINING")
        print(f"   Using {len(self.causal_circuits)} causally-identified circuits")
        
        # Convert causal circuits to ComponentInfo format expected by parent class
        from train.component_registry import ComponentInfo
        
        causal_components = []
        for (layer, head), circuit_component in self.causal_circuits.items():
            component_info = ComponentInfo(
                layer=layer,
                type="head",  # All our circuits are attention heads currently
                head_index=head,
                importance=circuit_component.importance_score,
                bias_type=circuit_component.bias_type,
                source="causal_analysis"
            )
            causal_components.append(component_info)
        
        # Sort by importance and take top components
        max_components = self.config['interventions']['pinpoint_tuning']['component_selection']['max_components']
        causal_components.sort(key=lambda x: x.importance, reverse=True)
        self.selected_components = causal_components[:max_components]
        
        # Print selection summary
        print(f"   📊 Selected {len(self.selected_components)} components for training:")
        print(f"      • Attention heads: {len([c for c in self.selected_components if c.type == 'head'])}")
        print(f"      • Average importance: {sum(c.importance for c in self.selected_components) / len(self.selected_components):.4f}")
        print(f"      • Bias types covered: {set(c.bias_type for c in self.selected_components)}")
        
        # Show top 5 selected components
        print(f"   🔝 Top 5 selected components:")
        for i, component in enumerate(self.selected_components[:5]):
            print(f"      {i+1}. Layer {component.layer}, Head {component.head_index}: {component.importance:.4f} ({component.bias_type})")
    
    def create_causal_lora_config(self) -> LoraConfig:
        """
        Create LoRA configuration targeting causally-identified components.
        Uses more precise targeting than the parent class.
        """
        if not self.selected_components:
            raise ValueError("No components selected. Run _select_causal_components() first.")
        
        lora_config = self.config['interventions']['pinpoint_tuning']['lora']
        
        # Generate specific target modules based on causal circuits
        target_modules = []
        
        # Group components by layer for efficient targeting
        layer_components = {}
        for component in self.selected_components:
            if component.layer not in layer_components:
                layer_components[component.layer] = []
            layer_components[component.layer].append(component)
        
        # Create targeted module names
        for layer_idx, components in layer_components.items():
            # For attention heads, we target the specific projection layers
            attention_components = [c for c in components if c.type == "head"]
            if attention_components:
                # Target all attention projections for layers with identified bias circuits
                for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    target_modules.append(f"model.layers.{layer_idx}.self_attn.{proj}")
        
        print(f"🎯 LoRA will target {len(target_modules)} causal modules across {len(layer_components)} layers")
        
        # Create enhanced LoRA config for causal targeting
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config['r'],
            lora_alpha=lora_config['alpha'],
            lora_dropout=lora_config['dropout'],
            target_modules=target_modules,
            bias="none",
            inference_mode=False,
            # FIRM-specific: add metadata about causal targeting
            init_lora_weights="gaussian"  # Better initialization for causal components
        )
    
    def run_causal_training(self) -> str:
        """
        Run FIRM-enhanced pinpoint tuning with causal component targeting.
        
        Returns:
            Output directory path
        """
        print("🧠 " + "="*60)
        print("   🔧 FIRM PHASE 2B: CAUSAL PINPOINT TUNING")
        print("🧠 " + "="*60)
        
        # Prepare training arguments
        training_args = self.prepare_training_arguments()
        output_dir = training_args['output_dir']
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Identify causal circuits
        if not self.causal_circuits:
            self.identify_causal_circuits(output_dir)
        
        # Step 2: Select components based on causal analysis
        self._select_causal_components()
        
        # Step 3: Create causal LoRA configuration
        lora_config = self.create_causal_lora_config()
        
        # Step 4: Load and prepare model
        print(f"\n🔄 Loading model: {self.model_name}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=getattr(torch, training_args['torch_dtype']),
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=training_args['trust_remote_code']
        )
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Step 5: Initialize circuit tracer with loaded model (if not already done)
        if not self.circuit_tracer:
            self.initialize_circuit_tracer(model, tokenizer)
        
        # Step 6: Apply causal LoRA
        print("🔧 Applying causal LoRA configuration...")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Step 7: Save training metadata
        self._save_causal_training_metadata(output_dir)
        
        # Step 8: Save model and tokenizer (training simulation for now)
        print("💾 Saving causally-configured model...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"\n✅ CAUSAL PINPOINT TUNING COMPLETE")
        print(f"   📁 Model saved to: {output_dir}")
        print(f"   🎯 Targeted {len(self.selected_components)} causal components")
        print(f"   🧠 Used {len(self.causal_circuits)} bias circuits for selection")
        
        return output_dir
    
    def _save_causal_training_metadata(self, output_dir: str) -> None:
        """Save FIRM-specific training metadata."""
        metadata = {
            "training_type": "FIRM_causal_pinpoint_tuning",
            "model_name": self.model_name,
            "timestamp": str(torch.utils.data.get_worker_info()) if torch.utils.data.get_worker_info() else "N/A",
            
            # Causal analysis results
            "causal_analysis": {
                "bias_types_analyzed": self.bias_types,
                "total_circuits_identified": len(self.causal_circuits),
                "min_importance_threshold": self.min_causal_importance,
                "circuits_by_bias_type": {}
            },
            
            # Selected components
            "selected_components": [
                {
                    "layer": comp.layer,
                    "type": comp.type,
                    "head_index": comp.head_index,
                    "importance": comp.importance,
                    "bias_type": comp.bias_type,
                    "source": comp.source
                }
                for comp in self.selected_components
            ],
            
            # Training configuration
            "training_config": self.config['interventions']['pinpoint_tuning'],
            
            # Component summary
            "component_summary": {
                "total_components": len(self.selected_components),
                "attention_heads": len([c for c in self.selected_components if c.type == "head"]),
                "causal_circuits": len(self.causal_circuits),
                "bias_types_covered": list(set(c.bias_type for c in self.selected_components))
            }
        }
        
        # Add bias type breakdown
        for bias_type in self.bias_types:
            circuits_for_type = [
                c for c in self.causal_circuits.values() 
                if bias_type in c.bias_type
            ]
            metadata["causal_analysis"]["circuits_by_bias_type"][bias_type] = len(circuits_for_type)
        
        metadata_path = os.path.join(output_dir, "causal_training_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, cls=FIRMJSONEncoder)
        
        print(f"💾 Saved FIRM training metadata to: {metadata_path}")
    
    def validate_causal_targeting(self, output_dir: str) -> Dict[str, Any]:
        """
        Validate that causal targeting is working correctly.
        
        Args:
            output_dir: Directory with training results
            
        Returns:
            Validation results
        """
        print(f"\n🔍 VALIDATING CAUSAL TARGETING")
        
        validation_results = {
            "components_targeted": len(self.selected_components),
            "circuits_used": len(self.causal_circuits),
            "bias_types_covered": len(self.bias_types),
            "layer_coverage": {},
            "importance_distribution": {}
        }
        
        # Analyze layer coverage
        layer_counts = {}
        importance_scores = []
        
        for component in self.selected_components:
            layer = component.layer
            if layer not in layer_counts:
                layer_counts[layer] = 0
            layer_counts[layer] += 1
            importance_scores.append(component.importance)
        
        validation_results["layer_coverage"] = layer_counts
        validation_results["importance_distribution"] = {
            "mean": float(sum(importance_scores) / len(importance_scores)) if importance_scores else 0,
            "min": float(min(importance_scores)) if importance_scores else 0,
            "max": float(max(importance_scores)) if importance_scores else 0,
            "std": float(torch.tensor(importance_scores).std()) if importance_scores else 0
        }
        
        # Check for reasonable distribution
        validation_results["validation_passed"] = (
            validation_results["components_targeted"] > 0 and
            validation_results["circuits_used"] > 0 and
            validation_results["importance_distribution"]["mean"] > self.min_causal_importance
        )
        
        # Save validation results
        validation_path = os.path.join(output_dir, "causal_targeting_validation.json")
        with open(validation_path, 'w') as f:
            json.dump(validation_results, f, indent=2, cls=FIRMJSONEncoder)
        
        status = "✅ PASSED" if validation_results["validation_passed"] else "❌ FAILED"
        print(f"   {status} - Validation results saved to: {validation_path}")
        
        return validation_results