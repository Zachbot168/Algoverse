#!/usr/bin/env python3
"""
Debiasing Steering Vector (DSV) Computation

Computes steering vectors for different types of bias using contrastive techniques.
These vectors are used by the DAS wrapper for inference-time debiasing.

Based on the Fairsteer methodology but integrated with the unified pipeline.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from train.component_registry import ComponentRegistryManager

warnings.filterwarnings('ignore')


class DSVComputer:
    """
    Computes Debiasing Steering Vectors using contrastive activation analysis.
    """
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                 config: Dict[str, Any]):
        """
        Initialize DSV computer.
        
        Args:
            model: HuggingFace model to analyze
            tokenizer: Associated tokenizer
            config: Configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device
        
        # DSV configuration
        dsv_config = config.get('interventions', {}).get('steering', {}).get('dsv', {})
        self.bias_categories = dsv_config.get('bias_categories', ['gender', 'race', 'religion', 'sycophancy'])
        self.magnitude_scale = dsv_config.get('magnitude_scale', 1.0)
        
        # Model info
        self.num_layers = len(model.model.layers)
        self.hidden_size = model.config.hidden_size
        
        # Registry manager for optimal layer selection
        self.registry_manager = ComponentRegistryManager()
        self.optimal_layers = []
        
        print(f"Initialized DSV computer for {self.num_layers} layers")
        print(f"Bias categories: {self.bias_categories}")
    
    def load_component_registry(self, registry_path: str) -> None:
        """Load component registry to identify optimal steering layers."""
        if os.path.exists(registry_path):
            registry_dir = os.path.dirname(registry_path)
            registry_file = os.path.basename(registry_path)
            
            self.registry_manager.registry_dir = Path(registry_dir)
            self.registry_manager.load_registry(registry_file)
            
            # Get steering layers from registry
            self.optimal_layers = self.registry_manager.get_steering_layers()
            print(f"Loaded registry, optimal steering layers: {self.optimal_layers}")
        else:
            print(f"Registry not found at {registry_path}, using default layers")
            # Default to middle layers
            self.optimal_layers = list(range(self.num_layers // 2, self.num_layers * 3 // 4))
    
    def create_contrastive_dataset(self, bias_category: str, num_pairs: int = 1000) -> List[Tuple[str, str]]:
        """
        Create contrastive prompt pairs for the specified bias category.
        
        Args:
            bias_category: Type of bias ('gender', 'race', 'religion', 'sycophancy')
            num_pairs: Number of contrastive pairs to generate
            
        Returns:
            List of (biased_prompt, unbiased_prompt) pairs
        """
        print(f"Creating contrastive dataset for {bias_category}...")
        
        if bias_category == "gender":
            return self._create_gender_pairs(num_pairs)
        elif bias_category == "race":
            return self._create_race_pairs(num_pairs)
        elif bias_category == "religion":
            return self._create_religion_pairs(num_pairs)
        elif bias_category == "sycophancy":
            return self._create_sycophancy_pairs(num_pairs)
        else:
            # General bias pairs
            return self._create_general_pairs(num_pairs)
    
    def _create_gender_pairs(self, num_pairs: int) -> List[Tuple[str, str]]:
        """Create gender bias contrastive pairs."""
        pairs = []
        
        # Templates for gender bias
        templates = [
            ("He is excellent at {}", "They are excellent at {}"),
            ("She should focus on {}", "They should focus on {}"),
            ("Men are naturally good at {}", "People are naturally good at {}"),
            ("Women are better suited for {}", "People are better suited for {}"),
            ("His career in {} is promising", "Their career in {} is promising"),
            ("Her role in {} is important", "Their role in {} is important")
        ]
        
        contexts = [
            "engineering", "nursing", "leadership", "caregiving", "science", "education",
            "technology", "management", "research", "teaching", "innovation", "collaboration"
        ]
        
        for i in range(min(num_pairs, len(templates) * len(contexts))):
            template_idx = i % len(templates)
            context_idx = (i // len(templates)) % len(contexts)
            
            biased_template, unbiased_template = templates[template_idx]
            context = contexts[context_idx]
            
            biased_prompt = biased_template.format(context)
            unbiased_prompt = unbiased_template.format(context)
            
            pairs.append((biased_prompt, unbiased_prompt))
        
        return pairs[:num_pairs]
    
    def _create_race_pairs(self, num_pairs: int) -> List[Tuple[str, str]]:
        """Create racial bias contrastive pairs."""
        pairs = []
        
        templates = [
            ("People from {} are typically {}", "People are typically {}"),
            ("Someone with a {} background might {}", "Someone might {}"),
            ("In {} culture, they often {}", "In many cultures, people often {}"),
            ("A person of {} descent would {}", "A person would {}"),
        ]
        
        contexts = [
            ("Asian", "good at math"),
            ("European", "value punctuality"),
            ("African", "have strong communities"),
            ("Latin American", "be family-oriented"),
        ]
        
        general_contexts = [
            "be skilled professionals",
            "contribute to society",
            "have diverse talents",
            "work hard for their goals"
        ]
        
        for i in range(min(num_pairs, len(templates) * len(contexts))):
            template_idx = i % len(templates)
            context_idx = (i // len(templates)) % len(contexts)
            
            biased_template, unbiased_template = templates[template_idx]
            specific_context, trait = contexts[context_idx]
            general_trait = general_contexts[context_idx % len(general_contexts)]
            
            if "{}" in biased_template:
                if biased_template.count('{}') == 2:
                    biased_prompt = biased_template.format(specific_context, trait)
                    unbiased_prompt = unbiased_template.format(general_trait)
                else:
                    biased_prompt = biased_template.format(specific_context)
                    unbiased_prompt = unbiased_template.format("diverse backgrounds")
            
            pairs.append((biased_prompt, unbiased_prompt))
        
        return pairs[:num_pairs]
    
    def _create_religion_pairs(self, num_pairs: int) -> List[Tuple[str, str]]:
        """Create religious bias contrastive pairs."""
        pairs = []
        
        templates = [
            ("{} people usually {}", "People usually {}"),
            ("Someone who is {} might {}", "Someone might {}"),
            ("In {} tradition, they {}", "In many traditions, people {}"),
            ("A {} person would {}", "A person would {}"),
        ]
        
        contexts = [
            ("Christian", "go to church on Sundays"),
            ("Muslim", "pray five times a day"),
            ("Jewish", "observe the Sabbath"),
            ("Hindu", "practice meditation"),
            ("Buddhist", "seek inner peace"),
            ("Atheist", "rely on science")
        ]
        
        general_contexts = [
            "have their own practices",
            "follow their beliefs",
            "value their traditions",
            "live according to their values"
        ]
        
        for i in range(min(num_pairs, len(templates) * len(contexts))):
            template_idx = i % len(templates)
            context_idx = (i // len(templates)) % len(contexts)
            
            biased_template, unbiased_template = templates[template_idx]
            religion, practice = contexts[context_idx]
            general_practice = general_contexts[context_idx % len(general_contexts)]
            
            if biased_template.count('{}') == 2:
                biased_prompt = biased_template.format(religion, practice)
                unbiased_prompt = unbiased_template.format(general_practice)
            else:
                biased_prompt = biased_template.format(religion)
                unbiased_prompt = unbiased_template.format("various backgrounds")
            
            pairs.append((biased_prompt, unbiased_prompt))
        
        return pairs[:num_pairs]
    
    def _create_sycophancy_pairs(self, num_pairs: int) -> List[Tuple[str, str]]:
        """Create sycophancy contrastive pairs."""
        pairs = []
        
        # Sycophantic vs truthful responses
        templates = [
            ("You're absolutely right about {}", "I think {} is a complex topic that deserves careful consideration"),
            ("I completely agree with your view on {}", "There are different perspectives on {}"),
            ("Your opinion on {} is perfect", "Your view on {} is one valid perspective among many"),
            ("I couldn't agree more about {}", "That's an interesting point about {}. Let me share some additional context"),
            ("You're so smart to think {}", "That's a thoughtful perspective on {}"),
            ("Exactly my thoughts on {}", "That's one way to look at {}")
        ]
        
        topics = [
            "climate change",
            "economic policy",
            "social media",
            "artificial intelligence",
            "education reform",
            "healthcare systems",
            "political systems",
            "technology adoption",
            "environmental protection",
            "social justice"
        ]
        
        for i in range(min(num_pairs, len(templates) * len(topics))):
            template_idx = i % len(templates)
            topic_idx = (i // len(templates)) % len(topics)
            
            sycophantic_template, truthful_template = templates[template_idx]
            topic = topics[topic_idx]
            
            biased_prompt = sycophantic_template.format(topic)
            unbiased_prompt = truthful_template.format(topic)
            
            pairs.append((biased_prompt, unbiased_prompt))
        
        return pairs[:num_pairs]
    
    def _create_general_pairs(self, num_pairs: int) -> List[Tuple[str, str]]:
        """Create general bias contrastive pairs."""
        pairs = []
        
        # Mix of different bias types
        gender_pairs = self._create_gender_pairs(num_pairs // 4)
        race_pairs = self._create_race_pairs(num_pairs // 4)
        religion_pairs = self._create_religion_pairs(num_pairs // 4)  
        sycophancy_pairs = self._create_sycophancy_pairs(num_pairs // 4)
        
        pairs.extend(gender_pairs)
        pairs.extend(race_pairs)
        pairs.extend(religion_pairs)
        pairs.extend(sycophancy_pairs)
        
        return pairs[:num_pairs]
    
    @torch.no_grad()
    def extract_contrastive_activations(self, prompt_pairs: List[Tuple[str, str]], 
                                      layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract activations from contrastive prompt pairs.
        
        Args:
            prompt_pairs: List of (biased, unbiased) prompt pairs
            layer_idx: Layer index to extract activations from
            
        Returns:
            Tuple of (biased_activations, unbiased_activations)
        """
        print(f"Extracting activations from layer {layer_idx}...")
        
        biased_activations = []
        unbiased_activations = []
        
        # Hook to capture activations
        activation_cache = {}
        
        def activation_hook(module, input, output):
            activation_cache['activation'] = output[0] if isinstance(output, tuple) else output
        
        # Register hook on target layer
        layer = self.model.model.layers[layer_idx]
        hook = layer.register_forward_hook(activation_hook)
        
        try:
            batch_size = 8  # Process in small batches
            
            for i in tqdm(range(0, len(prompt_pairs), batch_size), desc="Processing pairs"):
                batch_pairs = prompt_pairs[i:i+batch_size]
                
                # Process biased prompts
                biased_prompts = [pair[0] for pair in batch_pairs]
                biased_inputs = self.tokenizer(
                    biased_prompts, 
                    padding=True, 
                    truncation=True, 
                    max_length=128,
                    return_tensors="pt"
                ).to(self.device)
                
                # Forward pass for biased prompts
                self.model(**biased_inputs)
                biased_batch_activations = activation_cache['activation'].detach().cpu()
                
                # Use last token activations
                for j, activation in enumerate(biased_batch_activations):
                    seq_len = biased_inputs['attention_mask'][j].sum().item()
                    last_token_activation = activation[seq_len - 1]  # Last non-padding token
                    biased_activations.append(last_token_activation)
                
                # Process unbiased prompts
                unbiased_prompts = [pair[1] for pair in batch_pairs]
                unbiased_inputs = self.tokenizer(
                    unbiased_prompts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(self.device)
                
                # Forward pass for unbiased prompts
                self.model(**unbiased_inputs)
                unbiased_batch_activations = activation_cache['activation'].detach().cpu()
                
                # Use last token activations
                for j, activation in enumerate(unbiased_batch_activations):
                    seq_len = unbiased_inputs['attention_mask'][j].sum().item()
                    last_token_activation = activation[seq_len - 1]
                    unbiased_activations.append(last_token_activation)
                
                # Clear GPU memory
                torch.cuda.empty_cache()
        
        finally:
            # Remove hook
            hook.remove()
        
        # Stack activations
        biased_activations = torch.stack(biased_activations)
        unbiased_activations = torch.stack(unbiased_activations)
        
        return biased_activations, unbiased_activations
    
    def compute_steering_vector(self, biased_activations: torch.Tensor, 
                              unbiased_activations: torch.Tensor) -> torch.Tensor:
        """
        Compute steering vector from contrastive activations.
        
        Args:
            biased_activations: Activations from biased prompts
            unbiased_activations: Activations from unbiased prompts
            
        Returns:
            Steering vector pointing from biased to unbiased direction
        """
        # Compute mean difference (unbiased - biased)
        biased_mean = biased_activations.mean(dim=0)
        unbiased_mean = unbiased_activations.mean(dim=0)
        
        steering_vector = unbiased_mean - biased_mean
        
        # Normalize the steering vector
        steering_vector = F.normalize(steering_vector, dim=0)
        
        # Apply magnitude scaling
        steering_vector *= self.magnitude_scale
        
        return steering_vector
    
    def compute_dsv_for_category(self, bias_category: str, 
                               num_pairs: int = 1000) -> Dict[int, torch.Tensor]:
        """
        Compute DSV for a specific bias category across multiple layers.
        
        Args:
            bias_category: Type of bias to compute DSV for
            num_pairs: Number of contrastive pairs to use
            
        Returns:
            Dictionary mapping layer indices to steering vectors
        """
        print(f"\nComputing DSV for {bias_category}...")
        
        # Create contrastive dataset
        prompt_pairs = self.create_contrastive_dataset(bias_category, num_pairs)
        print(f"Created {len(prompt_pairs)} contrastive pairs")
        
        # Compute steering vectors for each optimal layer
        steering_vectors = {}
        
        layers_to_process = self.optimal_layers if self.optimal_layers else [self.num_layers // 2]
        
        for layer_idx in layers_to_process:
            print(f"Processing layer {layer_idx}...")
            
            # Extract contrastive activations
            biased_acts, unbiased_acts = self.extract_contrastive_activations(
                prompt_pairs, layer_idx
            )
            
            # Compute steering vector
            steering_vector = self.compute_steering_vector(biased_acts, unbiased_acts)
            steering_vectors[layer_idx] = steering_vector
            
            print(f"Layer {layer_idx}: Steering vector norm = {steering_vector.norm().item():.4f}")
        
        return steering_vectors
    
    def compute_all_dsv(self, num_pairs_per_category: int = 1000) -> Dict[str, torch.Tensor]:
        """
        Compute DSV for all bias categories.
        
        Args:
            num_pairs_per_category: Number of pairs per bias category
            
        Returns:
            Dictionary mapping bias categories to steering vectors
        """
        print("Computing DSV for all bias categories...")
        
        all_steering_vectors = {}
        
        for bias_category in self.bias_categories:
            category_vectors = self.compute_dsv_for_category(
                bias_category, num_pairs_per_category
            )
            
            # Use the best layer's steering vector for this category
            if category_vectors:
                # Select vector from optimal layer (or first available)
                optimal_layer = self.registry_manager.get_optimal_steering_layer()
                if optimal_layer in category_vectors:
                    best_vector = category_vectors[optimal_layer]
                else:
                    best_vector = list(category_vectors.values())[0]
                
                all_steering_vectors[bias_category] = best_vector
                print(f"Selected steering vector for {bias_category} from layer {list(category_vectors.keys())[0]}")
        
        # Compute general steering vector (average of all categories)
        if all_steering_vectors:
            general_vector = torch.stack(list(all_steering_vectors.values())).mean(dim=0)
            all_steering_vectors['general'] = general_vector
            print("Computed general steering vector")
        
        return all_steering_vectors
    
    def save_steering_vectors(self, steering_vectors: Dict[str, torch.Tensor], 
                            output_path: str) -> None:
        """Save computed steering vectors."""
        # Convert to numpy for serialization
        numpy_vectors = {}
        for category, vector in steering_vectors.items():
            numpy_vectors[category] = vector.detach().cpu().numpy()
        
        with open(output_path, 'wb') as f:
            pickle.dump(numpy_vectors, f)
        
        print(f"Saved {len(steering_vectors)} steering vectors to: {output_path}")
        
        # Save metadata
        metadata = {
            "bias_categories": list(steering_vectors.keys()),
            "vector_dimensions": steering_vectors[list(steering_vectors.keys())[0]].shape[0],
            "magnitude_scale": self.magnitude_scale,
            "model_name": getattr(self.model, 'name_or_path', 'unknown'),
            "optimal_layers": self.optimal_layers
        }
        
        metadata_path = output_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved metadata to: {metadata_path}")


def main():
    """Main entry point for DSV computation."""
    parser = argparse.ArgumentParser(description="Compute Debiasing Steering Vectors")
    parser.add_argument("--config", required=True, help="Configuration file path")
    parser.add_argument("--registry", help="Component registry path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--num_pairs", type=int, default=1000, help="Number of contrastive pairs per category")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model and tokenizer
    model_name = config['model']['name']
    print(f"Loading model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, config['model'].get('torch_dtype', 'float16')),
        device_map="auto",
        trust_remote_code=config['model'].get('trust_remote_code', False)
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize DSV computer
    dsv_computer = DSVComputer(model, tokenizer, config)
    
    # Load component registry if provided
    if args.registry:
        dsv_computer.load_component_registry(args.registry)
    
    # Compute steering vectors
    steering_vectors = dsv_computer.compute_all_dsv(args.num_pairs)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "steering_vectors.pkl")
    dsv_computer.save_steering_vectors(steering_vectors, output_path)
    
    print(f"\nDSV computation completed!")
    print(f"Computed steering vectors for: {list(steering_vectors.keys())}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()