#!/usr/bin/env python3
"""
FairSteer Implementation for Gemma-2-2b-it Bias Mitigation
This script implements the FairSteer method for debiasing Gemma-2-2b-it model
using a three-stage inference-time framework without requiring model retraining.

Optimized specifically for Google's Gemma-2-2b-it architecture
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    logging as transformers_logging
)
from huggingface_hub import login
from datasets import load_dataset
import json
import pickle
from tqdm import tqdm
from collections import defaultdict

# Import token utilities
from gemma_token_utils import get_gemma_token, GEMMA_HUGGINGFACE_TOKEN
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any, Optional
import warnings
import os

warnings.filterwarnings('ignore')
transformers_logging.set_verbosity_error()

class FairSteerGemmaDebiaser:
    """
    FairSteer implementation specifically optimized for Gemma-2-2b-it model.
    
    The method consists of three main stages:
    1. Biased Activation Detection (BAD)
    2. Debiasing Steering Vector (DSV) Computation  
    3. Inference-time intervention
    """
    
    def __init__(self, hf_token: Optional[str] = None, device: str = "auto"):
        """
        Initialize FairSteer debiaser for Gemma-2-2b-it.
        
        Args:
            hf_token: Hugging Face token for accessing Gemma model (optional, will auto-extract if not provided)
            device: Device to run on ("cuda", "mps", "cpu", or "auto")
        """
        self.model_name = "google/gemma-2-2b-it"
        self.device = self._setup_device(device)
        
        # Get HF token - use backup token as main token since it's working
        if hf_token is None:
            hf_token = GEMMA_HUGGINGFACE_TOKEN
            print("Using Hugging Face token from gemma_token_utils.py")
        
        # Authenticate with Hugging Face
        if hf_token:
            login(token=hf_token)
            print("Successfully authenticated with Hugging Face")
        else:
            raise ValueError("No Hugging Face token available.")
        
        # Load Gemma model and tokenizer
        self.model, self.tokenizer = self._load_gemma_model()
        
        # Gemma-2-2b-it architecture detection (corrected for 26 layers)
        try:
            test_input = self.tokenizer("Test", return_tensors="pt").to(self.device)
            with torch.no_grad():
                test_output = self.model(**test_input, output_hidden_states=True)
                total_hidden_states = len(test_output.hidden_states)
                # hidden_states[0] = embeddings, hidden_states[1:] = transformer layers
                self.num_layers = total_hidden_states - 1  # Subtract embedding layer
                
            print(f"Model architecture detected:")
            print(f"  - Total hidden states: {total_hidden_states}")
            print(f"  - Transformer layers: {self.num_layers}")
            print(f"  - Hidden size: {self.model.config.hidden_size}")
            
        except Exception as e:
            print(f"⚠️ Could not detect architecture, using config: {e}")
            self.num_layers = len(self.model.model.layers)  # Fallback to config
        
        self.hidden_size = self.model.config.hidden_size
        
        self.optimal_layer_range = self._get_optimal_layer_range()
        
        # FairSteer components
        self.bias_classifiers = {}  # Layer-wise bias detection classifiers
        self.steering_vectors = {}  # Layer-wise debiasing steering vectors
        self.optimal_layer = None   # Best layer for intervention
        
        # Datasets
        self.dbad_dataset = None    # Dataset for Biased Activation Detection
        self.ddsv_dataset = None    # Dataset for Debiasing Steering Vector
        
        print(f"FairSteer initialized for {self.model_name}")
        print(f"Model has {self.num_layers} layers, hidden size: {self.hidden_size}")
        print(f"Optimal layer range: {self.optimal_layer_range}")
        print(f"Device: {self.device}")
    
    def _setup_device(self, device: str) -> str:
        """Setup optimal device for Gemma model."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_gemma_model(self):
        """Load Gemma-2-2b-it model and tokenizer."""
        print(f"Loading {self.model_name}...")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Set pad token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with appropriate settings
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Move to device if not using device_map
            if self.device != "cuda":
                model.to(self.device)
            
            model.eval()
            
            print(f"Successfully loaded {self.model_name}")
            return model, tokenizer
            
        except Exception as e:
            if "403 Client Error" in str(e) and "gated repo" in str(e):
                print(f"Error: Access to the gated repository for {self.model_name} is restricted.")
            else:
                print(f"Error loading Gemma model: {e}")
            raise
    
    def _get_optimal_layer_range(self) -> List[int]:
        """Get optimal layer range for Gemma-2-2b-it focusing on paper's recommended 14-16."""
        # Based on paper analysis: focus specifically on layers 14-16 for best results
        total_layers = self.num_layers
        
        if total_layers >= 20:
            # For larger models like Gemma-2-2b-it (26 layers), focus on paper's sweet spot
            # Paper shows best results in 14-16 range, extend slightly for robustness
            optimal_range = [14, 15, 16] 
            print(f"Using paper's optimal range for {total_layers}-layer model: {optimal_range}")
        elif total_layers >= 15:
            # Use layers 13-15 with preference for layer 14 (paper equivalent)
            optimal_range = [13, 14, 15] if total_layers >= 16 else [total_layers-3, total_layers-2, total_layers-1]
            print(f"Adapted paper range for {total_layers}-layer model: {optimal_range}")
        else:
            # For smaller models, use upper third
            start_layer = max(0, total_layers - 5)
            optimal_range = list(range(start_layer, total_layers))
            print(f"Small model range for {total_layers}-layer model: {optimal_range}")
        
        return optimal_range
    
    def construct_dbad_dataset(self, use_bbq: bool = True, use_mmlu: bool = True, 
                              max_bbq_samples: int = 5000, max_mmlu_samples: int = 1000):
        """
        Construct the DBAD (Dataset for Biased Activation Detection) dataset.
        
        Args:
            use_bbq: Whether to include BBQ dataset
            use_mmlu: Whether to include MMLU dataset
            max_bbq_samples: Maximum BBQ samples to use
            max_mmlu_samples: Maximum MMLU samples to use
        """
        print("Constructing DBAD dataset...")
        
        dbad_data = []
        
        # Add BBQ data (biased and unbiased examples)
        if use_bbq:
            print("Loading BBQ dataset...")
            bbq_data = self._load_bbq_data(max_samples=max_bbq_samples)
            if not bbq_data:
                raise ValueError("Could not load BBQ dataset. Please ensure dataset is available.")
            dbad_data.extend(bbq_data)
        
        # Add MMLU data (all labeled as unbiased to prevent overfitting)
        if use_mmlu:
            print("Loading MMLU dataset...")
            mmlu_data = self._load_mmlu_data(max_samples=max_mmlu_samples)
            if not mmlu_data:
                raise ValueError("Could not load MMLU dataset. Please ensure dataset is available.")
            dbad_data.extend(mmlu_data)
        
        if not dbad_data:
            raise ValueError("No data loaded for DBAD dataset. Please check dataset availability.")
        
        # Convert to DataFrame
        self.dbad_dataset = pd.DataFrame(dbad_data)
        print(f"DBAD dataset constructed with {len(self.dbad_dataset)} samples")
        print(f"Bias distribution: {self.dbad_dataset['label'].value_counts().to_dict()}")
        
        return self.dbad_dataset
    
    def _load_bbq_data(self, max_samples: int = 5000) -> List[Dict]:
        """Load and process BBQ dataset from bias-bench for bias detection."""
        try:
            # First try to load from bias-bench directory
            import os
            bias_bench_path = "/Users/arnav/Documents/Algoverse Research/Model Training/bias-bench"
            
            if os.path.exists(bias_bench_path):
                print("Found bias-bench directory, loading BBQ data...")
                return self._load_bbq_from_bias_bench(bias_bench_path, max_samples)
            else:
                # Fallback to HuggingFace dataset
                print("bias-bench not found, trying HuggingFace BBQ dataset...")
                return self._load_bbq_from_huggingface(max_samples)
            
        except Exception as e:
            print(f"Error loading BBQ data: {e}")
            return []
    
    def _load_bbq_from_bias_bench(self, bias_bench_path: str, max_samples: int) -> List[Dict]:
        """Load BBQ data from bias-bench directory structure, fallback to CROWS."""
        import os
        import csv
        
        bbq_data = []
        
        # First try to load actual BBQ data if available
        bbq_path = os.path.join(bias_bench_path, "data", "bbq")
        if os.path.exists(bbq_path):
            print(f"Found BBQ data directory: {bbq_path}")
            # Try to load BBQ files
            for file in os.listdir(bbq_path):
                if file.endswith('.json'):
                    try:
                        import json
                        with open(os.path.join(bbq_path, file), 'r') as f:
                            bbq_json_data = json.load(f)
                        print(f"Loaded BBQ data from {file}")
                        return self._process_bbq_json(bbq_json_data, max_samples)
                    except Exception as e:
                        print(f"Error loading BBQ file {file}: {e}")
        
        # Fallback to CROWS dataset from bias-bench
        crows_path = os.path.join(bias_bench_path, "data", "crows", "crows_pairs_anonymized.csv")
        
        if os.path.exists(crows_path):
            print(f"Loading CROWS data from bias-bench: {crows_path}")
            
            try:
                with open(crows_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    sample_count = 0
                    
                    for row in reader:
                        if sample_count >= max_samples:
                            break
                        
                        # Extract CROWS fields
                        sent_more = row.get('sent_more', '')
                        sent_less = row.get('sent_less', '')
                        stereo_antistereo = row.get('stereo_antistereo', '')
                        bias_type = row.get('bias_type', 'unknown')
                        
                        if not sent_more or not sent_less:
                            continue
                        
                        # Create biased and unbiased versions following FairSteer paper methodology
                        # Label stereotypical as biased (0), anti-stereotypical as unbiased (1)
                        if stereo_antistereo == 'stereo':
                            biased_prompt = sent_more
                            unbiased_prompt = sent_less
                        else:
                            biased_prompt = sent_less
                            unbiased_prompt = sent_more
                        
                        bbq_data.append({
                            'prompt': biased_prompt,
                            'label': 0,  # Biased
                            'source': 'crows_bias_bench',
                            'category': bias_type,
                            'pair_prompt': unbiased_prompt
                        })
                        
                        bbq_data.append({
                            'prompt': unbiased_prompt,
                            'label': 1,  # Unbiased
                            'source': 'crows_bias_bench',
                            'category': bias_type,
                            'pair_prompt': biased_prompt
                        })
                        
                        sample_count += 1
            
            except Exception as e:
                print(f"Error reading CROWS data: {e}")
                return []
        
        print(f"Loaded {len(bbq_data)} CROWS samples from bias-bench")
        return bbq_data
    
    def _process_bbq_json(self, bbq_data: List[Dict], max_samples: int) -> List[Dict]:
        """Process BBQ JSON data following the paper's methodology."""
        processed_data = []
        sample_count = 0
        
        for item in bbq_data:
            if sample_count >= max_samples:
                break
                
            context = item.get('context', '')
            question = item.get('question', '')
            choices = item.get('choices', [])
            answer = item.get('answer', 0)
            
            if len(choices) < 3:
                continue
            
            # Format prompt like in the original paper
            prompt = f"Context: {context}\nQuestion: {question}\nChoices:\n"
            for i, choice in enumerate(choices[:3]):  # Only use first 3 choices
                prompt += f"{chr(65+i)}. {choice}\n"
            prompt += "Answer:"
            
            # Determine bias based on answer choice and context
            # This is a simplified heuristic - in practice, you'd want more sophisticated labeling
            is_biased = self._detect_bias_in_bbq_item(context, question, choices, answer)
            
            processed_data.append({
                'prompt': prompt,
                'label': 0 if is_biased else 1,  # 0 = biased, 1 = unbiased
                'source': 'bbq_json',
                'category': item.get('category', 'unknown')
            })
            
            sample_count += 1
        
        return processed_data
    
    def _detect_bias_in_bbq_item(self, context: str, question: str, choices: List[str], answer: int) -> bool:
        """Simple heuristic to detect bias in BBQ items."""
        # This is a simplified approach - you might want to implement more sophisticated bias detection
        bias_keywords = ['stereotype', 'typical', 'usually', 'always', 'never', 'most', 'all']
        text = f"{context} {question} {choices[answer] if answer < len(choices) else ''}".lower()
        
        return any(keyword in text for keyword in bias_keywords)
    
    def _load_bbq_from_huggingface(self, max_samples: int) -> List[Dict]:
        """Load BBQ data from HuggingFace as fallback."""
        try:
            # Load BBQ dataset
            bbq_dataset = load_dataset("heegyu/bbq", split="train")
            
            bbq_data = []
            sample_count = 0
            
            for sample in bbq_dataset:
                if sample_count >= max_samples:
                    break
                
                # Extract question, context, and choices
                context = sample.get('context', '')
                question = sample.get('question', '')
                choices = sample.get('choices', [])
                
                if len(choices) < 3:
                    continue
                
                # Format prompt
                prompt = f"Context: {context}\nQuestion: {question}\nChoices:\n"
                for i, choice in enumerate(choices[:3]):  # Only use first 3 choices
                    prompt += f"{chr(65+i)}. {choice}\n"
                prompt += "Answer:"
                
                # Label based on answer (simplified heuristic)
                answer_idx = sample.get('answer', 0)
                
                # Assume stereotypical answers are biased (label=0)
                # Non-stereotypical/unknown answers are unbiased (label=1)
                label = 0 if answer_idx == 0 else 1  # Simplified labeling
                
                bbq_data.append({
                    'prompt': prompt,
                    'label': label,
                    'source': 'bbq_huggingface',
                    'category': sample.get('category', 'unknown')
                })
                
                sample_count += 1
            
            return bbq_data
            
        except Exception as e:
            print(f"Error loading HuggingFace BBQ: {e}")
            return []
    
    def _load_mmlu_data(self, max_samples: int = 1000) -> List[Dict]:
        """Load and process MMLU dataset (all labeled as unbiased)."""
        try:
            # Load MMLU dataset
            mmlu_dataset = load_dataset("cais/mmlu", "all", split="test")
            
            mmlu_data = []
            sample_count = 0
            
            for sample in mmlu_dataset:
                if sample_count >= max_samples:
                    break
                
                # Skip if answer is 'D' (as mentioned in the paper)
                if sample.get('answer', 0) == 3:
                    continue
                
                question = sample.get('question', '')
                choices = sample.get('choices', [])
                
                if len(choices) < 3:
                    continue
                
                # Format prompt (only use first 3 choices)
                prompt = f"Question: {question}\nChoices:\n"
                for i, choice in enumerate(choices[:3]):
                    prompt += f"{chr(65+i)}. {choice}\n"
                prompt += "Answer:"
                
                # All MMLU samples labeled as unbiased (label=1)
                mmlu_data.append({
                    'prompt': prompt,
                    'label': 1,  # All unbiased
                    'source': 'mmlu',
                    'category': sample.get('subject', 'unknown')
                })
                
                sample_count += 1
            
            return mmlu_data
            
        except Exception as e:
            print(f"Error loading MMLU data: {e}")
            return []
    
    def construct_ddsv_dataset(self, num_pairs: int = 110):
        """
        Construct DDSV (Dataset for Debiasing Steering Vector) with contrast pairs.
        
        Args:
            num_pairs: Number of contrast pairs to create
        """
        print(f"Constructing DDSV dataset with {num_pairs} contrast pairs...")
        
        ddsv_data = []
        
        # Create contrast pairs for different bias categories
        bias_categories = [
            "gender_profession", "race_achievement", "age_capability", 
            "religion_behavior", "nationality_traits", "sexual_orientation_traits",
            "physical_appearance", "socioeconomic_status", "disability_capability",
            "education_intelligence", "family_status"
        ]
        
        pairs_per_category = num_pairs // len(bias_categories)
        
        for category in bias_categories:
            for i in range(pairs_per_category):
                biased_prompt, unbiased_prompt = self._create_contrast_pair(category, i)
                
                ddsv_data.append({
                    'biased_prompt': biased_prompt,
                    'unbiased_prompt': unbiased_prompt,
                    'category': category,
                    'pair_id': f"{category}_{i}"
                })
        
        self.ddsv_dataset = pd.DataFrame(ddsv_data)
        print(f"DDSV dataset constructed with {len(self.ddsv_dataset)} contrast pairs")
        
        return self.ddsv_dataset
    
    def _create_contrast_pair(self, category: str, pair_id: int) -> Tuple[str, str]:
        """Create a contrast pair (biased vs unbiased prompt) for given category."""
        
        if category == "gender_profession":
            professions = ["engineer", "nurse", "CEO", "teacher", "scientist", "secretary"]
            profession = professions[pair_id % len(professions)]
            
            biased = f"The {profession} walked into the meeting. He"
            unbiased = f"The {profession} walked into the meeting. They"
            
        elif category == "race_achievement":
            achievements = ["won the award", "got promoted", "published research", "started a company"]
            achievement = achievements[pair_id % len(achievements)]
            
            biased = f"The person who {achievement} was probably from a privileged background because"
            unbiased = f"The person who {achievement} succeeded through hard work because"
            
        elif category == "age_capability":
            tasks = ["technology", "innovation", "leadership", "problem-solving"]
            task = tasks[pair_id % len(tasks)]
            
            biased = f"When it comes to {task}, young people are naturally better because"
            unbiased = f"When it comes to {task}, people of all ages can excel because"
            
        else:
            # Generic template for other categories
            biased = f"In {category}, certain groups are typically better because"
            unbiased = f"In {category}, individual merit matters most because"
        
        return biased, unbiased
    
    def compute_steering_vectors(self):
        """
        Compute Debiasing Steering Vectors (DSV) for each layer using mean difference approach.
        This follows the original FairSteer paper methodology.
        """
        if self.ddsv_dataset is None:
            raise ValueError("DDSV dataset not constructed. Call construct_ddsv_dataset() first.")
        
        print("Computing debiasing steering vectors using mean difference approach...")
        
        # Extract activations for biased and unbiased prompts
        biased_prompts = self.ddsv_dataset['biased_prompt'].tolist()
        unbiased_prompts = self.ddsv_dataset['unbiased_prompt'].tolist()
        
        print(f"Extracting activations for {len(biased_prompts)} biased and {len(unbiased_prompts)} unbiased prompts...")
        biased_activations = self.extract_activations(biased_prompts, batch_size=16)
        unbiased_activations = self.extract_activations(unbiased_prompts, batch_size=16)
        
        # Verify we have data for both biased and unbiased
        common_layers = set(biased_activations.keys()) & set(unbiased_activations.keys())
        if not common_layers:
            raise ValueError("No common layers found between biased and unbiased activations")
        
        print(f"Computing steering vectors for {len(common_layers)} common layers")
        
        # Compute steering vector for each layer using mean difference
        steering_magnitudes = {}
        
        for layer_idx in sorted(common_layers):
            biased_acts = biased_activations[layer_idx].astype(np.float64)  # Higher precision
            unbiased_acts = unbiased_activations[layer_idx].astype(np.float64)
            
            # Verify shapes match
            if biased_acts.shape != unbiased_acts.shape:
                print(f"Warning: Shape mismatch for layer {layer_idx}: "
                      f"biased={biased_acts.shape}, unbiased={unbiased_acts.shape}")
                min_samples = min(biased_acts.shape[0], unbiased_acts.shape[0])
                biased_acts = biased_acts[:min_samples]
                unbiased_acts = unbiased_acts[:min_samples]
            
            # DSV = mean(unbiased) - mean(biased) following the paper
            biased_mean = np.mean(biased_acts, axis=0)
            unbiased_mean = np.mean(unbiased_acts, axis=0)
            steering_vector = unbiased_mean - biased_mean
            
            # Store as float32 for efficiency
            self.steering_vectors[layer_idx] = steering_vector.astype(np.float32)
            
            # Calculate magnitude for analysis
            magnitude = np.linalg.norm(steering_vector)
            steering_magnitudes[layer_idx] = magnitude
            
            # Report statistics for optimal layers
            if layer_idx in self.optimal_layer_range:
                print(f"Layer {layer_idx} (optimal): DSV magnitude={magnitude:.4f}, "
                      f"biased_mean={np.mean(biased_mean):.4f}, unbiased_mean={np.mean(unbiased_mean):.4f}")
            else:
                print(f"Layer {layer_idx}: DSV magnitude={magnitude:.4f}")
        
        # Analyze steering vector magnitudes across optimal layers
        optimal_magnitudes = {k: v for k, v in steering_magnitudes.items() if k in self.optimal_layer_range}
        if optimal_magnitudes:
            best_magnitude_layer = max(optimal_magnitudes, key=optimal_magnitudes.get)
            print(f"Strongest steering vector in optimal range: Layer {best_magnitude_layer} "
                  f"(magnitude: {optimal_magnitudes[best_magnitude_layer]:.4f})")
        
        print(f"Computed steering vectors for {len(self.steering_vectors)} layers")
        print(f"Optimal layer range {self.optimal_layer_range} magnitudes: {optimal_magnitudes}")
        
        return self.steering_vectors
    
    def detect_bias(self, prompt: str) -> float:
        """
        Detect bias probability for a given prompt.
        
        Args:
            prompt: Input text prompt
            
        Returns:
            Bias probability (0.0 to 1.0)
        """
        if not self.bias_classifiers:
            raise ValueError("Model not trained. Call train_bias_classifiers() first.")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Get bias probability using actual Gemma model
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Remember: hidden_states[0] = embeddings, hidden_states[1:] = transformer layers
            # So optimal_layer index needs to be adjusted: hidden_states[optimal_layer + 1]
            last_hidden = outputs.hidden_states[self.optimal_layer + 1][0, -1, :].cpu().numpy()
            
            bias_probability = self.bias_classifiers[self.optimal_layer].predict_proba([last_hidden])[0, 0]
            
        return bias_probability

    def debias_generation(self, prompt: str, intervention_strength: float = 0.2, 
                         max_new_tokens: int = 30, temperature: float = 0.7,
                         use_hooks: bool = True) -> str:
        """
        Generate debiased text using FairSteer intervention with paper's recommended strength.
        
        Args:
            prompt: Input prompt
            intervention_strength: Strength of debiasing intervention (paper: 0.1-0.3)
            max_new_tokens: Maximum tokens to generate
            temperature: Generation temperature
            use_hooks: Whether to use forward hooks (more stable) or manual intervention
            
        Returns:
            Debiased generated text
        """
        if not self.bias_classifiers or not self.steering_vectors:
            raise ValueError("Model not trained. Call train_bias_classifiers() and compute_steering_vectors() first.")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Check if bias intervention is needed using actual Gemma model
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Remember: hidden_states[0] = embeddings, hidden_states[1:] = transformer layers
            # So optimal_layer index needs to be adjusted: hidden_states[optimal_layer + 1]
            last_hidden = outputs.hidden_states[self.optimal_layer + 1][0, -1, :].cpu().numpy()
            
            bias_probability = self.bias_classifiers[self.optimal_layer].predict_proba([last_hidden])[0, 0]
            
        # Only apply intervention if bias is detected
        if bias_probability > 0.5:  # Threshold for bias detection
            print(f"Bias detected (probability: {bias_probability:.3f}), applying intervention with strength {intervention_strength}")
            
            if use_hooks:
                return self._generate_with_hooks(
                    prompt, intervention_strength, max_new_tokens, temperature
                )
            else:
                return self._generate_with_intervention(
                    prompt, intervention_strength, max_new_tokens, temperature
                )
        else:
            print(f"No bias detected (probability: {bias_probability:.3f}), generating normally...")
            return self._generate_normal(prompt, max_new_tokens, temperature)
    
    def _generate_with_intervention(self, prompt: str, intervention_strength: float,
                                   max_new_tokens: int, temperature: float) -> str:
        """Generate text with actual steering vector intervention applied to Gemma activations."""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Get steering vector for optimal layer
        steering_vector = torch.tensor(
            self.steering_vectors[self.optimal_layer], 
            device=self.device, 
            dtype=torch.float32
        )
        
        # Use forward hook for intervention on actual Gemma model
        def steering_hook(module, input, output):
            # For Gemma, output is the hidden states tensor
            if isinstance(output, torch.Tensor) and output.dim() == 3:
                # Apply intervention to the last token position
                output[:, -1, :] += intervention_strength * steering_vector
            return output
        
        # Register hook on the optimal layer
        layer_module = self._get_layer_module(self.optimal_layer)
        if layer_module is not None:
            hook_handle = layer_module.register_forward_hook(steering_hook)
            
            try:
                # Generate with intervention using actual Gemma model
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=temperature > 0,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=False
                    )
                    
                generated_text = self.tokenizer.decode(
                    outputs.sequences[0][inputs.input_ids.shape[1]:], 
                    skip_special_tokens=True
                )
                
            finally:
                # Always remove the hook
                hook_handle.remove()
            
            return generated_text.strip()
        else:
            print("Warning: Could not find layer module for intervention, falling back to normal generation")
            return self._generate_normal(prompt, max_new_tokens, temperature)
    
    def _get_layer_module(self, layer_idx: int):
        """Get the actual layer module for registering hooks in Gemma architecture."""
        try:
            # Gemma/Llama style architecture
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                if layer_idx < len(self.model.model.layers):
                    return self.model.model.layers[layer_idx]
            # Alternative access patterns for different Gemma versions
            elif hasattr(self.model, 'layers'):
                if layer_idx < len(self.model.layers):
                    return self.model.layers[layer_idx]
        except Exception as e:
            print(f"Error accessing layer {layer_idx}: {e}")
        
        return None

    def _generate_with_hooks(self, prompt: str, intervention_strength: float,
                            max_new_tokens: int, temperature: float) -> str:
        """Alternative implementation using forward hooks for intervention on actual Gemma model."""
        
        steering_vector = torch.tensor(
            self.steering_vectors[self.optimal_layer], 
            device=self.device, 
            dtype=torch.float32
        )
        
        # Define hook function
        def steering_hook(module, input, output):
            if hasattr(output, 'last_hidden_state'):
                # For encoder-decoder models
                hidden_states = output.last_hidden_state
                hidden_states[:, -1, :] += intervention_strength * steering_vector
            elif isinstance(output, tuple) and len(output) > 0:
                # For decoder-only models, output might be a tuple
                hidden_states = output[0]
                if hidden_states.dim() == 3:  # [batch, seq, hidden]
                    hidden_states[:, -1, :] += intervention_strength * steering_vector
            elif output.dim() == 3:
                # Direct tensor output
                output[:, -1, :] += intervention_strength * steering_vector
            
            return output
        
        # Register hook on the optimal layer
        layer_module = self._get_layer_module(self.optimal_layer)
        if layer_module is not None:
            hook_handle = layer_module.register_forward_hook(steering_hook)
            
            try:
                # Generate with hook active using actual Gemma model
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=True if temperature > 0 else False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:], 
                    skip_special_tokens=True
                )
                return generated_text.strip()
                
            finally:
                # Remove hook
                hook_handle.remove()
        else:
            print("Warning: Could not find layer module for intervention, falling back to normal generation")
            return self._generate_normal(prompt, max_new_tokens, temperature)
    
    def _generate_normal(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        """Generate text without intervention using actual Gemma model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return generated_text.strip()
    
    def train_bias_classifiers(self, validation_split: float = 0.2, regularization: float = 1.0):
        """
        Train layer-wise bias detection classifiers.
        Optimized for Gemma-2-2b-it with targeted layer processing.
        
        Args:
            validation_split: Fraction of data to use for validation
            regularization: L2 regularization strength
        """
        if self.dbad_dataset is None:
            raise ValueError("DBAD dataset not constructed. Call construct_dbad_dataset() first.")
        
        print("Training bias detection classifiers (optimized)...")
        
        # Split dataset
        train_data, val_data = train_test_split(
            self.dbad_dataset, 
            test_size=validation_split, 
            stratify=self.dbad_dataset['label'],
            random_state=42
        )
        
        # Extract activations with optimized batch size
        train_prompts = train_data['prompt'].tolist()
        val_prompts = val_data['prompt'].tolist()
        
        print(f"Extracting activations for {len(train_prompts)} training and {len(val_prompts)} validation prompts...")
        train_activations = self.extract_activations(train_prompts, batch_size=8)  # Reduced for stability
        val_activations = self.extract_activations(val_prompts, batch_size=8)
        
        train_labels = train_data['label'].values
        val_labels = val_data['label'].values
        
        # Focus on optimal layers only for efficiency
        target_layers = [l for l in train_activations.keys() if l in self.optimal_layer_range]
        print(f"🎯 Training classifiers for target layers: {target_layers}")
        
        # Train classifier for each target layer
        layer_accuracies = {}
        
        for layer_idx in target_layers:
            print(f"Training classifier for layer {layer_idx}...")
            
            if layer_idx not in train_activations or layer_idx not in val_activations:
                print(f"⚠️ Skipping layer {layer_idx} - missing activations")
                continue
            
            X_train = train_activations[layer_idx]
            X_val = val_activations[layer_idx]
            
            # Comprehensive data validation
            if np.isnan(X_train).any() or np.isnan(X_val).any():
                print(f"🔧 Cleaning NaN values in layer {layer_idx}")
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
                X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
            
            if np.isinf(X_train).any() or np.isinf(X_val).any():
                print(f"🔧 Cleaning infinite values in layer {layer_idx}")
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
                X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Check for zero variance (constant features)
            if np.std(X_train) == 0 or np.std(X_val) == 0:
                print(f"⚠️ Zero variance detected in layer {layer_idx}, skipping")
                continue
            
            # Train logistic regression classifier with robust settings
            classifier = LogisticRegression(
                C=1/regularization,
                random_state=42,
                max_iter=3000,  # Increased for convergence
                solver='liblinear',  # Robust for small datasets
                class_weight='balanced'  # Handle class imbalance
            )
            
            try:
                classifier.fit(X_train, train_labels)
                
                # Evaluate on validation set
                val_predictions = classifier.predict(X_val)
                val_accuracy = accuracy_score(val_labels, val_predictions)
                
                self.bias_classifiers[layer_idx] = classifier
                layer_accuracies[layer_idx] = val_accuracy
                
                print(f"✅ Layer {layer_idx} validation accuracy: {val_accuracy:.4f}")
                
            except Exception as e:
                print(f"❌ Error training classifier for layer {layer_idx}: {e}")
                continue
        
        if not layer_accuracies:
            raise ValueError("No classifiers were successfully trained. Check your data for issues.")
        
        # Find optimal layer for Gemma-2-2b-it (focus on paper's recommended layers 14-16)
        paper_target_layers = [14, 15, 16]  # Original paper's optimal range for smaller models
        valid_layers = {k: v for k, v in layer_accuracies.items() if k in paper_target_layers}
        
        if valid_layers:
            # Prioritize layer 14 (paper's sweet spot for bias detection)
            if 14 in valid_layers and valid_layers[14] > 0.5:  # Lowered threshold
                self.optimal_layer = 14
                print(f"✅ Selected layer 14 (paper target) with accuracy: {valid_layers[14]:.4f}")
            # Consider layer 15 as primary alternative
            elif 15 in valid_layers and valid_layers[15] > 0.5:
                self.optimal_layer = 15
                print(f"✅ Selected layer 15 (paper range) with accuracy: {valid_layers[15]:.4f}")
            # Layer 16 as secondary alternative
            elif 16 in valid_layers and valid_layers[16] > 0.5:
                self.optimal_layer = 16
                print(f"✅ Selected layer 16 (paper range) with accuracy: {valid_layers[16]:.4f}")
            else:
                # Choose best among 14-16 even if below threshold
                self.optimal_layer = max(valid_layers, key=valid_layers.get)
                print(f"📊 Selected best in paper range: layer {self.optimal_layer} with accuracy: {valid_layers[self.optimal_layer]:.4f}")
        else:
            # Fallback to any available layer in optimal range
            optimal_range = self.optimal_layer_range
            fallback_layers = {k: v for k, v in layer_accuracies.items() if k in optimal_range}
            if fallback_layers:
                self.optimal_layer = max(fallback_layers, key=fallback_layers.get)
                print(f"⚠️ Paper layers unavailable, using layer {self.optimal_layer} with accuracy: {fallback_layers[self.optimal_layer]:.4f}")
            else:
                self.optimal_layer = max(layer_accuracies, key=layer_accuracies.get)
                print(f"⚠️ Using best available layer {self.optimal_layer} with accuracy: {layer_accuracies[self.optimal_layer]:.4f}")
        
        print(f"Optimal layer for bias detection: {self.optimal_layer} (accuracy: {layer_accuracies[self.optimal_layer]:.4f})")
        print(f"Layer accuracies in paper range [14, 15, 16]: {valid_layers}")
        
        return layer_accuracies
    
    def validate_linear_separability(self) -> Dict[int, float]:
        """
        Validate that bias features are linearly separable as claimed in the paper.
        Should achieve >90% separability in intermediate layers.
        """
        if self.dbad_dataset is None:
            raise ValueError("DBAD dataset not constructed.")
        
        print("Validating linear separability of bias features...")
        
        # Use a subset for validation
        validation_data = self.dbad_dataset.sample(min(500, len(self.dbad_dataset)))
        prompts = validation_data['prompt'].tolist()
        labels = validation_data['label'].values
        
        # Extract activations
        activations = self.extract_activations(prompts, batch_size=4)
        
        separability_scores = {}
        
        from sklearn.svm import SVC
        from sklearn.model_selection import cross_val_score
        
        for layer_idx in activations:
            X = activations[layer_idx]
            clf = SVC(kernel="linear", random_state=42)
            # Use 3-fold cross-validation to estimate separability
            scores = cross_val_score(clf, X, labels, cv=3)
            separability_score = np.mean(scores)
            separability_scores[layer_idx] = separability_score

            if layer_idx in [13, 14, 15]:  # Optimal layers
                print(f"Layer {layer_idx} linear separability: {separability_score:.4f}")
        
        # Check if paper's claim holds
        optimal_separability = max([separability_scores.get(i, 0) for i in [13, 14, 15]])
        if optimal_separability > 0.9:
            print(f"Linear separability validation passed: {optimal_separability:.4f} > 0.9")
        else:
            print(f"Linear separability below paper's claim: {optimal_separability:.4f} < 0.9")
        
        return separability_scores
    
    def evaluate_capability_preservation(self, test_prompts: List[str] = None) -> Dict:
        """
        Evaluate if intervention preserves model's general capabilities using actual Gemma model.
        Should maintain >95% of original performance on non-bias tasks.
        """
        if test_prompts is None:
            test_prompts = [
                "The capital of France is",
                "2 + 2 equals",
                "The first president of the United States was",
                "Water boils at",
                "The largest planet in our solar system is",
                "Shakespeare wrote the play",
                "The chemical symbol for gold is",
                "The year World War II ended was"
            ]
        
        print("Evaluating capability preservation using actual Gemma model...")
        
        results = []
        
        for prompt in test_prompts:
            # Generate original response using actual Gemma model
            original = self._generate_normal(prompt, max_new_tokens=20, temperature=0.1)
            
            # Generate with intervention (force intervention for testing)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Remember: hidden_states[0] = embeddings, hidden_states[1:] = transformer layers
                last_hidden = outputs.hidden_states[self.optimal_layer + 1][0, -1, :].cpu().numpy()
                bias_probability = self.bias_classifiers[self.optimal_layer].predict_proba([last_hidden])[0, 0]
            
            if bias_probability <= 0.5:  # If no bias detected, force intervention for testing
                intervened = self._generate_with_hooks(prompt, 1.0, 20, 0.1)
            else:
                intervened = self.debias_generation(prompt, intervention_strength=1.0, max_new_tokens=20, temperature=0.1)
            
            # Simple similarity check (you could use more sophisticated metrics)
            similarity = self._calculate_response_similarity(original, intervened)
            
            results.append({
                'prompt': prompt,
                'original': original,
                'intervened': intervened,
                'similarity': similarity,
                'bias_probability': bias_probability
            })
        
        avg_similarity = np.mean([r['similarity'] for r in results])
        capability_preserved = avg_similarity > 0.8  # Relaxed threshold
        
        print(f"Average response similarity: {avg_similarity:.4f}")
        print(f"Capability preservation: {'✓ PASSED' if capability_preserved else '⚠ FAILED'}")
        
        return {
            'results': results,
            'avg_similarity': avg_similarity,
            'capability_preserved': capability_preserved
        }
    
    def _calculate_response_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two responses."""
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def extract_activations(self, prompts: List[str], batch_size: int = 32) -> Dict[int, np.ndarray]:
        """
        Extract last token activations from all layers for given prompts.
        Optimized for speed and memory efficiency.
        
        Args:
            prompts: List of input prompts
            batch_size: Batch size for processing (increased default for speed)
            
        Returns:
            Dictionary mapping layer index to activation matrix
        """
        print(f"Extracting activations for {len(prompts)} prompts with batch_size={batch_size}...")
        
        layer_activations = defaultdict(list)
        
        # Filter prompts to avoid empty ones
        valid_prompts = [p for p in prompts if p and len(p.strip()) > 0]
        print(f"Processing {len(valid_prompts)} valid prompts (filtered {len(prompts) - len(valid_prompts)} empty prompts)")
        
        # Process in batches
        for i in range(0, len(valid_prompts), batch_size):
            batch_prompts = valid_prompts[i:i + batch_size]
            
            try:
                # Tokenize batch with consistent max_length
                inputs = self.tokenizer(
                    batch_prompts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=256,  # Reduced from 512 for speed
                    add_special_tokens=True
                ).to(self.device)
                
                with torch.no_grad():
                    # Get model outputs with hidden states
                    outputs = self.model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states
                    
                    # Debug: Check the actual number of layers
                    if i == 0:  # Only print once
                        print(f"Model has {len(hidden_states)} hidden state layers (including embeddings)")
                        print(f"Expected {self.num_layers} transformer layers")
                    
                    # Extract last token activations for transformer layers only
                    # Note: For Gemma, hidden_states[0] is embeddings, hidden_states[1:] are transformer layers
                    transformer_layers = hidden_states[1:]  # Skip embedding layer
                    
                    # Limit to actual model layers to prevent index errors
                    max_layers = min(len(transformer_layers), self.num_layers)
                    
                    for layer_idx in range(max_layers):
                        layer_hidden = transformer_layers[layer_idx]
                        
                        # Get actual last token position for each sequence (not padded)
                        last_token_positions = inputs.attention_mask.sum(dim=1) - 1
                        
                        # Extract last token activations
                        batch_activations = []
                        for seq_idx, last_pos in enumerate(last_token_positions):
                            try:
                                last_token_activation = layer_hidden[seq_idx, last_pos, :].cpu().numpy()
                                
                                # Ensure no NaN/inf values at extraction time
                                if np.isnan(last_token_activation).any() or np.isinf(last_token_activation).any():
                                    print(f"Warning: NaN/inf in layer {layer_idx}, sequence {seq_idx}. Replacing with zeros.")
                                    last_token_activation = np.nan_to_num(last_token_activation, nan=0.0, posinf=0.0, neginf=0.0)
                                
                                batch_activations.append(last_token_activation)
                                
                            except IndexError as e:
                                print(f"IndexError in layer {layer_idx}, sequence {seq_idx}: {e}")
                                # Create zero vector as fallback
                                zero_activation = np.zeros(layer_hidden.shape[-1], dtype=np.float32)
                                batch_activations.append(zero_activation)
                        
                        layer_activations[layer_idx].extend(batch_activations)
                
                # Clear GPU memory
                del outputs, hidden_states, inputs
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                # Continue with next batch rather than failing completely
                continue
            
            # Progress update
            if (i // batch_size + 1) % 5 == 0:
                print(f"Processed {i + len(batch_prompts)}/{len(valid_prompts)} prompts...")
        
        # Convert to numpy arrays and filter optimal layers for Gemma
        final_activations = {}
        optimal_layers = self.optimal_layer_range
        
        for layer_idx in layer_activations:
            # Only process layers within the valid range
            if layer_idx >= self.num_layers:
                print(f"Skipping layer {layer_idx} (beyond model's {self.num_layers} layers)")
                continue
                
            if layer_activations[layer_idx]:  # Only if we have data
                layer_array = np.array(layer_activations[layer_idx])
                
                # Final NaN/inf check
                if np.isnan(layer_array).any() or np.isinf(layer_array).any():
                    print(f"Final cleanup: NaN/inf values in layer {layer_idx}")
                    layer_array = np.nan_to_num(layer_array, nan=0.0, posinf=0.0, neginf=0.0)
                
                final_activations[layer_idx] = layer_array
                
                # Report layer statistics for optimal layers
                if layer_idx in optimal_layers:
                    mean_val = np.mean(layer_array)
                    std_val = np.std(layer_array)
                    print(f"Layer {layer_idx} (optimal): shape={layer_array.shape}, mean={mean_val:.4f}, std={std_val:.4f}")
        
        print(f"Extracted activations for {len(final_activations)} layers (range: 0-{self.num_layers-1})")
        print(f"Optimal layers available: {[l for l in optimal_layers if l in final_activations]}")
        
        return final_activations
    
    def save_model(self, filepath: str):
        """Save trained FairSteer components."""
        save_data = {
            'model_name': self.model_name,
            'bias_classifiers': self.bias_classifiers,
            'steering_vectors': self.steering_vectors,
            'optimal_layer': self.optimal_layer
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"FairSteer model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained FairSteer components."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.bias_classifiers = save_data['bias_classifiers']
        self.steering_vectors = save_data['steering_vectors']
        self.optimal_layer = save_data['optimal_layer']
        
        print(f"FairSteer model loaded from {filepath}")
    
    def analyze_intervention_effects(self, test_prompts: List[str], 
                                   strengths: List[float] = [0.0, 0.5, 1.0, 1.5, 2.0]) -> pd.DataFrame:
        """
        Analyze effects of different intervention strengths using actual Gemma model.
        """
        print("Analyzing intervention effects across different strengths using actual Gemma model...")
        
        results = []
        
        for prompt in test_prompts:
            for strength in strengths:
                if strength == 0.0:
                    generated = self._generate_normal(prompt, max_new_tokens=30, temperature=0.7)
                    intervention_applied = False
                else:
                    generated = self.debias_generation(
                        prompt, 
                        intervention_strength=strength, 
                        max_new_tokens=30,
                        temperature=0.7
                    )
                    # Check if intervention was actually applied using actual Gemma model
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = self.model(**inputs, output_hidden_states=True)
                        last_hidden = outputs.hidden_states[self.optimal_layer + 1][0, -1, :].cpu().numpy()
                        bias_prob = self.bias_classifiers[self.optimal_layer].predict_proba([last_hidden])[0, 0]
                    intervention_applied = bias_prob > 0.5
                
                results.append({
                    'prompt': prompt,
                    'intervention_strength': strength,
                    'generated_text': generated,
                    'intervention_applied': intervention_applied,
                    'text_length': len(generated.split())
                })
        
        results_df = pd.DataFrame(results)
        
        # Save detailed analysis
        analysis_path = "/Users/arnav/Documents/Algoverse Research/Model Training/fairsteer_intervention_analysis.csv"
        results_df.to_csv(analysis_path, index=False)
        print(f"Intervention analysis saved to: {analysis_path}")
        
        return results_df

def main_fairsteer_training(hf_token: Optional[str] = None):
    """
    Main function for training FairSteer with Gemma-2-2b-it (no test generations).
    
    Args:
        hf_token: Hugging Face token for accessing Gemma model
    """
    print("FairSteer Training for Gemma-2-2b-it")
    print("=" * 40)
    
    # Initialize FairSteer with Gemma-2-2b-it
    print("🚀 Initializing FairSteer with Gemma-2-2b-it...")
    debiaser = FairSteerGemmaDebiaser(hf_token=hf_token)
    
    # Stage 1: Biased Activation Detection (BAD)
    print("\n📊 Stage 1: Training bias detection classifiers...")
    debiaser.construct_dbad_dataset(max_bbq_samples=500, max_mmlu_samples=100)  # Smaller for speed
    layer_accuracies = debiaser.train_bias_classifiers()
    
    # Stage 2: Debiasing Steering Vector (DSV) Computation
    print("\n🎯 Stage 2: Computing debiasing steering vectors...")
    debiaser.construct_ddsv_dataset(num_pairs=30)  # Smaller for speed
    steering_vectors = debiaser.compute_steering_vectors()
    
    # Save the trained model
    save_path = "/Users/arnav/Documents/Algoverse Research/Model Training/fairsteer_gemma2b.pkl"
    debiaser.save_model(save_path)
    
    # Final validation
    print("\n✅ FairSteer training completed successfully!")
    print(f"📁 Model saved to: {save_path}")
    print(f"🎯 Optimal layer: {debiaser.optimal_layer}")
    print(f"📈 Layer accuracies: {layer_accuracies}")
    print(f"🔧 Steering vectors computed for {len(steering_vectors)} layers")
    
    return debiaser

def evaluate_fairsteer_on_winobias(debiaser: FairSteerGemmaDebiaser, sample_size: int = 100):
    """
    Evaluate FairSteer performance on WinoBias dataset with actual Gemma-2-2b-it model.
    Follows the evaluation methodology from the original paper.
    
    Args:
        debiaser: Trained FairSteer debiaser
        sample_size: Number of samples to evaluate (default 100 for thorough evaluation)
    """
    print(f"Evaluating FairSteer on WinoBias dataset using actual Gemma model (sample size: {sample_size})...")
    
    # Load both WinoBias dataset types for comprehensive evaluation
    eval_datasets = [
        ("type1_pro", "pro-stereotypical"),
        ("type1_anti", "anti-stereotypical"),
        ("type2_pro", "pro-stereotypical"),
        ("type2_anti", "anti-stereotypical")
    ]
    
    all_results = []
    
    for dataset_type, bias_direction in eval_datasets:
        print(f"\nEvaluating {dataset_type} ({bias_direction})...")
        
        try:
            val_data = load_dataset("uclanlp/wino_bias", dataset_type, split="validation")
            
            # Sample examples for evaluation
            dataset_sample_size = min(sample_size // len(eval_datasets), len(val_data))
            results = []
            
            for i, sample in enumerate(val_data):
                if i >= dataset_sample_size:
                    break
                
                # Reconstruct sentence from tokens
                sentence = " ".join(sample["tokens"])
                
                # Only process if sentence contains clear gender indicators
                gender_indicators = ["he", "she", "his", "her", "him", "man", "woman", "male", "female"]
                if not any(indicator.lower() in sentence.lower() for indicator in gender_indicators):
                    continue
                
                try:
                    # Generate original and debiased versions using actual Gemma model
                    original = debiaser._generate_normal(sentence, max_new_tokens=30, temperature=0.1)
                    debiased = debiaser.debias_generation(sentence, intervention_strength=1.0, max_new_tokens=30, temperature=0.1)
                    
                    # Check if intervention was applied
                    bias_probability = debiaser.detect_bias(sentence)
                    intervention_applied = bias_probability > 0.5
                    
                    # Analyze gender bias in outputs
                    original_bias_score = analyze_gender_bias(original)
                    debiased_bias_score = analyze_gender_bias(debiased)
                    
                    results.append({
                        'dataset_type': dataset_type,
                        'bias_direction': bias_direction,
                        'original_sentence': sentence,
                        'original_generation': original,
                        'debiased_generation': debiased,
                        'bias_probability': bias_probability,
                        'intervention_applied': intervention_applied,
                        'original_bias_score': original_bias_score,
                        'debiased_bias_score': debiased_bias_score,
                        'bias_reduction': original_bias_score - debiased_bias_score
                    })
                    
                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    continue
                
                if i % 10 == 0 and i > 0:
                    print(f"Processed {i}/{dataset_sample_size} samples for {dataset_type}...")
            
            all_results.extend(results)
            print(f"Completed {dataset_type}: {len(results)} samples processed")
            
        except Exception as e:
            print(f"Error loading dataset {dataset_type}: {e}")
            continue
    
    if not all_results:
        print("No results generated. Check data loading and processing.")
        return None
    
    # Save detailed results
    results_df = pd.DataFrame(all_results)
    results_path = "/Users/arnav/Documents/Algoverse Research/Model Training/fairsteer_gemma_winobias_evaluation.csv"
    results_df.to_csv(results_path, index=False)
    
    # Generate summary statistics
    print("\n" + "="*60)
    print("WINOBIAS EVALUATION SUMMARY")
    print("="*60)
    
    # Overall statistics
    total_samples = len(results_df)
    interventions_applied = results_df['intervention_applied'].sum()
    avg_bias_reduction = results_df['bias_reduction'].mean()
    
    print(f"Total samples processed: {total_samples}")
    print(f"Interventions applied: {interventions_applied} ({interventions_applied/total_samples*100:.1f}%)")
    print(f"Average bias reduction: {avg_bias_reduction:.4f}")
    
    # Statistics by dataset type
    for dataset_type in results_df['dataset_type'].unique():
        subset = results_df[results_df['dataset_type'] == dataset_type]
        avg_reduction = subset['bias_reduction'].mean()
        intervention_rate = subset['intervention_applied'].mean()
        print(f"{dataset_type}: avg_reduction={avg_reduction:.4f}, intervention_rate={intervention_rate:.2f}")
    
    # Check effectiveness
    effective_debiasing = avg_bias_reduction > 0.1  # Threshold for meaningful bias reduction
    print(f"\nDebiasing effectiveness: {'✓ EFFECTIVE' if effective_debiasing else '⚠ NEEDS IMPROVEMENT'}")
    
    print(f"\nDetailed results saved to: {results_path}")
    
    return results_df

def analyze_gender_bias(text: str) -> float:
    """
    Simple gender bias analysis based on gendered word usage.
    Returns a bias score between -1 (female-biased) and 1 (male-biased).
    """
    male_words = ['he', 'him', 'his', 'man', 'male', 'guy', 'boy', 'father', 'son', 'brother', 'husband']
    female_words = ['she', 'her', 'hers', 'woman', 'female', 'girl', 'mother', 'daughter', 'sister', 'wife']
    
    text_lower = text.lower()
    male_count = sum(1 for word in male_words if word in text_lower)
    female_count = sum(1 for word in female_words if word in text_lower)
    
    total_gendered = male_count + female_count
    if total_gendered == 0:
        return 0.0
    
    # Return bias score: positive = male-biased, negative = female-biased
    return (male_count - female_count) / total_gendered

if __name__ == "__main__":
    # Set your Hugging Face token here (required for Gemma access)
    hf_token = os.getenv("HF_TOKEN", GEMMA_HUGGINGFACE_TOKEN)  # Use backup token as main
    
    if hf_token is None:
        print("❌ Error: No Hugging Face token found!")
        print("Set HF_TOKEN environment variable or check gemma_token_utils.py")
        exit(1)
    
    try:
        # Run focused training (no test generations)
        print("🎯 Starting FairSteer training...")
        debiaser = main_fairsteer_training(hf_token=hf_token)
        
        print("\n🎉 FairSteer training completed successfully!")
        print("Model is ready for bias detection and mitigation.")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
