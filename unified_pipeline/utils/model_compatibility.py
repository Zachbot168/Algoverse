#!/usr/bin/env python3
"""
Model Compatibility Handler for Unified Bias Evaluation

Handles different model architectures and their unique requirements for
bias evaluation. Ensures consistent behavior across different model types.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoModel,
    PreTrainedModel, PreTrainedTokenizer,
    GPT2LMHeadModel, GPT2Tokenizer,
    LlamaForCausalLM, LlamaTokenizer,
    BertModel, BertTokenizer,
    RobertaModel, RobertaTokenizer
)
import warnings

warnings.filterwarnings('ignore')


class ModelCompatibilityHandler:
    """
    Handles compatibility across different model architectures for bias evaluation.
    
    Different models have different:
    - Tokenization schemes (pad tokens, special tokens)
    - Generation capabilities
    - Hidden state access
    - Input/output formats
    - Device handling
    """
    
    def __init__(self, model, tokenizer):
        """
        Initialize compatibility handler.
        
        Args:
            model: Pre-trained model instance
            tokenizer: Pre-trained tokenizer instance
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = self._detect_model_type()
        self.device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'
        
        # Setup model-specific configurations
        self._setup_tokenizer()
        self._setup_generation_config()
        
        print(f"Initialized compatibility handler for {self.model_type}")
    
    def _detect_model_type(self) -> str:
        """Detect the model architecture type."""
        model_class = type(self.model).__name__
        model_name = getattr(self.model, 'name_or_path', str(self.model))
        
        # Detect based on class name
        if 'GPT2' in model_class or 'gpt2' in model_name.lower():
            return 'gpt2'
        elif 'Llama' in model_class or 'llama' in model_name.lower():
            return 'llama'
        elif 'Gemma' in model_class or 'gemma' in model_name.lower():
            return 'gemma'
        elif 'Bert' in model_class or 'bert' in model_name.lower():
            return 'bert'
        elif 'Roberta' in model_class or 'roberta' in model_name.lower():
            return 'roberta'
        elif 'T5' in model_class or 't5' in model_name.lower():
            return 't5'
        elif 'Mistral' in model_class or 'mistral' in model_name.lower():
            return 'mistral'
        elif 'Qwen' in model_class or 'qwen' in model_name.lower():
            return 'qwen'
        else:
            return 'unknown'
    
    def _setup_tokenizer(self):
        """Setup tokenizer for consistent behavior across models."""
        # Handle pad token
        if hasattr(self.tokenizer, 'pad_token') and self.tokenizer.pad_token is None:
            if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif hasattr(self.tokenizer, 'unk_token') and self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            elif hasattr(self.tokenizer, 'add_special_tokens'):
                # Add a pad token if none exists
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                if hasattr(self.model, 'resize_token_embeddings'):
                    self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Set padding side based on model type
        if self.model_type in ['gpt2', 'llama', 'gemma', 'mistral', 'qwen']:
            # Causal LMs typically pad left for generation
            self.tokenizer.padding_side = 'left'
        else:
            # Encoder models typically pad right
            self.tokenizer.padding_side = 'right'
    
    def _setup_generation_config(self):
        """Setup generation configuration for different models."""
        self.generation_config = {
            'max_length': 512,
            'max_new_tokens': 100,
            'temperature': 0.7,
            'do_sample': True,
            'top_p': 0.9,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }
        
        # Model-specific adjustments
        if self.model_type == 'gpt2':
            self.generation_config.update({
                'max_new_tokens': 50,  # GPT2 can be verbose
                'repetition_penalty': 1.1
            })
        elif self.model_type in ['llama', 'mistral']:
            self.generation_config.update({
                'max_new_tokens': 100,
                'temperature': 0.8
            })
        elif self.model_type == 'gemma':
            self.generation_config.update({
                'max_new_tokens': 80,
                'temperature': 0.7
            })
    
    def tokenize_input(
        self, 
        text: Union[str, List[str]], 
        max_length: int = 512,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize input text with model-specific handling.
        
        Args:
            text: Input text(s) to tokenize
            max_length: Maximum sequence length
            return_tensors: Format for returned tensors
            
        Returns:
            Tokenized inputs ready for model
        """
        try:
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors=return_tensors
            )
            
            # Move to model device
            if return_tensors == "pt":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            return inputs
            
        except Exception as e:
            print(f"Tokenization error for {self.model_type}: {e}")
            # Fallback tokenization
            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_tensors=return_tensors
            )
            if return_tensors == "pt":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            return inputs
    
    def generate_text(
        self, 
        prompt: str, 
        max_new_tokens: Optional[int] = None,
        **generation_kwargs
    ) -> str:
        """
        Generate text with model-specific handling.
        
        Args:
            prompt: Input prompt for generation
            max_new_tokens: Maximum new tokens to generate
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated text (only the new part, not including prompt)
        """
        if not self.supports_generation():
            return f"Generation not supported for {self.model_type}"
        
        try:
            # Tokenize input
            inputs = self.tokenize_input(prompt)
            input_length = inputs['input_ids'].shape[1]
            
            # Update generation config
            gen_config = self.generation_config.copy()
            if max_new_tokens is not None:
                gen_config['max_new_tokens'] = max_new_tokens
            gen_config.update(generation_kwargs)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    **gen_config
                )
            
            # Decode only the new tokens
            generated_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return generated_text.strip()
            
        except Exception as e:
            print(f"Generation error for {self.model_type}: {e}")
            return f"Generation failed: {str(e)}"
    
    def get_logits(
        self, 
        text: str, 
        return_full_sequence: bool = False
    ) -> torch.Tensor:
        """
        Get model logits for input text.
        
        Args:
            text: Input text
            return_full_sequence: Whether to return logits for full sequence
            
        Returns:
            Logits tensor
        """
        try:
            inputs = self.tokenize_input(text)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif hasattr(outputs, 'prediction_scores'):
                    logits = outputs.prediction_scores
                else:
                    raise ValueError(f"Cannot extract logits for {self.model_type}")
                
                if return_full_sequence:
                    return logits
                else:
                    # Return logits for last token
                    return logits[:, -1, :]
                    
        except Exception as e:
            print(f"Logits extraction error for {self.model_type}: {e}")
            return torch.zeros((1, self.tokenizer.vocab_size))
    
    def get_embeddings(
        self, 
        text: str, 
        layer: int = -1
    ) -> torch.Tensor:
        """
        Get hidden state embeddings from specified layer.
        
        Args:
            text: Input text
            layer: Layer index (-1 for last layer)
            
        Returns:
            Hidden state embeddings
        """
        try:
            inputs = self.tokenize_input(text)
            
            with torch.no_grad():
                if self.model_type in ['bert', 'roberta']:
                    outputs = self.model(**inputs, output_hidden_states=True)
                    embeddings = outputs.hidden_states[layer]
                elif self.model_type in ['gpt2', 'llama', 'gemma', 'mistral', 'qwen']:
                    outputs = self.model(**inputs, output_hidden_states=True)
                    embeddings = outputs.hidden_states[layer]
                else:
                    # Fallback: try to get last hidden state
                    outputs = self.model(**inputs)
                    if hasattr(outputs, 'last_hidden_state'):
                        embeddings = outputs.last_hidden_state
                    else:
                        raise ValueError(f"Cannot extract embeddings for {self.model_type}")
                
                # Return mean pooled embeddings for simplicity
                return embeddings.mean(dim=1)
                
        except Exception as e:
            print(f"Embedding extraction error for {self.model_type}: {e}")
            # Return zero embeddings as fallback
            hidden_size = getattr(self.model.config, 'hidden_size', 768)
            return torch.zeros((1, hidden_size))
    
    def compute_perplexity(self, text: str) -> float:
        """
        Compute perplexity of text under the model.
        
        Args:
            text: Input text
            
        Returns:
            Perplexity score
        """
        if not self.supports_generation():
            return float('inf')
        
        try:
            inputs = self.tokenize_input(text)
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                
                if hasattr(outputs, 'loss'):
                    loss = outputs.loss
                else:
                    # Calculate loss manually
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = inputs['input_ids'][..., 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                
                perplexity = torch.exp(loss).item()
                return perplexity
                
        except Exception as e:
            print(f"Perplexity computation error for {self.model_type}: {e}")
            return float('inf')
    
    def supports_generation(self) -> bool:
        """Check if model supports text generation."""
        return hasattr(self.model, 'generate') and self.model_type in [
            'gpt2', 'llama', 'gemma', 'mistral', 'qwen', 't5'
        ]
    
    def supports_masked_lm(self) -> bool:
        """Check if model supports masked language modeling."""
        return self.model_type in ['bert', 'roberta']
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        info = {
            'model_type': self.model_type,
            'model_class': type(self.model).__name__,
            'tokenizer_class': type(self.tokenizer).__name__,
            'vocab_size': self.tokenizer.vocab_size,
            'supports_generation': self.supports_generation(),
            'supports_masked_lm': self.supports_masked_lm(),
            'device': str(self.device),
            'pad_token': self.tokenizer.pad_token,
            'eos_token': self.tokenizer.eos_token,
            'padding_side': self.tokenizer.padding_side
        }
        
        # Add model-specific parameters
        if hasattr(self.model, 'config'):
            config = self.model.config
            info.update({
                'hidden_size': getattr(config, 'hidden_size', None),
                'num_layers': getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', None)),
                'num_attention_heads': getattr(config, 'num_attention_heads', getattr(config, 'n_head', None)),
                'max_position_embeddings': getattr(config, 'max_position_embeddings', getattr(config, 'n_positions', None))
            })
        
        return info
    
    def evaluate_bias_sample(
        self, 
        sample: Dict[str, Any], 
        evaluation_mode: str = "classification"
    ) -> Any:
        """
        Evaluate a bias sample with model-specific handling.
        
        Args:
            sample: Sample dictionary with text, choices, etc.
            evaluation_mode: Type of evaluation to perform
            
        Returns:
            Model prediction
        """
        text = sample.get('text', '')
        
        try:
            if evaluation_mode == "generation":
                return self.generate_text(text, max_new_tokens=50)
            
            elif evaluation_mode == "multiple_choice":
                choices = sample.get('metadata', {}).get('choices', [])
                if choices:
                    return self._evaluate_multiple_choice(text, choices)
                else:
                    return self._classify_text(text)
            
            elif evaluation_mode == "classification":
                return self._classify_text(text)
            
            elif evaluation_mode == "perplexity":
                return self.compute_perplexity(text)
            
            else:
                # Default classification
                return self._classify_text(text)
                
        except Exception as e:
            print(f"Evaluation error for {self.model_type}: {e}")
            return f"Error: {str(e)}"
    
    def _evaluate_multiple_choice(self, text: str, choices: List[str]) -> str:
        """Evaluate multiple choice question."""
        best_choice = ""
        best_score = float('-inf')
        
        for choice in choices:
            full_text = f"{text} {choice}"
            
            if self.supports_generation():
                # For generative models, use perplexity
                score = -self.compute_perplexity(full_text)  # Lower perplexity = higher score
            else:
                # For other models, use logits
                logits = self.get_logits(full_text)
                score = logits.max().item()
            
            if score > best_score:
                best_score = score
                best_choice = choice
        
        return best_choice if best_choice else (choices[0] if choices else "No choice")
    
    def _classify_text(self, text: str) -> Union[str, int]:
        """Classify text using model-specific approach."""
        if self.supports_generation():
            # For generative models, generate a classification
            prompt = f"Classify the following text as positive or negative: {text}\nClassification:"
            result = self.generate_text(prompt, max_new_tokens=10)
            return result.lower().strip()
        else:
            # For encoder models, use logits
            logits = self.get_logits(text)
            return torch.argmax(logits, dim=-1).item()


def create_compatible_model_handler(model_name: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer, ModelCompatibilityHandler]:
    """
    Create a model handler with compatibility support.
    
    Args:
        model_name: Name or path of the model
        
    Returns:
        Tuple of (model, tokenizer, compatibility_handler)
    """
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with appropriate settings
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Create compatibility handler
        handler = ModelCompatibilityHandler(model, tokenizer)
        
        return model, tokenizer, handler
        
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        raise


def test_model_compatibility(model_name: str) -> Dict[str, Any]:
    """
    Test model compatibility and return compatibility report.
    
    Args:
        model_name: Name of model to test
        
    Returns:
        Compatibility test results
    """
    try:
        model, tokenizer, handler = create_compatible_model_handler(model_name)
        
        # Test basic functionality
        test_text = "The weather is nice today."
        
        results = {
            'model_name': model_name,
            'model_info': handler.get_model_info(),
            'tests': {}
        }
        
        # Test tokenization
        try:
            inputs = handler.tokenize_input(test_text)
            results['tests']['tokenization'] = {'status': 'pass', 'input_shape': inputs['input_ids'].shape}
        except Exception as e:
            results['tests']['tokenization'] = {'status': 'fail', 'error': str(e)}
        
        # Test generation (if supported)
        if handler.supports_generation():
            try:
                generated = handler.generate_text(test_text, max_new_tokens=10)
                results['tests']['generation'] = {'status': 'pass', 'sample_output': generated[:50]}
            except Exception as e:
                results['tests']['generation'] = {'status': 'fail', 'error': str(e)}
        
        # Test logits extraction
        try:
            logits = handler.get_logits(test_text)
            results['tests']['logits'] = {'status': 'pass', 'logits_shape': logits.shape}
        except Exception as e:
            results['tests']['logits'] = {'status': 'fail', 'error': str(e)}
        
        # Test embeddings extraction
        try:
            embeddings = handler.get_embeddings(test_text)
            results['tests']['embeddings'] = {'status': 'pass', 'embeddings_shape': embeddings.shape}
        except Exception as e:
            results['tests']['embeddings'] = {'status': 'fail', 'error': str(e)}
        
        return results
        
    except Exception as e:
        return {
            'model_name': model_name,
            'status': 'failed_to_load',
            'error': str(e)
        }


if __name__ == "__main__":
    # Test compatibility with common models
    test_models = [
        "gpt2",
        "distilgpt2", 
        "bert-base-uncased",
        "roberta-base"
    ]
    
    print("Model Compatibility Test Report")
    print("=" * 50)
    
    for model_name in test_models:
        print(f"\nTesting {model_name}...")
        try:
            results = test_model_compatibility(model_name)
            print(f"Model Type: {results.get('model_info', {}).get('model_type', 'unknown')}")
            print(f"Supports Generation: {results.get('model_info', {}).get('supports_generation', False)}")
            
            tests = results.get('tests', {})
            for test_name, test_result in tests.items():
                status = test_result.get('status', 'unknown')
                print(f"  {test_name}: {status}")
                
        except Exception as e:
            print(f"  Error testing {model_name}: {e}")
        
        print("-" * 30)