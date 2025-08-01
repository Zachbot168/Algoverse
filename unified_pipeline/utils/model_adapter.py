#!/usr/bin/env python3
"""
Model Architecture Adapter

Provides unified interface for working with different model architectures
(BERT, RoBERTa, GPT-2, Gemma, Llama) in the bias mitigation pipeline.

This adapter handles architecture-specific differences in:
- Layer access patterns
- Attention head structures  
- Hook registration
- Token handling
- Generation vs. encoding
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoModelForCausalLM, AutoTokenizer,
    BertModel, RobertaModel, GPT2LMHeadModel,
    LlamaForCausalLM, GemmaForCausalLM
)


class ModelArchitectureInfo:
    """Container for model architecture information."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = getattr(model, 'name_or_path', 'unknown')
        
        # Detect architecture
        self.architecture = self._detect_architecture()
        self.model_type = self._detect_model_type()
        
        # Get layer structure
        self.num_layers = self._get_num_layers()
        self.hidden_size = self._get_hidden_size()
        self.num_heads = self._get_num_heads()
        self.head_dim = self.hidden_size // self.num_heads if self.num_heads > 0 else 0
        
        # Get layer access patterns
        self.layer_access_pattern = self._get_layer_access_pattern()
        self.attention_access_pattern = self._get_attention_access_pattern()
        
        print(f"Detected architecture: {self.architecture}")
        print(f"Model type: {self.model_type}")
        print(f"Layers: {self.num_layers}, Hidden size: {self.hidden_size}, Heads: {self.num_heads}")
    
    def _detect_architecture(self) -> str:
        """Detect the model architecture."""
        model_class = self.model.__class__.__name__
        
        if 'Bert' in model_class:
            return 'bert'
        elif 'Roberta' in model_class or 'RoBERTa' in model_class:
            return 'roberta'
        elif 'GPT2' in model_class:
            return 'gpt2'
        elif 'Llama' in model_class:
            return 'llama'
        elif 'Gemma' in model_class:
            return 'gemma'
        else:
            # Try to detect from model name
            model_name = self.model_name.lower()
            if 'bert' in model_name:
                return 'bert'
            elif 'roberta' in model_name:
                return 'roberta'
            elif 'gpt2' in model_name:
                return 'gpt2'
            elif 'llama' in model_name:
                return 'llama'
            elif 'gemma' in model_name:
                return 'gemma'
            else:
                return 'unknown'
    
    def _detect_model_type(self) -> str:
        """Detect if model is encoder, decoder, or encoder-decoder."""
        if hasattr(self.model, 'generate'):
            return 'decoder'  # Can generate text
        else:
            return 'encoder'  # Encoder-only (BERT, RoBERTa)
    
    def _get_num_layers(self) -> int:
        """Get number of transformer layers."""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Llama/Gemma style: model.model.layers
            return len(self.model.model.layers)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT-2 style: model.transformer.h
            return len(self.model.transformer.h)
        elif hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            # BERT style: model.encoder.layer
            return len(self.model.encoder.layer)
        elif hasattr(self.model, 'roberta') and hasattr(self.model.roberta.encoder, 'layer'):
            # RoBERTa style: model.roberta.encoder.layer
            return len(self.model.roberta.encoder.layer)
        else:
            # Fallback to config
            return getattr(self.model.config, 'num_hidden_layers', 12)
    
    def _get_hidden_size(self) -> int:
        """Get hidden size."""
        return getattr(self.model.config, 'hidden_size', 768)
    
    def _get_num_heads(self) -> int:
        """Get number of attention heads."""
        return getattr(self.model.config, 'num_attention_heads', 12)
    
    def _get_layer_access_pattern(self) -> str:
        """Get the pattern for accessing layers."""
        if self.architecture in ['llama', 'gemma']:
            return 'model.model.layers'
        elif self.architecture == 'gpt2':
            return 'transformer.h'
        elif self.architecture == 'bert':
            return 'encoder.layer'
        elif self.architecture == 'roberta':
            return 'roberta.encoder.layer'
        else:
            return 'unknown'
    
    def _get_attention_access_pattern(self) -> str:
        """Get the pattern for accessing attention layers."""
        if self.architecture in ['llama', 'gemma']:
            return 'self_attn'
        elif self.architecture == 'gpt2':
            return 'attn'
        elif self.architecture in ['bert', 'roberta']:
            return 'attention.self'
        else:
            return 'unknown'


class UniversalModelAdapter:
    """Adapter that provides unified interface for all model architectures."""
    
    def __init__(self, model, tokenizer):
        """Initialize adapter with model and tokenizer."""
        self.model = model
        self.tokenizer = tokenizer
        self.arch_info = ModelArchitectureInfo(model, tokenizer)
        
        # Setup tokenizer
        self._setup_tokenizer()
    
    def _setup_tokenizer(self):
        """Setup tokenizer with proper padding token."""
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                # Add a pad token
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model.resize_token_embeddings(len(self.tokenizer))
    
    def get_layer(self, layer_idx: int) -> nn.Module:
        """Get a specific transformer layer."""
        if self.arch_info.layer_access_pattern == 'model.model.layers':
            return self.model.model.layers[layer_idx]
        elif self.arch_info.layer_access_pattern == 'transformer.h':
            return self.model.transformer.h[layer_idx]
        elif self.arch_info.layer_access_pattern == 'encoder.layer':
            return self.model.encoder.layer[layer_idx]
        elif self.arch_info.layer_access_pattern == 'roberta.encoder.layer':
            return self.model.roberta.encoder.layer[layer_idx]
        else:
            raise ValueError(f"Unknown layer access pattern: {self.arch_info.layer_access_pattern}")
    
    def get_attention_layer(self, layer_idx: int) -> nn.Module:
        """Get the attention sublayer of a specific layer."""
        layer = self.get_layer(layer_idx)
        
        if self.arch_info.attention_access_pattern == 'self_attn':
            return layer.self_attn
        elif self.arch_info.attention_access_pattern == 'attn':
            return layer.attn
        elif self.arch_info.attention_access_pattern == 'attention.self':
            return layer.attention.self
        else:
            raise ValueError(f"Unknown attention access pattern: {self.arch_info.attention_access_pattern}")
    
    def get_mlp_layer(self, layer_idx: int) -> nn.Module:
        """Get the MLP/feed-forward sublayer of a specific layer."""
        layer = self.get_layer(layer_idx)
        
        # Different architectures have different MLP naming
        if hasattr(layer, 'mlp'):  # Llama, Gemma
            return layer.mlp
        elif hasattr(layer, 'feed_forward'):  # Some transformers
            return layer.feed_forward
        elif hasattr(layer, 'intermediate'):  # BERT, RoBERTa
            return layer.intermediate
        else:
            # Try to find MLP-like layer
            for name, module in layer.named_children():
                if 'mlp' in name.lower() or 'feed' in name.lower() or 'intermediate' in name.lower():
                    return module
            raise ValueError(f"Could not find MLP layer in layer {layer_idx}")
    
    def register_layer_hook(self, layer_idx: int, hook_fn) -> torch.utils.hooks.RemovableHandle:
        """Register a forward hook on a specific layer."""
        layer = self.get_layer(layer_idx)
        return layer.register_forward_hook(hook_fn)
    
    def register_attention_hook(self, layer_idx: int, hook_fn) -> torch.utils.hooks.RemovableHandle:
        """Register a forward hook on a specific attention layer."""
        attention_layer = self.get_attention_layer(layer_idx)
        return attention_layer.register_forward_hook(hook_fn)
    
    def register_mlp_hook(self, layer_idx: int, hook_fn) -> torch.utils.hooks.RemovableHandle:
        """Register a forward hook on a specific MLP layer."""
        mlp_layer = self.get_mlp_layer(layer_idx)
        return mlp_layer.register_forward_hook(hook_fn)
    
    def extract_last_token_activations(self, outputs, attention_mask=None) -> torch.Tensor:
        """Extract last token activations, handling different architectures."""
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs
        
        if attention_mask is not None:
            # Get the last non-padding token for each sequence
            batch_size = hidden_states.size(0)
            last_token_indices = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
            
            # Extract last token activations
            last_token_activations = hidden_states[
                torch.arange(batch_size), last_token_indices
            ]
        else:
            # Just use the last token
            last_token_activations = hidden_states[:, -1, :]
        
        return last_token_activations
    
    def prepare_inputs(self, texts: List[str], max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Prepare inputs for the model, handling architecture differences."""
        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        # Move to model device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        return inputs
    
    def forward_pass(self, inputs: Dict[str, torch.Tensor], 
                    output_hidden_states: bool = True) -> Dict[str, Any]:
        """Run forward pass, handling different model types."""
        with torch.no_grad():
            if self.arch_info.model_type == 'encoder':
                # BERT, RoBERTa - encoder only
                outputs = self.model(**inputs, output_hidden_states=output_hidden_states)
                return {
                    'last_hidden_state': outputs.last_hidden_state,
                    'hidden_states': outputs.hidden_states if output_hidden_states else None,
                    'attention_mask': inputs.get('attention_mask')
                }
            else:
                # Decoder models (GPT-2, Llama, Gemma)
                outputs = self.model(**inputs, output_hidden_states=output_hidden_states)
                return {
                    'last_hidden_state': outputs.last_hidden_state,
                    'hidden_states': outputs.hidden_states if output_hidden_states else None,
                    'attention_mask': inputs.get('attention_mask'),
                    'logits': outputs.logits
                }
    
    def generate_text(self, prompt: str, max_new_tokens: int = 50, **generation_kwargs) -> str:
        """Generate text (only for decoder models)."""
        if self.arch_info.model_type != 'decoder':
            raise ValueError(f"Text generation not supported for {self.arch_info.model_type} models")
        
        inputs = self.prepare_inputs([prompt])
        
        # Default generation parameters
        gen_kwargs = {
            'max_new_tokens': max_new_tokens,
            'do_sample': True,
            'temperature': 0.7,
            'pad_token_id': self.tokenizer.pad_token_id,
            **generation_kwargs
        }
        
        with torch.no_grad():
            outputs = self.model.generate(inputs['input_ids'], **gen_kwargs)
        
        # Decode only the new tokens
        new_tokens = outputs[0][inputs['input_ids'].size(1):]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return generated_text
    
    def get_lora_target_modules(self) -> List[str]:
        """Get appropriate LoRA target modules for this architecture."""
        if self.arch_info.architecture in ['llama', 'gemma']:
            return ['q_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        elif self.arch_info.architecture == 'gpt2':
            return ['c_attn', 'c_proj', 'c_fc']
        elif self.arch_info.architecture in ['bert', 'roberta']:
            return ['query', 'value', 'key', 'dense']
        else:
            # Safe default
            return ['q_proj', 'v_proj']
    
    def get_component_path(self, layer_idx: int, component_type: str, 
                          head_idx: Optional[int] = None) -> str:
        """Get the full path to a specific component for LoRA targeting."""
        base_pattern = self.arch_info.layer_access_pattern.replace('.', '.')
        
        if component_type == 'attention':
            if self.arch_info.architecture in ['llama', 'gemma']:
                return f"{base_pattern}.{layer_idx}.self_attn"
            elif self.arch_info.architecture == 'gpt2':
                return f"{base_pattern}.{layer_idx}.attn"
            elif self.arch_info.architecture in ['bert', 'roberta']:
                return f"{base_pattern}.{layer_idx}.attention.self"
        elif component_type == 'mlp':
            if self.arch_info.architecture in ['llama', 'gemma']:
                return f"{base_pattern}.{layer_idx}.mlp"
            elif self.arch_info.architecture == 'gpt2':
                return f"{base_pattern}.{layer_idx}.mlp"
            elif self.arch_info.architecture in ['bert', 'roberta']:
                return f"{base_pattern}.{layer_idx}.intermediate"
        
        return f"{base_pattern}.{layer_idx}"
    
    def is_compatible_with_pipeline(self) -> Tuple[bool, List[str]]:
        """Check if model is compatible with the unified pipeline."""
        issues = []
        
        # Check if we can detect architecture
        if self.arch_info.architecture == 'unknown':
            issues.append("Unknown architecture - may have limited support")
        
        # Check if we can access layers
        try:
            _ = self.get_layer(0)
        except Exception as e:
            issues.append(f"Cannot access layers: {e}")
        
        # Check if we can access attention
        try:
            _ = self.get_attention_layer(0)
        except Exception as e:
            issues.append(f"Cannot access attention layers: {e}")
        
        # Check tokenizer
        if self.tokenizer.pad_token is None:
            issues.append("No pad token (will be auto-fixed)")
        
        # For bias mitigation, we need either generation or masking capability
        if self.arch_info.model_type == 'encoder':
            if not hasattr(self.model, 'cls'):
                issues.append("Encoder model without classification head")
        
        return len(issues) == 0, issues


def create_model_adapter(model_name: str, model=None, tokenizer=None, 
                        device: str = "auto", **model_kwargs) -> UniversalModelAdapter:
    """
    Create a model adapter for any supported architecture.
    
    Args:
        model_name: HuggingFace model name or path
        model: Pre-loaded model (optional)
        tokenizer: Pre-loaded tokenizer (optional)
        device: Device to load model on
        **model_kwargs: Additional arguments for model loading
        
    Returns:
        UniversalModelAdapter instance
    """
    # Setup device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # Load tokenizer if not provided
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model if not provided
    if model is None:
        try:
            # Try as causal LM first (for decoder models)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if device == "cuda" else None,
                **model_kwargs
            )
            if device != "cuda":
                model = model.to(device)
        except:
            # Fall back to encoder model
            model = AutoModel.from_pretrained(
                model_name,
                device_map="auto" if device == "cuda" else None,
                **model_kwargs
            )
            if device != "cuda":
                model = model.to(device)
    
    # Create adapter
    adapter = UniversalModelAdapter(model, tokenizer)
    
    # Check compatibility
    is_compatible, issues = adapter.is_compatible_with_pipeline()
    if not is_compatible:
        print(f"⚠️  Model compatibility issues detected:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print(f"✅ Model {model_name} is fully compatible with the unified pipeline")
    
    return adapter


# Convenience functions for specific architectures
def load_bert_model(model_name: str = "bert-base-uncased", **kwargs) -> UniversalModelAdapter:
    """Load BERT model with adapter."""
    return create_model_adapter(model_name, **kwargs)

def load_roberta_model(model_name: str = "roberta-base", **kwargs) -> UniversalModelAdapter:
    """Load RoBERTa model with adapter."""
    return create_model_adapter(model_name, **kwargs)

def load_gpt2_model(model_name: str = "gpt2", **kwargs) -> UniversalModelAdapter:
    """Load GPT-2 model with adapter."""
    return create_model_adapter(model_name, **kwargs)

def load_llama_model(model_name: str = "meta-llama/Llama-3.2-1B", **kwargs) -> UniversalModelAdapter:
    """Load Llama model with adapter."""
    return create_model_adapter(model_name, **kwargs)

def load_gemma_model(model_name: str = "google/gemma-2-2b-it", **kwargs) -> UniversalModelAdapter:
    """Load Gemma model with adapter."""
    return create_model_adapter(model_name, **kwargs)


if __name__ == "__main__":
    # Test with a simple model
    try:
        print("Testing model adapter with GPT-2...")
        adapter = load_gpt2_model()
        
        print(f"Architecture: {adapter.arch_info.architecture}")
        print(f"Model type: {adapter.arch_info.model_type}")  
        print(f"Layers: {adapter.arch_info.num_layers}")
        
        # Test text generation
        if adapter.arch_info.model_type == 'decoder':
            generated = adapter.generate_text("The future of AI is", max_new_tokens=10)
            print(f"Generated: {generated}")
        
        print("✅ Model adapter test passed!")
        
    except Exception as e:
        print(f"❌ Model adapter test failed: {e}")