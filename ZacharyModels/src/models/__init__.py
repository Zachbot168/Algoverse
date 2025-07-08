from typing import Dict, Any, Optional
from .base import BaseModel
from .mlm import MLMModel
from .llm import LLMModel

class ModelFactory:
    """Factory class for creating model instances."""
    
    # Model type mapping
    MLM_MODELS = {
        "bert-base-uncased",
        "bert-large-uncased",
        "roberta-base"
    }
    
    INSTRUCTION_MODELS = {
        "Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B-Instruct",
        "gemma-2-2b-it",
        "gemma-3-1b-it",
        "llama-3.2-1b-it"
    }
    
    LLM_MODELS = {
        "gpt2",
        "gpt2-medium",
        "Qwen3-0.6B",
        "llama-3b",
        "llama-3.2-1b"
    }.union(INSTRUCTION_MODELS)
    
    @classmethod
    def create(
        cls,
        model_name: str,
        max_length: int = 512,
        instruction_template: Optional[str] = None,
        **kwargs
    ) -> BaseModel:
        """Create a model instance based on model name.
        
        Args:
            model_name: Name/path of the model
            max_length: Maximum sequence length
            instruction_template: Custom instruction template (if applicable)
            **kwargs: Additional arguments to pass to model constructor
            
        Returns:
            BaseModel: Instantiated model of appropriate type
            
        Raises:
            ValueError: If model_name is not recognized
        """
        if model_name in cls.MLM_MODELS:
            return MLMModel(
                model_path=model_name,
                max_length=max_length,
                **kwargs
            )
        elif model_name in cls.LLM_MODELS:
            return LLMModel(
                model_path=model_name,
                max_length=max_length,
                is_instruction_model=model_name in cls.INSTRUCTION_MODELS,
                instruction_template=instruction_template,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

# Make factory accessible from models package
from .base import BaseModel
__all__ = ["ModelFactory", "BaseModel"]
