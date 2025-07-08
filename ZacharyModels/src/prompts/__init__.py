"""Prompt management system for model inputs."""

from .templates import PromptTemplate, PromptLibrary
from .wrapper import PromptWrapper

def load_template(name: str, model_type: str) -> PromptTemplate:
    """Convenience function to load a prompt template.
    
    Args:
        name: Template name
        model_type: Model type identifier
        
    Returns:
        PromptTemplate: Loaded template
    """
    return PromptLibrary.get_template(name, model_type)

def list_templates(model_type: str) -> list:
    """List available templates for model type.
    
    Args:
        model_type: Model type identifier
        
    Returns:
        list: Available template names
    """
    return PromptLibrary.list_templates(model_type)

__all__ = [
    "PromptTemplate",
    "PromptLibrary",
    "PromptWrapper",
    "load_template",
    "list_templates"
]
