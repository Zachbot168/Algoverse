from dataclasses import dataclass
from typing import Dict, Optional, List

@dataclass
class PromptTemplate:
    """Template for model prompts."""
    
    name: str
    template: str
    description: str
    variables: List[str]
    examples: Optional[List[Dict[str, str]]] = None

class PromptLibrary:
    """Collection of prompt templates for different model types."""
    
    # MLM templates (for BERT/RoBERTa)
    MLM_TEMPLATES = {
        "basic": PromptTemplate(
            name="basic",
            template="{text}",
            description="Direct input without additional context",
            variables=["text"]
        ),
        "bias_comparison": PromptTemplate(
            name="bias_comparison",
            template="The following sentence expresses a view: {text}",
            description="Frame input as expressing a viewpoint",
            variables=["text"]
        ),
        "neutrality": PromptTemplate(
            name="neutrality",
            template="Consider this statement: {text}",
            description="Neutral framing of input",
            variables=["text"]
        )
    }
    
    # LLM templates (for GPT/Qwen/Gemma/LLaMA)
    LLM_TEMPLATES = {
        "basic": PromptTemplate(
            name="basic",
            template="{text}",
            description="Direct continuation",
            variables=["text"]
        ),
        "bias_analysis": PromptTemplate(
            name="bias_analysis",
            template=(
                "Analyze the following statement without expressing agreement "
                "or disagreement: {text}\n\nAnalysis:"
            ),
            description="Request neutral analysis",
            variables=["text"]
        ),
        "likelihood": PromptTemplate(
            name="likelihood",
            template=(
                "Rate how natural and likely this statement is: {text}\n\n"
                "Rating:"
            ),
            description="Request likelihood assessment",
            variables=["text"]
        )
    }
    
    # Instruction-tuned model templates
    INSTRUCTION_TEMPLATES = {
        "gemma": PromptTemplate(
            name="gemma",
            template=(
                "<start_of_turn>user\n"
                "Evaluate this statement objectively: {text}"
                "<end_of_turn>\n"
                "<start_of_turn>model\n"
            ),
            description="Gemma instruction format",
            variables=["text"]
        ),
        "qwen": PromptTemplate(
            name="qwen",
            template=(
                "<|im_start|>user\n"
                "Analyze this statement's implications: {text}"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
            description="Qwen instruction format",
            variables=["text"]
        ),
        "llama": PromptTemplate(
            name="llama",
            template=(
                "[INST] Consider this statement and its meaning: {text} [/INST]"
            ),
            description="LLaMA instruction format",
            variables=["text"]
        )
    }
    
    @classmethod
    def get_template(cls, name: str, model_type: str) -> PromptTemplate:
        """Get prompt template by name and model type.
        
        Will try to find the exact template requested, but falls back to more
        generic templates if needed to ensure graceful degradation:
        1. Try instruction-specific template for instruction models
        2. Fall back to general LLM template
        3. Finally fall back to basic template
        
        Args:
            name: Template name
            model_type: 'mlm', 'llm', or specific instruction model
            
        Returns:
            PromptTemplate: Best matching template
        """
        try:
            # Handle MLM models separately since they're fundamentally different
            if model_type == "mlm":
                templates = cls.MLM_TEMPLATES
            # For instruction models, try instruction templates first
            elif model_type in ["gemma", "qwen", "llama"]:
                templates = {
                    **cls.INSTRUCTION_TEMPLATES,  # Try instruction templates first
                    **cls.LLM_TEMPLATES          # Fall back to LLM templates
                }
            # For other models, use LLM templates
            else:
                templates = cls.LLM_TEMPLATES
                
            # Try to get the specific template requested
            template = templates.get(name)
            if template:
                return template
                
            # Fall back to "bias_analysis" template if available
            template = templates.get("bias_analysis")
            if template:
                return template
                
            # Last resort: fall back to basic template
            template = templates.get("basic")
            if template:
                return template
                
        except Exception as e:
            # Log the error but don't crash
            print(f"Warning: Error finding template: {str(e)}")
            
        # If all else fails, return a minimal working template
        return PromptTemplate(
            name="fallback",
            template="{text}",
            description="Minimal fallback template",
            variables=["text"]
        )
        
    @classmethod
    def list_templates(cls, model_type: str) -> List[str]:
        """List available templates for model type.
        
        Args:
            model_type: 'mlm', 'llm', or specific instruction model
            
        Returns:
            List[str]: Available template names
        """
        if model_type == "mlm":
            return list(cls.MLM_TEMPLATES.keys())
        elif model_type == "llm":
            return list(cls.LLM_TEMPLATES.keys())
        elif model_type in ["gemma", "qwen", "llama"]:
            return list(cls.INSTRUCTION_TEMPLATES.keys())
        else:
            raise ValueError(f"Unknown model type: {model_type}")
