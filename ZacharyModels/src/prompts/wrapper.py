from typing import Dict, Any, Optional, List, Union
from .templates import PromptTemplate, PromptLibrary

class PromptWrapper:
    """Wrapper for dynamic prompt generation and formatting."""
    
    def __init__(
        self,
        model_type: str,
        template_name: str = "basic",
        custom_template: Optional[str] = None,
        few_shot_examples: Optional[List[Dict[str, str]]] = None
    ):
        """Initialize prompt wrapper.
        
        Args:
            model_type: Type of model ('mlm', 'llm', or specific instruction model)
            template_name: Name of template to use
            custom_template: Optional custom template string
            few_shot_examples: Optional examples for few-shot prompting
        """
        self.model_type = model_type
        
        if custom_template:
            self.template = PromptTemplate(
                name="custom",
                template=custom_template,
                description="Custom template",
                variables=["text"],
                examples=few_shot_examples
            )
        else:
            self.template = PromptLibrary.get_template(template_name, model_type)
            if few_shot_examples:
                self.template.examples = few_shot_examples
                
    def format_few_shot(self, variables: Dict[str, str]) -> str:
        """Format few-shot examples if available.
        
        Args:
            variables: Variables for template formatting
            
        Returns:
            str: Formatted few-shot examples
        """
        if not self.template.examples:
            return ""
            
        examples_text = []
        for example in self.template.examples:
            # Format each example using the same template
            formatted = self.template.template.format(**example)
            examples_text.append(formatted)
            
        return "\n\n".join(examples_text) + "\n\n"
        
    def wrap(
        self,
        text: Union[str, List[str]],
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Union[str, List[str]]:
        """Wrap text with prompt template.
        
        Args:
            text: Input text or list of texts
            additional_context: Optional additional variables for template
            
        Returns:
            Union[str, List[str]]: Wrapped text(s)
        """
        if isinstance(text, list):
            return [self.wrap(t, additional_context) for t in text]
            
        variables = {"text": text}
        if additional_context:
            variables.update(additional_context)
            
        # Add few-shot examples if available
        few_shot = self.format_few_shot(variables)
        
        # Format template with variables
        formatted = self.template.template.format(**variables)
        
        # Combine few-shot and formatted template
        return few_shot + formatted
        
    def generate_variants(
        self,
        text: str,
        num_variants: int = 1,
        temperature: float = 0.7
    ) -> List[str]:
        """Generate prompt variants for the same input.
        
        This is useful for testing different phrasings/framings.
        
        Args:
            text: Base input text
            num_variants: Number of variants to generate
            temperature: Sampling temperature for generation
            
        Returns:
            List[str]: List of prompt variants
        """
        if num_variants == 1:
            return [self.wrap(text)]
            
        templates = PromptLibrary.list_templates(self.model_type)
        variants = []
        
        # Use different templates if available
        for i in range(num_variants):
            template_name = templates[i % len(templates)]
            variant_wrapper = PromptWrapper(
                self.model_type,
                template_name=template_name
            )
            variants.append(variant_wrapper.wrap(text))
            
        return variants
        
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PromptWrapper":
        """Create wrapper from configuration dictionary.
        
        Args:
            config: Configuration dictionary with keys:
                - model_type: str
                - template_name: str (optional)
                - custom_template: str (optional)
                - few_shot_examples: List[Dict] (optional)
                
        Returns:
            PromptWrapper: Configured wrapper
        """
        return cls(
            model_type=config["model_type"],
            template_name=config.get("template_name", "basic"),
            custom_template=config.get("custom_template"),
            few_shot_examples=config.get("few_shot_examples")
        )
