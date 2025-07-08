from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import BaseModel

class LLMModel(BaseModel):
    """Implementation for autoregressive language models (GPT, Qwen, Gemma, LLaMA)."""
    
    def __init__(
        self, 
        model_path: str, 
        max_length: int = 512,
        is_instruction_model: bool = False,
        instruction_template: Optional[str] = None
    ):
        """Initialize LLM model.
        
        Args:
            model_path: HuggingFace model identifier
            max_length: Maximum sequence length for tokenization
            is_instruction_model: Whether model is instruction-tuned
            instruction_template: Template for instruction format (if applicable)
        """
        super().__init__(model_path)
        self.max_length = max_length
        self.is_instruction_model = is_instruction_model
        self.instruction_template = instruction_template
        self.model = None
        self.tokenizer = None
        
    def load(self, token: Optional[str] = None) -> None:
        """Load model and tokenizer from HuggingFace.
        
        Args:
            token: Optional HuggingFace token for gated models
        """
        try:
            kwargs = {
                "torch_dtype": torch.float16,  # Use fp16 for efficiency
                "trust_remote_code": True,  # Required for some models
            }
            if token:
                kwargs["token"] = token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **kwargs
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                **kwargs
            )
        except OSError as e:
            if "is not a local folder and is not a valid model identifier" in str(e):
                print(f"Warning: Model {self.model_path} not found on HuggingFace. Skipping...")
                return
            raise
            
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.to(self.device)
        self.model.eval()
        
    def format_instruction(self, text: str) -> str:
        """Format text with instruction template if applicable.
        
        Args:
            text: Input text to format
            
        Returns:
            str: Formatted text with instruction template if applicable
        """
        if not self.is_instruction_model or not self.instruction_template:
            return text
            
        # Handle different instruction formats
        if "gemma" in self.model_path.lower():
            return f"<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"
        elif "qwen" in self.model_path.lower():
            return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        elif "llama" in self.model_path.lower():
            return f"[INST] {text} [/INST]"
        else:
            return self.instruction_template.format(text=text)
            
    def preprocess_text(self, text: str) -> Dict[str, Any]:
        """Tokenize and prepare text for LLM input.
        
        Args:
            text: Input text to process
            
        Returns:
            Dict containing input_ids and attention_mask
        """
        if not self.tokenizer:
            raise RuntimeError("Model and tokenizer must be loaded first")
            
        # Format text if instruction model
        formatted_text = self.format_instruction(text)
            
        inputs = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        return self.to_device(inputs)
        
    def compute_score(self, text: str) -> float:
        """Compute likelihood score for text.
        
        For autoregressive models, we:
        1. Get the log-likelihood of the full sequence
        2. Normalize by sequence length
        
        Args:
            text: Input text to score
            
        Returns:
            float: Normalized log-likelihood score
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model and tokenizer must be loaded first")
            
        inputs = self.preprocess_text(text)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Create shifted labels for causal LM
        labels = input_ids.clone()
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
        # Get loss and convert to per-token likelihood
        loss = outputs.loss.item()
        seq_length = attention_mask.sum().item()
        
        # Return negative loss (higher is better, like MLM probabilities)
        return -loss / seq_length
        
    def batch_compute_scores(self, texts: List[str]) -> List[float]:
        """Compute scores for a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List[float]: Scores for each input text
        """
        scores = []
        batch_size = 8  # Process in smaller batches to manage memory
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_scores = [self.compute_score(text) for text in batch_texts]
            scores.extend(batch_scores)
            
        return scores
