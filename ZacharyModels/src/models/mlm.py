from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from .base import BaseModel

class MLMModel(BaseModel):
    """Implementation for Masked Language Models (BERT, RoBERTa)."""
    
    def __init__(self, model_path: str, max_length: int = 512):
        """Initialize MLM model.
        
        Args:
            model_path: HuggingFace model identifier
            max_length: Maximum sequence length for tokenization
        """
        super().__init__(model_path)
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        
    def load(self, token: Optional[str] = None) -> None:
        """Load model and tokenizer from HuggingFace.
        
        Args:
            token: Optional HuggingFace token for gated models
            
        Note: 
            Warnings about GenerationMixin can be safely ignored as
            this class does not use generation capabilities.
        """
        try:
            kwargs = {}
            if token:
                kwargs["token"] = token
                kwargs["trust_remote_code"] = True
                
            self.model = AutoModelForMaskedLM.from_pretrained(
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
            
        self.model.to(self.device)
        self.model.eval()
        
    def preprocess_text(self, text: str) -> Dict[str, Any]:
        """Tokenize and prepare text for MLM input.
        
        Args:
            text: Input text to process
            
        Returns:
            Dict containing input_ids, attention_mask, and token_type_ids
        """
        if not self.tokenizer:
            raise RuntimeError("Model and tokenizer must be loaded first")
            
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        return self.to_device(inputs)
        
    def compute_score(self, text: str) -> float:
        """Compute MLM score for a single text input.
        
        For MLMs, we:
        1. Mask each token one at a time
        2. Compute probability of original token being predicted
        3. Average these probabilities for final score
        
        Args:
            text: Input text to score
            
        Returns:
            float: Average MLM score
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model and tokenizer must be loaded first")
            
        inputs = self.preprocess_text(text)
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        
        # Store original tokens to compute their probabilities
        original_tokens = input_ids.clone()
        mask_token_id = self.tokenizer.mask_token_id
        
        total_score = 0.0
        count = 0
        
        # Iterate through each position
        for pos in range(len(input_ids)):
            if attention_mask[pos] == 0:  # Skip padding tokens
                continue
                
            # Create masked version of input
            masked_input_ids = input_ids.clone()
            masked_input_ids[pos] = mask_token_id
            
            with torch.no_grad():
                outputs = self.model(
                    masked_input_ids.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0)
                )
                
            # Get probability of original token
            logits = outputs.logits[0, pos]
            probs = torch.softmax(logits, dim=0)
            original_token_prob = probs[original_tokens[pos]].item()
            
            total_score += original_token_prob
            count += 1
            
        return total_score / count if count > 0 else 0.0
        
    def batch_compute_scores(self, texts: List[str]) -> List[float]:
        """Compute MLM scores for a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List[float]: MLM scores for each input text
        """
        return [self.compute_score(text) for text in texts]
