from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any
import torch
from torch import Tensor

class BaseModel(ABC):
    """Abstract base class for all models (MLM and LLM)."""
    
    def __init__(self, model_path: str):
        """Initialize base model.
        
        Args:
            model_path: Path or HuggingFace model identifier
        """
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @abstractmethod
    def load(self, token: Optional[str] = None) -> None:
        """Load model and tokenizer.
        
        Args:
            token: Optional HuggingFace token for gated models
        """
        pass
    
    @abstractmethod
    def compute_score(self, text: str) -> float:
        """Compute likelihood/probability score for text.
        
        Args:
            text: Input text to score
            
        Returns:
            float: Score for the input text
        """
        pass
    
    @abstractmethod
    def batch_compute_scores(self, texts: List[str]) -> List[float]:
        """Compute scores for a batch of texts.
        
        Args:
            texts: List of input texts to score
            
        Returns:
            List[float]: Scores for each input text
        """
        pass
    
    def to_device(self, tensor: Union[Tensor, Dict[str, Tensor]]) -> Union[Tensor, Dict[str, Tensor]]:
        """Helper to move tensors to correct device."""
        if isinstance(tensor, dict):
            return {k: v.to(self.device) for k, v in tensor.items()}
        return tensor.to(self.device)

    @abstractmethod
    def preprocess_text(self, text: str) -> Dict[str, Any]:
        """Preprocess text for model input.
        
        Args:
            text: Raw input text
            
        Returns:
            Dict containing processed inputs
        """
        pass
