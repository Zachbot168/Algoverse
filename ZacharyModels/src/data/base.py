from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator, Tuple
import json
import pandas as pd

@dataclass
class Example:
    """Base class for dataset examples."""
    id: str
    text: str
    metadata: Dict[str, Any]

class BaseDataset(ABC):
    """Abstract base class for datasets."""
    
    def __init__(
        self,
        data_dir: str,
        cache_dir: Optional[str] = None,
        preprocess_fn: Optional[callable] = None
    ):
        """Initialize dataset.
        
        Args:
            data_dir: Directory containing dataset files
            cache_dir: Directory for caching processed data
            preprocess_fn: Optional function to preprocess examples
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.preprocess_fn = preprocess_fn
        self._examples: List[Example] = []
        self._metadata: Dict[str, Any] = {}
        
    @abstractmethod
    def load(self) -> None:
        """Load dataset from disk."""
        pass
        
    def preprocess(self, text: str) -> str:
        """Preprocess text using provided function or identity.
        
        Args:
            text: Input text to process
            
        Returns:
            str: Processed text
        """
        if self.preprocess_fn:
            return self.preprocess_fn(text)
        return text
        
    def save_cache(self) -> None:
        """Save processed examples to cache."""
        if not self.cache_dir:
            return
            
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / "processed.json"
        
        data = {
            "examples": [
                {
                    "id": ex.id,
                    "text": ex.text,
                    "metadata": ex.metadata
                }
                for ex in self._examples
            ],
            "metadata": self._metadata
        }
        
        with open(cache_path, "w") as f:
            json.dump(data, f)
            
    def create_example(self, id: str, text: str, metadata: Dict[str, Any]) -> Example:
        """Create an example instance of the appropriate type.
        
        This method should be overridden by subclasses that use custom Example types.
        
        Args:
            id: Example ID
            text: Primary text content
            metadata: Additional metadata
            
        Returns:
            Example: New example instance
        """
        return Example(id=id, text=text, metadata=metadata)

    def load_cache(self) -> bool:
        """Load processed examples from cache if available.
        
        Returns:
            bool: Whether cache was successfully loaded
        """
        if not self.cache_dir:
            return False
            
        cache_path = self.cache_dir / "processed.json"
        if not cache_path.exists():
            return False
            
        try:
            with open(cache_path) as f:
                data = json.load(f)
                
            self._examples = [
                self.create_example(
                    id=item["id"],
                    text=item["text"],
                    metadata=item["metadata"]
                )
                for item in data["examples"]
            ]
            self._metadata = data["metadata"]
            return True
        except:
            return False
            
    def __len__(self) -> int:
        """Get number of examples."""
        return len(self._examples)
        
    def __getitem__(self, idx: int) -> Example:
        """Get example by index."""
        return self._examples[idx]
        
    def __iter__(self) -> Iterator[Example]:
        """Iterate over examples."""
        return iter(self._examples)
        
    def to_pandas(self) -> pd.DataFrame:
        """Convert dataset to pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing examples
        """
        data = []
        for ex in self._examples:
            row = {"id": ex.id, "text": ex.text}
            row.update(ex.metadata)
            data.append(row)
        return pd.DataFrame(data)
        
    def filter(self, condition: callable) -> "BaseDataset":
        """Filter examples using provided condition.
        
        Args:
            condition: Function that takes Example and returns bool
            
        Returns:
            BaseDataset: New dataset with filtered examples
        """
        self._examples = [ex for ex in self._examples if condition(ex)]
        return self
        
    def map(self, transform: callable) -> "BaseDataset":
        """Apply transformation to all examples.
        
        Args:
            transform: Function that takes Example and returns modified Example
            
        Returns:
            BaseDataset: Dataset with transformed examples
        """
        self._examples = [transform(ex) for ex in self._examples]
        return self
