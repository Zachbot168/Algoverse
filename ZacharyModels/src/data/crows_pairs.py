import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from .base import BaseDataset, Example

class CrowsPairsExample(Example):
    """CrowS-Pairs example with stereotype and anti-stereotype sentences."""
    
    def __init__(
        self,
        id: str,
        sent_more: str,
        sent_less: str,
        stereo_antistereo: str,
        bias_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize CrowS-Pairs example.
        
        Args:
            id: Unique identifier
            sent_more: More stereotypical sentence
            sent_less: Less stereotypical sentence
            stereo_antistereo: Direction of stereotype ('stereo' or 'antistereo')
            bias_type: Type of bias (e.g., 'gender', 'race')
            metadata: Additional metadata
        """
        # Determine which sentence is stereotypical vs anti-stereotypical
        stereo = sent_more if stereo_antistereo == "stereo" else sent_less
        anti_stereo = sent_less if stereo_antistereo == "stereo" else sent_more
        
        super().__init__(
            id=id,
            text=stereo,  # Store stereotype as primary text
            metadata={
                "anti_stereo": anti_stereo,
                "sent_more": sent_more,
                "sent_less": sent_less,
                "stereo_antistereo": stereo_antistereo,
                "bias_type": bias_type,
                **(metadata or {})
            }
        )
        
    @property
    def stereo(self) -> str:
        """Get stereotypical sentence."""
        return self.text
        
    @property
    def anti_stereo(self) -> str:
        """Get anti-stereotypical sentence."""
        return self.metadata["anti_stereo"]
        
    @property
    def bias_type(self) -> str:
        """Get bias type."""
        return self.metadata["bias_type"]
        
    def as_pair(self) -> Tuple[str, str]:
        """Get sentence pair.
        
        Returns:
            Tuple[str, str]: (stereotypical, anti-stereotypical)
        """
        return self.stereo, self.anti_stereo

class CrowsPairsDataset(BaseDataset):
    """Dataset handler for CrowS-Pairs."""
    
    def create_example(self, id: str, text: str, metadata: Dict[str, Any]) -> Example:
        """Create a CrowsPairsExample instance.
        
        Override base class method to ensure proper example type creation,
        particularly when loading from cache.
        
        Args:
            id: Example ID
            text: Primary text (stereotype sentence)
            metadata: Additional metadata
            
        Returns:
            CrowsPairsExample: New example instance
        """
        # Extract fields needed for CrowsPairsExample construction
        anti_stereo = metadata.pop("anti_stereo")
        bias_type = metadata.pop("bias_type", "")
        
        return CrowsPairsExample(
            id=id,
            sent_more=metadata.get("sent_more", text),
            sent_less=metadata.get("sent_less", anti_stereo),
            stereo_antistereo=metadata.get("stereo_antistereo", "stereo"),
            bias_type=bias_type,
            metadata=metadata
        )
    
    def __init__(
        self,
        data_dir: str,
        bias_types: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        preprocess_fn: Optional[callable] = None
    ):
        """Initialize CrowS-Pairs dataset.
        
        Args:
            data_dir: Base directory containing dataset
            bias_types: Optional list of bias types to include
            cache_dir: Directory for caching processed data
            preprocess_fn: Optional preprocessing function
        """
        super().__init__(data_dir, cache_dir, preprocess_fn)
        self.bias_types = set(bias_types) if bias_types else None
        
    def load(self) -> None:
        """Load CrowS-Pairs dataset from disk.
        
        Expects CSV file with columns:
        - sent_more: More stereotypical sentence
        - sent_less: Less stereotypical sentence
        - stereo_antistereo: Direction of stereotype ('stereo' or 'antistereo')
        - bias_type: Type of bias
        - annotations: Bias type annotations
        - anon_writer: Anonymized writer ID
        - anon_annotators: Anonymized annotator IDs
        """
        # Try loading from cache first
        if self.load_cache():
            return
            
        # Find main dataset file
        data_path = self.data_dir / "crows_pairs_anonymized.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {data_path}")
            
        examples = []
        df = pd.read_csv(data_path, index_col=0)
        for _, row in df.iterrows():
            # Skip if bias type not in specified types
            if self.bias_types and row["bias_type"] not in self.bias_types:
                continue
                
            example = CrowsPairsExample(
                id=str(row.name),  # Use index as ID since we're using index_col=0
                sent_more=self.preprocess(row["sent_more"].strip()),
                sent_less=self.preprocess(row["sent_less"].strip()),
                stereo_antistereo=row["stereo_antistereo"],
                bias_type=row["bias_type"],
                metadata={
                    k: v for k, v in row.items()
                    if k not in ["sent_more", "sent_less", "stereo_antistereo", "bias_type"]
                }
            )
            examples.append(example)
        
        self._examples = examples
        
        # Save processed examples
        self.save_cache()
        
    def get_bias_types(self) -> List[str]:
        """Get list of unique bias types in dataset.
        
        Returns:
            List[str]: Unique bias types
        """
        return list({ex.bias_type for ex in self._examples})
        
    def filter_by_bias(self, bias_types: List[str]) -> "CrowsPairsDataset":
        """Filter examples by bias type.
        
        Args:
            bias_types: List of bias types to include
            
        Returns:
            CrowsPairsDataset: Filtered dataset
        """
        bias_types = set(bias_types)
        return self.filter(lambda ex: ex.bias_type in bias_types)
        
    def get_pairs(self) -> List[Tuple[str, str]]:
        """Get all sentence pairs.
        
        Returns:
            List[Tuple[str, str]]: List of (stereo, anti-stereo) pairs
        """
        return [ex.as_pair() for ex in self._examples]
