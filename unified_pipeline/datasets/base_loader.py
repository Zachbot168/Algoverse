"""
Base Dataset Loader for Unified Bias Mitigation Pipeline

Provides abstract base classes and common functionality for all dataset loaders.
Each dataset loader preserves unique characteristics while providing standardized interface.
"""

import json
import csv
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd


class BiasType(Enum):
    """Types of bias that can be evaluated."""
    GENDER = "gender"
    RACIAL = "racial"
    RELIGIOUS = "religious" 
    DEMOGRAPHIC = "demographic"
    SYCOPHANCY = "sycophancy"
    PROFESSION = "profession"
    STEREOTYPE = "stereotype"
    GENERAL = "general"


class EvaluationMode(Enum):
    """Different evaluation methodologies."""
    MULTIPLE_CHOICE = "multiple_choice"
    SENTENCE_COMPLETION = "sentence_completion" 
    GENERATION = "generation"
    CLASSIFICATION = "classification"
    ASSOCIATION_TEST = "association_test"
    QA = "qa"
    TRUTHFULNESS = "truthfulness"


@dataclass
class DatasetMetadata:
    """Metadata describing dataset characteristics."""
    name: str
    bias_types: List[BiasType]
    evaluation_mode: EvaluationMode
    size: int
    description: str
    citation: str
    data_format: str  # json, csv, jsonl
    unique_features: List[str]
    requires_generation: bool = False
    supports_few_shot: bool = False
    has_demographics: bool = False


class BaseDatasetLoader(ABC):
    """
    Abstract base class for all dataset loaders.
    
    Each dataset loader must implement standardized methods while preserving
    the unique evaluation characteristics of their specific dataset.
    """
    
    def __init__(self, data_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize dataset loader.
        
        Args:
            data_path: Path to dataset directory
            config: Optional configuration parameters
        """
        self.data_path = Path(data_path)
        self.config = config or {}
        self.metadata = self.get_metadata()
        self._validate_data_path()
    
    @abstractmethod
    def get_metadata(self) -> DatasetMetadata:
        """Return dataset metadata."""
        pass
    
    @abstractmethod
    def load_data(self, split: str = "test", sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load dataset samples.
        
        Args:
            split: Dataset split (train/dev/test)
            sample_size: Optional limit on samples returned
            
        Returns:
            List of standardized sample dictionaries
        """
        pass
    
    @abstractmethod
    def prepare_for_evaluation(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare samples for model evaluation, preserving dataset-specific format.
        
        Args:
            samples: Raw dataset samples
            
        Returns:
            Evaluation-ready samples with standardized format
        """
        pass
    
    @abstractmethod
    def compute_metrics(self, predictions: List[Any], targets: List[Any]) -> Dict[str, float]:
        """
        Compute dataset-specific evaluation metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of metric name -> value
        """
        pass
    
    def _validate_data_path(self):
        """Validate that data path exists and contains required files."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.data_path}")
    
    def _load_json(self, file_path: Path) -> Union[Dict, List]:
        """Load JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_jsonl(self, file_path: Path) -> List[Dict]:
        """Load JSONL file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        return data
    
    def _load_csv(self, file_path: Path) -> List[Dict]:
        """Load CSV file."""
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    
    def _sample_data(self, data: List[Dict], sample_size: Optional[int]) -> List[Dict]:
        """Sample data if sample_size is specified."""
        if sample_size is not None and len(data) > sample_size:
            import random
            random.seed(42)  # For reproducibility
            return random.sample(data, sample_size)
        return data
    
    def get_bias_types(self) -> List[BiasType]:
        """Return list of bias types this dataset evaluates."""
        return self.metadata.bias_types
    
    def get_evaluation_mode(self) -> EvaluationMode:
        """Return evaluation methodology for this dataset.""" 
        return self.metadata.evaluation_mode
    
    def supports_bias_type(self, bias_type: BiasType) -> bool:
        """Check if dataset supports a specific bias type."""
        return bias_type in self.metadata.bias_types


class StandardizedSample:
    """
    Standardized sample format that preserves dataset-specific information
    while providing common interface for pipeline processing.
    """
    
    def __init__(
        self,
        text: str,
        target: Any,
        bias_type: BiasType,
        evaluation_mode: EvaluationMode,
        metadata: Dict[str, Any],
        original_format: Dict[str, Any]
    ):
        self.text = text
        self.target = target
        self.bias_type = bias_type
        self.evaluation_mode = evaluation_mode
        self.metadata = metadata
        self.original_format = original_format  # Preserve original for dataset-specific evaluation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'text': self.text,
            'target': self.target,
            'bias_type': self.bias_type.value,
            'evaluation_mode': self.evaluation_mode.value,
            'metadata': self.metadata,
            'original_format': self.original_format
        }