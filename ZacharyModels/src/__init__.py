"""
CrowS-Pairs evaluation infrastructure for MLMs and LLMs.
"""

from .data import CrowsPairsDataset
from .models import ModelFactory, BaseModel
from .prompts import PromptWrapper, PromptTemplate
from .evaluation import BiasScorer, BiasMetrics

__version__ = "0.1.0"

__all__ = [
    "CrowsPairsDataset",
    "ModelFactory",
    "BaseModel",
    "PromptWrapper",
    "PromptTemplate",
    "BiasScorer",
    "BiasMetrics"
]
