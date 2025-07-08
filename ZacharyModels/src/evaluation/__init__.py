"""Evaluation framework for bias assessment."""

from .scorer import BiasScorer
from .metrics import BiasMetrics

def create_scorer(
    model,
    prompt_wrapper=None,
    cache_dir=None
) -> BiasScorer:
    """Convenience function to create a bias scorer.
    
    Args:
        model: Model instance to evaluate
        prompt_wrapper: Optional prompt wrapper
        cache_dir: Optional cache directory
        
    Returns:
        BiasScorer: Configured scorer instance
    """
    return BiasScorer(
        model=model,
        prompt_wrapper=prompt_wrapper,
        cache_dir=cache_dir
    )

def analyze_results(results, threshold=0.5):
    """Convenience function to analyze results.
    
    Args:
        results: Evaluation results
        threshold: Bias threshold
        
    Returns:
        dict: Analysis results
    """
    return BiasMetrics.analyze_results(results, threshold)

__all__ = [
    "BiasScorer",
    "BiasMetrics",
    "create_scorer",
    "analyze_results"
]
