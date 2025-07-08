from typing import List, Dict, Any, Tuple, Optional
import torch
from pathlib import Path
import json

from ..data.crows_pairs import CrowsPairsDataset, CrowsPairsExample
from ..models import BaseModel
from ..prompts import PromptWrapper

class BiasScorer:
    """Scorer for evaluating bias in model predictions."""
    
    def __init__(
        self,
        model: BaseModel,
        prompt_wrapper: Optional[PromptWrapper] = None,
        cache_dir: Optional[str] = None
    ):
        """Initialize scorer.
        
        Args:
            model: Model to evaluate
            prompt_wrapper: Optional prompt wrapper for formatting inputs
            cache_dir: Optional directory for caching results
        """
        self.model = model
        self.prompt_wrapper = prompt_wrapper
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
    def prepare_input(self, text: str) -> str:
        """Prepare input text with prompt wrapper if available.
        
        Args:
            text: Raw input text
            
        Returns:
            str: Processed input text
        """
        if self.prompt_wrapper:
            return self.prompt_wrapper.wrap(text)
        return text
        
    def score_pair(
        self,
        stereo: str,
        anti_stereo: str
    ) -> Tuple[float, float]:
        """Score a stereotype/anti-stereotype pair.
        
        Args:
            stereo: Stereotypical sentence
            anti_stereo: Anti-stereotypical sentence
            
        Returns:
            Tuple[float, float]: (stereo_score, anti_stereo_score)
        """
        # Prepare inputs
        stereo_input = self.prepare_input(stereo)
        anti_stereo_input = self.prepare_input(anti_stereo)
        
        # Get model scores
        stereo_score = self.model.compute_score(stereo_input)
        anti_stereo_score = self.model.compute_score(anti_stereo_input)
        
        return stereo_score, anti_stereo_score
        
    def score_dataset(
        self,
        dataset: CrowsPairsDataset,
        batch_size: int = 8
    ) -> List[Dict[str, Any]]:
        """Score all pairs in dataset.
        
        Args:
            dataset: CrowS-Pairs dataset
            batch_size: Batch size for processing
            
        Returns:
            List[Dict[str, Any]]: Scoring results for each pair
        """
        results = []
        pairs = dataset.get_pairs()
        
        # Process in batches
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            
            # Prepare inputs for batch
            stereo_inputs = [
                self.prepare_input(stereo)
                for stereo, _ in batch_pairs
            ]
            anti_stereo_inputs = [
                self.prepare_input(anti_stereo)
                for _, anti_stereo in batch_pairs
            ]
            
            # Get model scores
            stereo_scores = self.model.batch_compute_scores(stereo_inputs)
            anti_stereo_scores = self.model.batch_compute_scores(anti_stereo_inputs)
            
            # Combine results
            for j, ((stereo, anti_stereo), s_score, a_score) in enumerate(
                zip(batch_pairs, stereo_scores, anti_stereo_scores)
            ):
                example = dataset[i + j]
                results.append({
                    "id": example.id,
                    "stereo_text": stereo,
                    "anti_stereo_text": anti_stereo,
                    "stereo_score": s_score,
                    "anti_stereo_score": a_score,
                    "bias_type": example.bias_type,
                    "bias_score": s_score - a_score  # Higher = more biased
                })
                
        return results
        
    def save_results(
        self,
        results: List[Dict[str, Any]],
        model_name: str,
        output_file: Optional[str] = None
    ) -> None:
        """Save scoring results to file.
        
        Args:
            results: Scoring results to save
            model_name: Name of evaluated model
            output_file: Optional output file path
        """
        if not output_file and not self.cache_dir:
            return
            
        output_path = Path(output_file) if output_file else (
            self.cache_dir / f"{model_name}_results.json"
        )
        
        # Create parent directories if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save results with metadata
        data = {
            "model_name": model_name,
            "num_examples": len(results),
            "results": results,
            "metadata": {
                "prompt_template": (
                    self.prompt_wrapper.template.template
                    if self.prompt_wrapper else None
                )
            }
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
            
    @classmethod
    def load_results(cls, results_file: str) -> Dict[str, Any]:
        """Load saved results from file.
        
        Args:
            results_file: Path to results file
            
        Returns:
            Dict[str, Any]: Loaded results
        """
        with open(results_file) as f:
            return json.load(f)
