from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import json

class BiasMetrics:
    """Metrics for analyzing bias in model predictions."""
    
    @staticmethod
    def compute_basic_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute basic bias metrics from results.
        
        Args:
            results: List of scoring results
            
        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        bias_scores = np.array([r["bias_score"] for r in results])
        abs_bias_scores = np.abs(bias_scores)
        
        return {
            "mean_bias": float(np.mean(bias_scores)),
            "std_bias": float(np.std(bias_scores)),
            "mean_abs_bias": float(np.mean(abs_bias_scores)),
            "median_abs_bias": float(np.median(abs_bias_scores)),
            "max_abs_bias": float(np.max(abs_bias_scores)),
            "min_abs_bias": float(np.min(abs_bias_scores))
        }
        
    @staticmethod
    def compute_bias_by_type(
        results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """Compute bias metrics broken down by bias type.
        
        Args:
            results: List of scoring results
            
        Returns:
            Dict[str, Dict[str, float]]: Metrics per bias type
        """
        df = pd.DataFrame(results)
        metrics_by_type = {}
        
        for bias_type in df["bias_type"].unique():
            type_scores = df[df["bias_type"] == bias_type]["bias_score"]
            abs_scores = np.abs(type_scores)
            
            metrics_by_type[bias_type] = {
                "count": len(type_scores),
                "mean_bias": float(np.mean(type_scores)),
                "std_bias": float(np.std(type_scores)),
                "mean_abs_bias": float(np.mean(abs_scores)),
                "median_abs_bias": float(np.median(abs_scores))
            }
            
        return metrics_by_type
        
    @staticmethod
    def compute_threshold_metrics(
        results: List[Dict[str, Any]],
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Compute metrics based on bias threshold.
        
        Args:
            results: List of scoring results
            threshold: Bias score threshold
            
        Returns:
            Dict[str, float]: Threshold-based metrics
        """
        bias_scores = np.array([r["bias_score"] for r in results])
        abs_scores = np.abs(bias_scores)
        
        return {
            "prop_above_threshold": float(np.mean(abs_scores > threshold)),
            "count_above_threshold": int(np.sum(abs_scores > threshold)),
            "mean_score_above_threshold": float(
                np.mean(abs_scores[abs_scores > threshold])
            ) if any(abs_scores > threshold) else 0.0
        }
        
    @staticmethod
    def compute_correlation_metrics(
        results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Compute correlations between scores and metadata.
        
        Args:
            results: List of scoring results
            
        Returns:
            Dict[str, float]: Correlation metrics
        """
        df = pd.DataFrame(results)
        correlations = {}
        
        # Correlation between stereotype and anti-stereotype scores
        correlations["stereo_anti_correlation"] = float(
            df["stereo_score"].corr(df["anti_stereo_score"])
        )
        
        # Additional correlations can be added here based on available metadata
        
        return correlations
        
    @classmethod
    def analyze_results(
        cls,
        results: List[Dict[str, Any]],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Comprehensive analysis of bias results.
        
        Args:
            results: List of scoring results
            threshold: Bias score threshold
            
        Returns:
            Dict[str, Any]: Complete analysis results
        """
        analysis = {
            "basic_metrics": cls.compute_basic_metrics(results),
            "bias_by_type": cls.compute_bias_by_type(results),
            "threshold_metrics": cls.compute_threshold_metrics(
                results, threshold
            ),
            "correlations": cls.compute_correlation_metrics(results)
        }
        
        # Add summary statistics
        df = pd.DataFrame(results)
        analysis["summary"] = {
            "total_examples": len(results),
            "bias_types": list(df["bias_type"].unique()),
            "percent_biased": float(
                np.mean(np.abs(df["bias_score"]) > threshold) * 100
            )
        }
        
        return analysis
        
    @classmethod
    def save_analysis(
        cls,
        analysis: Dict[str, Any],
        output_file: str,
        model_name: Optional[str] = None
    ) -> None:
        """Save analysis results to file.
        
        Args:
            analysis: Analysis results
            output_file: Output file path
            model_name: Optional model name to include
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if model_name:
            analysis["model_name"] = model_name
            
        with open(output_path, "w") as f:
            json.dump(analysis, f, indent=2)
            
    @classmethod
    def load_analysis(cls, analysis_file: str) -> Dict[str, Any]:
        """Load saved analysis from file.
        
        Args:
            analysis_file: Path to analysis file
            
        Returns:
            Dict[str, Any]: Loaded analysis
        """
        with open(analysis_file) as f:
            return json.load(f)
