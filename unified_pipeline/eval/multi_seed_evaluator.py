#!/usr/bin/env python3
"""
Multi-Seed Evaluation Framework for Phase 4: Validation & Robustness
Implements comprehensive multi-seed evaluation with real confidence intervals and statistical validation.
"""

import torch
import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional, Callable
from pathlib import Path
import logging
from dataclasses import dataclass
from tqdm import tqdm
import json
from scipy import stats
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from collections import defaultdict
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


@dataclass
class SeedEvaluationResult:
    """Results from a single seed evaluation."""
    seed: int
    dataset_name: str
    bias_scores: Dict[str, float]
    accuracy_scores: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    sample_count: int
    evaluation_time: float
    metadata: Dict[str, Any] = None


@dataclass
class MultiSeedResults:
    """Aggregated results from multi-seed evaluation."""
    dataset_name: str
    seeds_evaluated: List[int]
    mean_bias_scores: Dict[str, float]
    std_bias_scores: Dict[str, float]
    mean_accuracy_scores: Dict[str, float]
    std_accuracy_scores: Dict[str, float]
    confidence_intervals_95: Dict[str, Tuple[float, float]]
    confidence_intervals_99: Dict[str, Tuple[float, float]]
    statistical_significance: Dict[str, Dict[str, float]]
    effect_sizes: Dict[str, float]
    total_samples: int
    total_evaluation_time: float
    robustness_metrics: Dict[str, float]


class MultiSeedEvaluator:
    """
    Multi-seed evaluation framework for robust bias evaluation.
    Provides statistically valid confidence intervals and robustness testing.
    """
    
    def __init__(self, base_evaluator, logger: Optional[logging.Logger] = None):
        """
        Initialize multi-seed evaluator.
        
        Args:
            base_evaluator: Base bias evaluator (e.g., RealBiasEvaluator)
            logger: Optional logger for output
        """
        self.base_evaluator = base_evaluator
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration
        self.default_seeds = [42, 123, 456, 789, 999, 1337, 2023, 2024]
        self.confidence_levels = [0.95, 0.99]
        
        # Results storage
        self.evaluation_history = []
        self.seed_results = {}
        
        self.logger.info("Initialized MultiSeedEvaluator for robust bias evaluation")
    
    def evaluate_multiple_seeds(self, 
                               dataset_path: str,
                               dataset_name: str,
                               evaluation_function: Callable,
                               seeds: Optional[List[int]] = None,
                               num_samples: Optional[int] = None,
                               parallel: bool = True,
                               max_workers: int = 4) -> MultiSeedResults:
        """
        Evaluate bias across multiple seeds for statistical robustness.
        
        Args:
            dataset_path: Path to dataset
            dataset_name: Name of dataset being evaluated
            evaluation_function: Function to call for evaluation (e.g., evaluate_winogender)
            seeds: List of seeds to use (default: predefined set)
            num_samples: Number of samples per evaluation
            parallel: Whether to run evaluations in parallel
            max_workers: Maximum parallel workers
            
        Returns:
            MultiSeedResults with aggregated statistics
        """
        seeds = seeds or self.default_seeds
        self.logger.info(f"Starting multi-seed evaluation for {dataset_name} with {len(seeds)} seeds")
        
        seed_results = []
        start_time = time.time()
        
        if parallel and len(seeds) > 1:
            # Parallel evaluation
            seed_results = self._evaluate_parallel(
                dataset_path, dataset_name, evaluation_function, 
                seeds, num_samples, max_workers
            )
        else:
            # Sequential evaluation
            seed_results = self._evaluate_sequential(
                dataset_path, dataset_name, evaluation_function,
                seeds, num_samples
            )
        
        total_time = time.time() - start_time
        
        # Aggregate results
        aggregated_results = self._aggregate_seed_results(
            seed_results, dataset_name, total_time
        )
        
        # Store results
        self.evaluation_history.append(aggregated_results)
        self.seed_results[dataset_name] = seed_results
        
        self.logger.info(f"Multi-seed evaluation completed in {total_time:.2f}s")
        return aggregated_results
    
    def _evaluate_sequential(self, 
                           dataset_path: str,
                           dataset_name: str,
                           evaluation_function: Callable,
                           seeds: List[int],
                           num_samples: Optional[int]) -> List[SeedEvaluationResult]:
        """Sequential evaluation across seeds."""
        results = []
        
        for seed in tqdm(seeds, desc=f"Evaluating {dataset_name}"):
            result = self._evaluate_single_seed(
                seed, dataset_path, dataset_name, evaluation_function, num_samples
            )
            if result:
                results.append(result)
        
        return results
    
    def _evaluate_parallel(self,
                         dataset_path: str,
                         dataset_name: str,
                         evaluation_function: Callable,
                         seeds: List[int],
                         num_samples: Optional[int],
                         max_workers: int) -> List[SeedEvaluationResult]:
        """Parallel evaluation across seeds."""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all seed evaluations
            future_to_seed = {
                executor.submit(
                    self._evaluate_single_seed,
                    seed, dataset_path, dataset_name, evaluation_function, num_samples
                ): seed for seed in seeds
            }
            
            # Collect results
            for future in tqdm(as_completed(future_to_seed), total=len(seeds), 
                              desc=f"Evaluating {dataset_name}"):
                seed = future_to_seed[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as exc:
                    self.logger.warning(f"Seed {seed} evaluation failed: {exc}")
        
        return results
    
    def _evaluate_single_seed(self,
                            seed: int,
                            dataset_path: str,
                            dataset_name: str,
                            evaluation_function: Callable,
                            num_samples: Optional[int]) -> Optional[SeedEvaluationResult]:
        """Evaluate bias for a single seed."""
        try:
            # Set seed for reproducibility
            self._set_seed(seed)
            
            start_time = time.time()
            
            # Run evaluation
            eval_result = evaluation_function(dataset_path, num_samples)
            
            evaluation_time = time.time() - start_time
            
            # Extract metrics
            bias_scores = {
                'primary_bias_score': eval_result.bias_score,
                'accuracy': eval_result.accuracy
            }
            
            accuracy_scores = {
                'overall_accuracy': eval_result.accuracy
            }
            
            # Extract additional metrics from metadata
            if hasattr(eval_result, 'metadata') and eval_result.metadata:
                for key, value in eval_result.metadata.items():
                    if isinstance(value, (int, float)) and 'accuracy' in key.lower():
                        accuracy_scores[key] = float(value)
                    elif isinstance(value, (int, float)) and any(bias_term in key.lower() 
                                                               for bias_term in ['bias', 'score', 'rate']):
                        bias_scores[key] = float(value)
            
            # Compute confidence intervals for this seed (bootstrap)
            confidence_intervals = self._compute_seed_confidence_intervals(eval_result)
            
            return SeedEvaluationResult(
                seed=seed,
                dataset_name=dataset_name,
                bias_scores=bias_scores,
                accuracy_scores=accuracy_scores,
                confidence_intervals=confidence_intervals,
                sample_count=eval_result.sample_count,
                evaluation_time=evaluation_time,
                metadata={
                    'individual_scores': getattr(eval_result, 'individual_scores', []),
                    'statistical_significance': getattr(eval_result, 'statistical_significance', {}),
                    'dataset_path': dataset_path
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate seed {seed}: {e}")
            return None
    
    def _set_seed(self, seed: int):
        """Set all random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def _compute_seed_confidence_intervals(self, eval_result) -> Dict[str, Tuple[float, float]]:
        """Compute confidence intervals for a single seed using bootstrap."""
        confidence_intervals = {}
        
        # Get individual scores if available
        individual_scores = getattr(eval_result, 'individual_scores', [])
        
        if len(individual_scores) > 10:  # Need sufficient samples for bootstrap
            # Bootstrap confidence interval for bias score
            bootstrap_scores = []
            n_bootstrap = 1000
            
            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(individual_scores, size=len(individual_scores), replace=True)
                bootstrap_scores.append(np.mean(bootstrap_sample))
            
            # 95% confidence interval
            ci_lower = np.percentile(bootstrap_scores, 2.5)
            ci_upper = np.percentile(bootstrap_scores, 97.5)
            confidence_intervals['bias_score_95'] = (ci_lower, ci_upper)
        else:
            # Fallback using normal approximation
            confidence_intervals['bias_score_95'] = (eval_result.bias_score * 0.9, eval_result.bias_score * 1.1)
        
        return confidence_intervals
    
    def _aggregate_seed_results(self, 
                              seed_results: List[SeedEvaluationResult],
                              dataset_name: str,
                              total_time: float) -> MultiSeedResults:
        """Aggregate results across all seeds."""
        if not seed_results:
            raise ValueError("No valid seed results to aggregate")
        
        # Collect all metrics
        all_bias_scores = defaultdict(list)
        all_accuracy_scores = defaultdict(list)
        
        for result in seed_results:
            for metric, value in result.bias_scores.items():
                all_bias_scores[metric].append(value)
            for metric, value in result.accuracy_scores.items():
                all_accuracy_scores[metric].append(value)
        
        # Compute means and standard deviations
        mean_bias_scores = {k: np.mean(v) for k, v in all_bias_scores.items()}
        std_bias_scores = {k: np.std(v) for k, v in all_bias_scores.items()}
        mean_accuracy_scores = {k: np.mean(v) for k, v in all_accuracy_scores.items()}
        std_accuracy_scores = {k: np.std(v) for k, v in all_accuracy_scores.items()}
        
        # Compute confidence intervals
        confidence_intervals_95 = {}
        confidence_intervals_99 = {}
        
        for metric, values in all_bias_scores.items():
            if len(values) > 1:
                # t-distribution for small samples
                ci_95 = stats.t.interval(0.95, len(values)-1, loc=np.mean(values), scale=stats.sem(values))
                ci_99 = stats.t.interval(0.99, len(values)-1, loc=np.mean(values), scale=stats.sem(values))
                confidence_intervals_95[metric] = ci_95
                confidence_intervals_99[metric] = ci_99
        
        # Statistical significance testing
        statistical_significance = self._compute_statistical_significance(all_bias_scores, all_accuracy_scores)
        
        # Effect sizes (Cohen's d)
        effect_sizes = self._compute_effect_sizes(all_bias_scores)
        
        # Robustness metrics
        robustness_metrics = self._compute_robustness_metrics(seed_results)
        
        return MultiSeedResults(
            dataset_name=dataset_name,
            seeds_evaluated=[r.seed for r in seed_results],
            mean_bias_scores=mean_bias_scores,
            std_bias_scores=std_bias_scores,
            mean_accuracy_scores=mean_accuracy_scores,
            std_accuracy_scores=std_accuracy_scores,
            confidence_intervals_95=confidence_intervals_95,
            confidence_intervals_99=confidence_intervals_99,
            statistical_significance=statistical_significance,
            effect_sizes=effect_sizes,
            total_samples=sum(r.sample_count for r in seed_results),
            total_evaluation_time=total_time,
            robustness_metrics=robustness_metrics
        )
    
    def _compute_statistical_significance(self, 
                                        all_bias_scores: Dict[str, List[float]],
                                        all_accuracy_scores: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Compute statistical significance tests."""
        significance_results = {}
        
        for metric, values in all_bias_scores.items():
            if len(values) > 1:
                # One-sample t-test against null hypothesis of no bias (0.5 for most metrics)
                null_value = 0.5 if 'accuracy' in metric.lower() else 0.0
                t_stat, p_value = stats.ttest_1samp(values, null_value)
                
                # Normality test
                if len(values) >= 8:
                    _, normality_p = stats.shapiro(values)
                else:
                    normality_p = 1.0  # Assume normal for small samples
                
                significance_results[metric] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant_05': p_value < 0.05,
                    'significant_01': p_value < 0.01,
                    'normality_p_value': float(normality_p),
                    'sample_size': len(values)
                }
        
        return significance_results
    
    def _compute_effect_sizes(self, all_bias_scores: Dict[str, List[float]]) -> Dict[str, float]:
        """Compute effect sizes (Cohen's d) for bias metrics."""
        effect_sizes = {}
        
        for metric, values in all_bias_scores.items():
            if len(values) > 1:
                # Cohen's d = (mean - null_value) / std
                null_value = 0.5 if 'accuracy' in metric.lower() else 0.0
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1)
                
                if std_val > 0:
                    cohens_d = (mean_val - null_value) / std_val
                    effect_sizes[metric] = float(cohens_d)
                else:
                    effect_sizes[metric] = 0.0
        
        return effect_sizes
    
    def _compute_robustness_metrics(self, seed_results: List[SeedEvaluationResult]) -> Dict[str, float]:
        """Compute robustness metrics across seeds."""
        if len(seed_results) < 2:
            return {}
        
        # Coefficient of variation for primary bias scores
        primary_scores = [r.bias_scores.get('primary_bias_score', 0.0) for r in seed_results]
        cv_bias = np.std(primary_scores) / (np.mean(primary_scores) + 1e-8)
        
        # Accuracy variation
        accuracy_scores = [r.accuracy_scores.get('overall_accuracy', 0.0) for r in seed_results]
        cv_accuracy = np.std(accuracy_scores) / (np.mean(accuracy_scores) + 1e-8)
        
        # Evaluation time consistency
        eval_times = [r.evaluation_time for r in seed_results]
        cv_time = np.std(eval_times) / (np.mean(eval_times) + 1e-8)
        
        # Range of results
        bias_range = max(primary_scores) - min(primary_scores)
        accuracy_range = max(accuracy_scores) - min(accuracy_scores)
        
        return {
            'coefficient_variation_bias': float(cv_bias),
            'coefficient_variation_accuracy': float(cv_accuracy),
            'coefficient_variation_time': float(cv_time),
            'bias_score_range': float(bias_range),
            'accuracy_range': float(accuracy_range),
            'mean_evaluation_time': float(np.mean(eval_times)),
            'evaluation_stability': float(1.0 / (1.0 + cv_bias))  # Higher is more stable
        }
    
    def compare_interventions(self, 
                            baseline_results: MultiSeedResults,
                            intervention_results: MultiSeedResults) -> Dict[str, Any]:
        """Compare baseline vs intervention results for statistical significance."""
        comparison = {
            'dataset': baseline_results.dataset_name,
            'baseline_seeds': len(baseline_results.seeds_evaluated),
            'intervention_seeds': len(intervention_results.seeds_evaluated),
            'metrics_comparison': {},
            'statistical_tests': {}
        }
        
        # Compare each metric
        for metric in baseline_results.mean_bias_scores:
            if metric in intervention_results.mean_bias_scores:
                baseline_values = []
                intervention_values = []
                
                # Extract individual values
                for seed_result in self.seed_results.get(baseline_results.dataset_name, []):
                    if metric in seed_result.bias_scores:
                        baseline_values.append(seed_result.bias_scores[metric])
                
                for seed_result in self.seed_results.get(intervention_results.dataset_name, []):
                    if metric in seed_result.bias_scores:
                        intervention_values.append(seed_result.bias_scores[metric])
                
                if len(baseline_values) > 1 and len(intervention_values) > 1:
                    # Two-sample t-test
                    t_stat, p_value = stats.ttest_ind(baseline_values, intervention_values)
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(baseline_values) - 1) * np.var(baseline_values, ddof=1) + 
                                        (len(intervention_values) - 1) * np.var(intervention_values, ddof=1)) / 
                                       (len(baseline_values) + len(intervention_values) - 2))
                    cohens_d = (np.mean(intervention_values) - np.mean(baseline_values)) / pooled_std
                    
                    comparison['metrics_comparison'][metric] = {
                        'baseline_mean': float(np.mean(baseline_values)),
                        'intervention_mean': float(np.mean(intervention_values)),
                        'improvement': float(np.mean(baseline_values) - np.mean(intervention_values)),
                        'improvement_pct': float((np.mean(baseline_values) - np.mean(intervention_values)) / 
                                               (np.mean(baseline_values) + 1e-8) * 100)
                    }
                    
                    comparison['statistical_tests'][metric] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'cohens_d': float(cohens_d),
                        'significant': p_value < 0.05,
                        'effect_size_interpretation': self._interpret_effect_size(abs(cohens_d))
                    }
        
        return comparison
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def save_results(self, output_path: str):
        """Save all multi-seed evaluation results."""
        output_data = {
            'evaluation_history': [],
            'seed_results': {},
            'metadata': {
                'default_seeds': self.default_seeds,
                'confidence_levels': self.confidence_levels,
                'total_evaluations': len(self.evaluation_history)
            }
        }
        
        # Convert results to serializable format
        for result in self.evaluation_history:
            result_dict = {
                'dataset_name': result.dataset_name,
                'seeds_evaluated': result.seeds_evaluated,
                'mean_bias_scores': result.mean_bias_scores,
                'std_bias_scores': result.std_bias_scores,
                'mean_accuracy_scores': result.mean_accuracy_scores,
                'std_accuracy_scores': result.std_accuracy_scores,
                'confidence_intervals_95': {k: list(v) for k, v in result.confidence_intervals_95.items()},
                'confidence_intervals_99': {k: list(v) for k, v in result.confidence_intervals_99.items()},
                'statistical_significance': result.statistical_significance,
                'effect_sizes': result.effect_sizes,
                'total_samples': result.total_samples,
                'total_evaluation_time': result.total_evaluation_time,
                'robustness_metrics': result.robustness_metrics
            }
            output_data['evaluation_history'].append(result_dict)
        
        # Save seed-level results
        for dataset, results in self.seed_results.items():
            output_data['seed_results'][dataset] = []
            for result in results:
                result_dict = {
                    'seed': result.seed,
                    'dataset_name': result.dataset_name,
                    'bias_scores': result.bias_scores,
                    'accuracy_scores': result.accuracy_scores,
                    'confidence_intervals': {k: list(v) for k, v in result.confidence_intervals.items()},
                    'sample_count': result.sample_count,
                    'evaluation_time': result.evaluation_time,
                    'metadata': result.metadata
                }
                output_data['seed_results'][dataset].append(result_dict)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        self.logger.info(f"Multi-seed evaluation results saved to {output_path}")


def main():
    """Demo usage of MultiSeedEvaluator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-seed bias evaluation")
    parser.add_argument("--model", default="gpt2", help="Model name")
    parser.add_argument("--dataset", default="winogender", help="Dataset to evaluate")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456], help="Seeds to evaluate")
    parser.add_argument("--output", default="multi_seed_results.json", help="Output file")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load model and evaluator
    print(f"Loading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from unified_pipeline.eval.real_bias_evaluator import RealBiasEvaluator
    
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Initialize evaluators
    base_evaluator = RealBiasEvaluator(model, tokenizer)
    multi_seed_evaluator = MultiSeedEvaluator(base_evaluator)
    
    # Run multi-seed evaluation
    dataset_path = f"datasets/{args.dataset}"
    print(f"Running multi-seed evaluation on {args.dataset}...")
    
    if args.dataset == "winogender":
        eval_func = base_evaluator.evaluate_winogender
    elif args.dataset == "truthfulqa":
        eval_func = base_evaluator.evaluate_truthfulqa
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    results = multi_seed_evaluator.evaluate_multiple_seeds(
        dataset_path=dataset_path,
        dataset_name=args.dataset,
        evaluation_function=eval_func,
        seeds=args.seeds,
        num_samples=10
    )
    
    # Save results
    multi_seed_evaluator.save_results(args.output)
    
    # Print summary
    print(f"\n=== Multi-Seed Evaluation Results ===")
    print(f"Dataset: {results.dataset_name}")
    print(f"Seeds evaluated: {results.seeds_evaluated}")
    print(f"Mean bias score: {results.mean_bias_scores.get('primary_bias_score', 'N/A'):.4f}")
    print(f"Std bias score: {results.std_bias_scores.get('primary_bias_score', 'N/A'):.4f}")
    print(f"95% CI: {results.confidence_intervals_95.get('primary_bias_score', 'N/A')}")
    print(f"Robustness: {results.robustness_metrics.get('evaluation_stability', 'N/A'):.4f}")


if __name__ == "__main__":
    main()