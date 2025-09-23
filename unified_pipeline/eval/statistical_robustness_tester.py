#!/usr/bin/env python3
"""
Statistical Robustness Testing Suite for Phase 4: Validation & Robustness
Implements comprehensive statistical validation, bootstrapping, and robustness testing.
"""

import torch
import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from pathlib import Path
import logging
from dataclasses import dataclass, field
from tqdm import tqdm
import json
from scipy import stats
from scipy.stats import bootstrap, permutation_test
import pandas as pd
from collections import defaultdict
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.power import ttest_power
import time

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


@dataclass
class RobustnessTestResult:
    """Results from a single robustness test."""
    test_name: str
    test_type: str  # "bootstrap", "permutation", "cross_validation", "power_analysis"
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    power: float
    test_statistic: float
    sample_size: int
    robust: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComprehensiveRobustnessResults:
    """Complete statistical robustness assessment."""
    dataset_name: str
    intervention_type: str
    test_results: List[RobustnessTestResult]
    overall_robustness_score: float
    statistical_power: float
    effect_size_reliability: float
    distributional_assumptions: Dict[str, bool]
    recommendations: List[str]
    cross_validation_results: Dict[str, float]
    bootstrap_distributions: Dict[str, np.ndarray]
    metadata: Dict[str, Any]


class StatisticalRobustnessTester:
    """
    Comprehensive statistical robustness testing for bias intervention validation.
    Provides multiple statistical tests to validate intervention effectiveness.
    """
    
    def __init__(self, base_evaluator, logger: Optional[logging.Logger] = None):
        """
        Initialize statistical robustness tester.
        
        Args:
            base_evaluator: Base bias evaluator for taking measurements
            logger: Optional logger
        """
        self.base_evaluator = base_evaluator
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration
        self.bootstrap_samples = 10000
        self.permutation_samples = 5000
        self.cv_folds = 5
        self.alpha_level = 0.05
        self.power_threshold = 0.8
        self.effect_size_threshold = 0.5  # Medium effect size
        
        # Results storage
        self.test_history = []
        self.robustness_assessments = {}
        
        self.logger.info("Initialized StatisticalRobustnessTester")
    
    def comprehensive_robustness_test(self,
                                    baseline_data: List[Dict[str, Any]],
                                    intervention_data: List[Dict[str, Any]],
                                    dataset_name: str,
                                    intervention_type: str,
                                    evaluation_function: Callable) -> ComprehensiveRobustnessResults:
        """
        Perform comprehensive statistical robustness testing.
        
        Args:
            baseline_data: Baseline evaluation results
            intervention_data: Post-intervention evaluation results
            dataset_name: Name of dataset
            intervention_type: Type of intervention
            evaluation_function: Function to evaluate bias
            
        Returns:
            ComprehensiveRobustnessResults with complete analysis
        """
        self.logger.info(f"Starting comprehensive robustness testing for {intervention_type} on {dataset_name}")
        
        test_results = []
        
        # Extract numerical values for statistical testing
        baseline_scores = [result.get('bias_score', 0.0) for result in baseline_data]
        intervention_scores = [result.get('bias_score', 0.0) for result in intervention_data]
        
        # Test 1: Bootstrap Confidence Intervals
        bootstrap_result = self._bootstrap_test(baseline_scores, intervention_scores, dataset_name)
        test_results.append(bootstrap_result)
        
        # Test 2: Permutation Test
        permutation_result = self._permutation_test(baseline_scores, intervention_scores, dataset_name)
        test_results.append(permutation_result)
        
        # Test 3: Cross-Validation Robustness
        cv_result = self._cross_validation_robustness(
            baseline_data, intervention_data, evaluation_function, dataset_name
        )
        test_results.append(cv_result['test_result'])
        
        # Test 4: Distributional Assumption Testing
        distribution_result = self._test_distributional_assumptions(baseline_scores, intervention_scores)
        test_results.append(distribution_result)
        
        # Test 5: Power Analysis
        power_result = self._power_analysis(baseline_scores, intervention_scores, dataset_name)
        test_results.append(power_result)
        
        # Test 6: Effect Size Stability
        stability_result = self._effect_size_stability_test(baseline_scores, intervention_scores, dataset_name)
        test_results.append(stability_result)
        
        # Test 7: Outlier Robustness
        outlier_result = self._outlier_robustness_test(baseline_scores, intervention_scores, dataset_name)
        test_results.append(outlier_result)
        
        # Aggregate results
        overall_robustness = self._compute_overall_robustness(test_results)
        
        # Generate recommendations
        recommendations = self._generate_robustness_recommendations(test_results, overall_robustness)
        
        # Create comprehensive results
        results = ComprehensiveRobustnessResults(
            dataset_name=dataset_name,
            intervention_type=intervention_type,
            test_results=test_results,
            overall_robustness_score=overall_robustness,
            statistical_power=power_result.power,
            effect_size_reliability=stability_result.effect_size,
            distributional_assumptions=distribution_result.metadata.get('assumptions', {}),
            recommendations=recommendations,
            cross_validation_results=cv_result['cv_scores'],
            bootstrap_distributions=self._extract_bootstrap_distributions(test_results),
            metadata={
                'baseline_sample_size': len(baseline_scores),
                'intervention_sample_size': len(intervention_scores),
                'alpha_level': self.alpha_level,
                'bootstrap_samples': self.bootstrap_samples,
                'permutation_samples': self.permutation_samples
            }
        )
        
        # Store results
        self.test_history.append(results)
        self.robustness_assessments[f"{dataset_name}_{intervention_type}"] = results
        
        self.logger.info(f"Comprehensive robustness testing completed: {overall_robustness:.3f} robustness score")
        return results
    
    def _bootstrap_test(self, baseline: List[float], intervention: List[float], 
                       dataset_name: str) -> RobustnessTestResult:
        """Perform bootstrap confidence interval test."""
        self.logger.info("Running bootstrap confidence interval test...")
        
        # Compute observed difference
        observed_diff = np.mean(baseline) - np.mean(intervention)
        
        # Bootstrap resampling
        bootstrap_diffs = []
        n_baseline, n_intervention = len(baseline), len(intervention)
        
        for _ in range(self.bootstrap_samples):
            bootstrap_baseline = np.random.choice(baseline, size=n_baseline, replace=True)
            bootstrap_intervention = np.random.choice(intervention, size=n_intervention, replace=True)
            bootstrap_diff = np.mean(bootstrap_baseline) - np.mean(bootstrap_intervention)
            bootstrap_diffs.append(bootstrap_diff)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Compute confidence interval
        ci_lower = np.percentile(bootstrap_diffs, (self.alpha_level/2) * 100)
        ci_upper = np.percentile(bootstrap_diffs, (1 - self.alpha_level/2) * 100)
        
        # Compute p-value (two-tailed test against null hypothesis of no difference)
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((n_baseline - 1) * np.var(baseline, ddof=1) + 
                             (n_intervention - 1) * np.var(intervention, ddof=1)) / 
                            (n_baseline + n_intervention - 2))
        effect_size = observed_diff / pooled_std if pooled_std > 0 else 0.0
        
        return RobustnessTestResult(
            test_name="Bootstrap Confidence Interval",
            test_type="bootstrap",
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            power=0.0,  # Not applicable for bootstrap
            test_statistic=observed_diff,
            sample_size=n_baseline + n_intervention,
            robust=p_value < self.alpha_level and abs(effect_size) > self.effect_size_threshold,
            metadata={
                'bootstrap_samples': self.bootstrap_samples,
                'bootstrap_distribution': bootstrap_diffs.tolist()[:1000],  # Save subset
                'observed_difference': observed_diff
            }
        )
    
    def _permutation_test(self, baseline: List[float], intervention: List[float],
                         dataset_name: str) -> RobustnessTestResult:
        """Perform permutation test for intervention effectiveness."""
        self.logger.info("Running permutation test...")
        
        # Observed test statistic (difference in means)
        observed_diff = np.mean(baseline) - np.mean(intervention)
        
        # Combine all data
        combined_data = baseline + intervention
        n_baseline = len(baseline)
        n_total = len(combined_data)
        
        # Permutation test
        permutation_diffs = []
        
        for _ in range(self.permutation_samples):
            # Randomly permute labels
            permuted_data = np.random.permutation(combined_data)
            perm_baseline = permuted_data[:n_baseline]
            perm_intervention = permuted_data[n_baseline:]
            
            perm_diff = np.mean(perm_baseline) - np.mean(perm_intervention)
            permutation_diffs.append(perm_diff)
        
        permutation_diffs = np.array(permutation_diffs)
        
        # Compute p-value
        p_value = np.mean(np.abs(permutation_diffs) >= np.abs(observed_diff))
        
        # Effect size
        pooled_std = np.sqrt(((len(baseline) - 1) * np.var(baseline, ddof=1) + 
                             (len(intervention) - 1) * np.var(intervention, ddof=1)) / 
                            (len(baseline) + len(intervention) - 2))
        effect_size = observed_diff / pooled_std if pooled_std > 0 else 0.0
        
        return RobustnessTestResult(
            test_name="Permutation Test",
            test_type="permutation",
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(0.0, 0.0),  # Not directly applicable
            power=0.0,  # Not directly applicable
            test_statistic=observed_diff,
            sample_size=n_total,
            robust=p_value < self.alpha_level,
            metadata={
                'permutation_samples': self.permutation_samples,
                'null_distribution_mean': float(np.mean(permutation_diffs)),
                'null_distribution_std': float(np.std(permutation_diffs))
            }
        )
    
    def _cross_validation_robustness(self, baseline_data: List[Dict[str, Any]],
                                   intervention_data: List[Dict[str, Any]],
                                   evaluation_function: Callable,
                                   dataset_name: str) -> Dict[str, Any]:
        """Test robustness using cross-validation."""
        self.logger.info("Running cross-validation robustness test...")
        
        # Combine and shuffle data
        all_data = baseline_data + intervention_data
        labels = [0] * len(baseline_data) + [1] * len(intervention_data)
        
        indices = list(range(len(all_data)))
        random.shuffle(indices)
        
        fold_size = len(indices) // self.cv_folds
        cv_scores = []
        effect_sizes = []
        
        for fold in range(self.cv_folds):
            # Split data
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < self.cv_folds - 1 else len(indices)
            
            test_indices = indices[start_idx:end_idx]
            train_indices = [i for i in indices if i not in test_indices]
            
            # Extract fold data
            fold_baseline = [all_data[i] for i in train_indices if labels[i] == 0]
            fold_intervention = [all_data[i] for i in train_indices if labels[i] == 1]
            
            if len(fold_baseline) == 0 or len(fold_intervention) == 0:
                continue
            
            # Compute fold statistics
            baseline_scores = [d.get('bias_score', 0.0) for d in fold_baseline]
            intervention_scores = [d.get('bias_score', 0.0) for d in fold_intervention]
            
            # Effect size for this fold
            pooled_std = np.sqrt(((len(baseline_scores) - 1) * np.var(baseline_scores, ddof=1) + 
                                 (len(intervention_scores) - 1) * np.var(intervention_scores, ddof=1)) / 
                                (len(baseline_scores) + len(intervention_scores) - 2))
            
            if pooled_std > 0:
                fold_effect_size = (np.mean(baseline_scores) - np.mean(intervention_scores)) / pooled_std
                effect_sizes.append(fold_effect_size)
                
                # Statistical test for this fold
                if len(baseline_scores) > 1 and len(intervention_scores) > 1:
                    t_stat, p_val = stats.ttest_ind(baseline_scores, intervention_scores)
                    cv_scores.append(1.0 if p_val < self.alpha_level else 0.0)
        
        # Aggregate CV results
        cv_accuracy = np.mean(cv_scores) if cv_scores else 0.0
        cv_effect_stability = 1.0 - (np.std(effect_sizes) / (np.mean(np.abs(effect_sizes)) + 1e-8)) if effect_sizes else 0.0
        
        # Create test result
        test_result = RobustnessTestResult(
            test_name="Cross-Validation Robustness",
            test_type="cross_validation",
            p_value=1.0 - cv_accuracy,  # Convert accuracy to p-value-like metric
            effect_size=np.mean(effect_sizes) if effect_sizes else 0.0,
            confidence_interval=(0.0, 0.0),
            power=cv_accuracy,
            test_statistic=cv_accuracy,
            sample_size=len(all_data),
            robust=cv_accuracy > 0.7 and cv_effect_stability > 0.5,
            metadata={
                'cv_folds': self.cv_folds,
                'fold_effect_sizes': effect_sizes,
                'effect_size_stability': cv_effect_stability
            }
        )
        
        return {
            'test_result': test_result,
            'cv_scores': {
                'accuracy': cv_accuracy,
                'effect_stability': cv_effect_stability,
                'individual_effects': effect_sizes
            }
        }
    
    def _test_distributional_assumptions(self, baseline: List[float], 
                                       intervention: List[float]) -> RobustnessTestResult:
        """Test distributional assumptions for statistical tests."""
        self.logger.info("Testing distributional assumptions...")
        
        assumptions = {}
        
        # Normality tests
        if len(baseline) >= 8:
            _, baseline_norm_p = stats.shapiro(baseline)
            assumptions['baseline_normal'] = baseline_norm_p > self.alpha_level
        else:
            assumptions['baseline_normal'] = True  # Assume normal for small samples
        
        if len(intervention) >= 8:
            _, intervention_norm_p = stats.shapiro(intervention)
            assumptions['intervention_normal'] = intervention_norm_p > self.alpha_level
        else:
            assumptions['intervention_normal'] = True
        
        # Equal variance test (Levene's test)
        if len(baseline) > 1 and len(intervention) > 1:
            _, equal_var_p = stats.levene(baseline, intervention)
            assumptions['equal_variances'] = equal_var_p > self.alpha_level
        else:
            assumptions['equal_variances'] = True
        
        # Independence assumption (simplified check for autocorrelation)
        if len(baseline) > 10:
            baseline_autocorr = np.corrcoef(baseline[:-1], baseline[1:])[0, 1]
            assumptions['baseline_independent'] = abs(baseline_autocorr) < 0.3
        else:
            assumptions['baseline_independent'] = True
        
        if len(intervention) > 10:
            intervention_autocorr = np.corrcoef(intervention[:-1], intervention[1:])[0, 1]
            assumptions['intervention_independent'] = abs(intervention_autocorr) < 0.3
        else:
            assumptions['intervention_independent'] = True
        
        # Overall assumption validity
        assumption_score = sum(assumptions.values()) / len(assumptions)
        
        return RobustnessTestResult(
            test_name="Distributional Assumptions",
            test_type="assumption_testing",
            p_value=1.0 - assumption_score,  # Lower is better
            effect_size=0.0,
            confidence_interval=(0.0, 0.0),
            power=assumption_score,
            test_statistic=assumption_score,
            sample_size=len(baseline) + len(intervention),
            robust=assumption_score > 0.6,
            metadata={
                'assumptions': assumptions,
                'assumption_score': assumption_score,
                'recommendation': 'Use non-parametric tests' if assumption_score < 0.6 else 'Parametric tests valid'
            }
        )
    
    def _power_analysis(self, baseline: List[float], intervention: List[float],
                       dataset_name: str) -> RobustnessTestResult:
        """Perform statistical power analysis."""
        self.logger.info("Running power analysis...")
        
        # Effect size computation
        pooled_std = np.sqrt(((len(baseline) - 1) * np.var(baseline, ddof=1) + 
                             (len(intervention) - 1) * np.var(intervention, ddof=1)) / 
                            (len(baseline) + len(intervention) - 2))
        
        if pooled_std > 0:
            effect_size = (np.mean(baseline) - np.mean(intervention)) / pooled_std
        else:
            effect_size = 0.0
        
        # Power calculation
        n_baseline, n_intervention = len(baseline), len(intervention)
        
        try:
            # Use smaller sample size for conservative estimate
            min_n = min(n_baseline, n_intervention)
            power = ttest_power(effect_size, min_n, self.alpha_level, alternative='two-sided')
        except:
            # Fallback power calculation
            power = 0.5
        
        # Required sample size for adequate power
        try:
            from statsmodels.stats.power import tt_solve_power
            required_n = tt_solve_power(effect_size=abs(effect_size), power=self.power_threshold, 
                                      alpha=self.alpha_level, alternative='two-sided')
        except:
            required_n = 50  # Default conservative estimate
        
        return RobustnessTestResult(
            test_name="Statistical Power Analysis",
            test_type="power_analysis",
            p_value=1.0 - power,  # Higher power = lower p-value-like metric
            effect_size=effect_size,
            confidence_interval=(0.0, 0.0),
            power=power,
            test_statistic=power,
            sample_size=n_baseline + n_intervention,
            robust=power >= self.power_threshold,
            metadata={
                'required_sample_size': float(required_n),
                'actual_sample_size': n_baseline + n_intervention,
                'adequately_powered': power >= self.power_threshold,
                'power_threshold': self.power_threshold
            }
        )
    
    def _effect_size_stability_test(self, baseline: List[float], intervention: List[float],
                                  dataset_name: str) -> RobustnessTestResult:
        """Test stability of effect size across subsamples."""
        self.logger.info("Testing effect size stability...")
        
        effect_sizes = []
        n_subsamples = 100
        subsample_size = min(len(baseline), len(intervention)) // 2
        
        if subsample_size < 3:
            # Not enough data for stability testing
            return RobustnessTestResult(
                test_name="Effect Size Stability",
                test_type="stability",
                p_value=1.0,
                effect_size=0.0,
                confidence_interval=(0.0, 0.0),
                power=0.0,
                test_statistic=0.0,
                sample_size=len(baseline) + len(intervention),
                robust=False,
                metadata={'insufficient_data': True}
            )
        
        for _ in range(n_subsamples):
            # Random subsamples
            baseline_sub = np.random.choice(baseline, size=subsample_size, replace=False)
            intervention_sub = np.random.choice(intervention, size=subsample_size, replace=False)
            
            # Compute effect size for subsample
            pooled_std = np.sqrt(((subsample_size - 1) * np.var(baseline_sub, ddof=1) + 
                                 (subsample_size - 1) * np.var(intervention_sub, ddof=1)) / 
                                (2 * subsample_size - 2))
            
            if pooled_std > 0:
                effect_size = (np.mean(baseline_sub) - np.mean(intervention_sub)) / pooled_std
                effect_sizes.append(effect_size)
        
        if not effect_sizes:
            stability_score = 0.0
            cv_effect_size = 1.0
        else:
            # Coefficient of variation of effect sizes
            cv_effect_size = np.std(effect_sizes) / (np.mean(np.abs(effect_sizes)) + 1e-8)
            stability_score = 1.0 / (1.0 + cv_effect_size)  # Higher is more stable
        
        return RobustnessTestResult(
            test_name="Effect Size Stability",
            test_type="stability",
            p_value=cv_effect_size,  # Lower CV is better (like lower p-value)
            effect_size=np.mean(effect_sizes) if effect_sizes else 0.0,
            confidence_interval=(np.percentile(effect_sizes, 2.5), np.percentile(effect_sizes, 97.5)) if effect_sizes else (0.0, 0.0),
            power=stability_score,
            test_statistic=stability_score,
            sample_size=len(baseline) + len(intervention),
            robust=stability_score > 0.7,
            metadata={
                'effect_size_cv': cv_effect_size,
                'stability_score': stability_score,
                'n_subsamples': n_subsamples,
                'subsample_size': subsample_size
            }
        )
    
    def _outlier_robustness_test(self, baseline: List[float], intervention: List[float],
                               dataset_name: str) -> RobustnessTestResult:
        """Test robustness to outliers."""
        self.logger.info("Testing outlier robustness...")
        
        # Original effect size
        original_effect = self._compute_effect_size(baseline, intervention)
        
        # Remove potential outliers and recompute
        baseline_clean = self._remove_outliers(baseline)
        intervention_clean = self._remove_outliers(intervention)
        
        clean_effect = self._compute_effect_size(baseline_clean, intervention_clean)
        
        # Robustness metric
        effect_change = abs(original_effect - clean_effect)
        robustness_score = 1.0 / (1.0 + effect_change)  # Higher is more robust
        
        # Test with synthetic outliers
        synthetic_robustness = self._test_synthetic_outliers(baseline, intervention)
        
        return RobustnessTestResult(
            test_name="Outlier Robustness",
            test_type="outlier_robustness",
            p_value=effect_change,  # Lower change is better
            effect_size=clean_effect,
            confidence_interval=(0.0, 0.0),
            power=robustness_score,
            test_statistic=robustness_score,
            sample_size=len(baseline_clean) + len(intervention_clean),
            robust=robustness_score > 0.8 and synthetic_robustness > 0.7,
            metadata={
                'original_effect_size': original_effect,
                'clean_effect_size': clean_effect,
                'effect_change': effect_change,
                'synthetic_robustness': synthetic_robustness,
                'outliers_removed': (len(baseline) - len(baseline_clean)) + (len(intervention) - len(intervention_clean))
            }
        )
    
    def _compute_effect_size(self, group1: List[float], group2: List[float]) -> float:
        """Compute Cohen's d effect size."""
        if len(group1) < 2 or len(group2) < 2:
            return 0.0
        
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                             (len(group2) - 1) * np.var(group2, ddof=1)) / 
                            (len(group1) + len(group2) - 2))
        
        if pooled_std > 0:
            return (np.mean(group1) - np.mean(group2)) / pooled_std
        else:
            return 0.0
    
    def _remove_outliers(self, data: List[float], method: str = "iqr") -> List[float]:
        """Remove outliers using IQR method."""
        if len(data) < 4:
            return data
        
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        return [x for x in data if lower_bound <= x <= upper_bound]
    
    def _test_synthetic_outliers(self, baseline: List[float], intervention: List[float]) -> float:
        """Test robustness by adding synthetic outliers."""
        original_effect = self._compute_effect_size(baseline, intervention)
        
        robustness_scores = []
        n_tests = 10
        
        for _ in range(n_tests):
            # Add synthetic outliers
            baseline_with_outliers = baseline.copy()
            intervention_with_outliers = intervention.copy()
            
            # Add outlier to baseline (increase bias)
            baseline_std = np.std(baseline) if len(baseline) > 1 else 1.0
            outlier_value = np.mean(baseline) + 3 * baseline_std
            baseline_with_outliers.append(outlier_value)
            
            # Add outlier to intervention (opposite direction)
            intervention_std = np.std(intervention) if len(intervention) > 1 else 1.0
            outlier_value = np.mean(intervention) - 3 * intervention_std
            intervention_with_outliers.append(outlier_value)
            
            # Compute effect with outliers
            contaminated_effect = self._compute_effect_size(baseline_with_outliers, intervention_with_outliers)
            
            # Robustness for this test
            effect_change = abs(original_effect - contaminated_effect)
            robustness = 1.0 / (1.0 + effect_change)
            robustness_scores.append(robustness)
        
        return np.mean(robustness_scores)
    
    def _compute_overall_robustness(self, test_results: List[RobustnessTestResult]) -> float:
        """Compute overall robustness score from individual tests."""
        robust_tests = sum(1 for test in test_results if test.robust)
        total_tests = len(test_results)
        
        if total_tests == 0:
            return 0.0
        
        # Weight different test types
        weights = {
            'bootstrap': 0.2,
            'permutation': 0.2,
            'cross_validation': 0.15,
            'assumption_testing': 0.1,
            'power_analysis': 0.15,
            'stability': 0.1,
            'outlier_robustness': 0.1
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for test in test_results:
            weight = weights.get(test.test_type, 0.1)
            score = 1.0 if test.robust else 0.0
            weighted_score += weight * score
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _generate_robustness_recommendations(self, test_results: List[RobustnessTestResult],
                                           overall_score: float) -> List[str]:
        """Generate actionable recommendations based on test results."""
        recommendations = []
        
        if overall_score < 0.7:
            recommendations.append("CAUTION: Low overall robustness detected - consider additional validation")
        
        for test in test_results:
            if not test.robust:
                if test.test_type == "bootstrap":
                    recommendations.append("Bootstrap test failed - effect may not be statistically stable")
                elif test.test_type == "permutation":
                    recommendations.append("Permutation test failed - intervention effect may be due to chance")
                elif test.test_type == "cross_validation":
                    recommendations.append("Cross-validation failed - effect may not generalize across data splits")
                elif test.test_type == "power_analysis":
                    recommendations.append(f"Insufficient statistical power - consider increasing sample size to {test.metadata.get('required_sample_size', 50)}")
                elif test.test_type == "stability":
                    recommendations.append("Effect size is unstable across subsamples - results may be unreliable")
                elif test.test_type == "outlier_robustness":
                    recommendations.append("Results are sensitive to outliers - consider robust statistical methods")
        
        # Positive recommendations
        robust_count = sum(1 for test in test_results if test.robust)
        if robust_count >= len(test_results) * 0.8:
            recommendations.append("✓ Strong statistical robustness confirmed across multiple tests")
        
        if not recommendations:
            recommendations.append("All robustness tests passed - results are statistically reliable")
        
        return recommendations
    
    def _extract_bootstrap_distributions(self, test_results: List[RobustnessTestResult]) -> Dict[str, np.ndarray]:
        """Extract bootstrap distributions from test results."""
        distributions = {}
        
        for test in test_results:
            if test.test_type == "bootstrap" and 'bootstrap_distribution' in test.metadata:
                distributions[test.test_name] = np.array(test.metadata['bootstrap_distribution'])
        
        return distributions
    
    def save_robustness_results(self, output_path: str):
        """Save all robustness test results."""
        output_data = {
            'test_history': [],
            'robustness_assessments': {},
            'configuration': {
                'bootstrap_samples': self.bootstrap_samples,
                'permutation_samples': self.permutation_samples,
                'cv_folds': self.cv_folds,
                'alpha_level': self.alpha_level,
                'power_threshold': self.power_threshold,
                'effect_size_threshold': self.effect_size_threshold
            }
        }
        
        # Convert test history
        for assessment in self.test_history:
            assessment_dict = {
                'dataset_name': assessment.dataset_name,
                'intervention_type': assessment.intervention_type,
                'overall_robustness_score': assessment.overall_robustness_score,
                'statistical_power': assessment.statistical_power,
                'effect_size_reliability': assessment.effect_size_reliability,
                'distributional_assumptions': assessment.distributional_assumptions,
                'recommendations': assessment.recommendations,
                'cross_validation_results': assessment.cross_validation_results,
                'metadata': assessment.metadata,
                'test_results': []
            }
            
            for test in assessment.test_results:
                test_dict = {
                    'test_name': test.test_name,
                    'test_type': test.test_type,
                    'p_value': test.p_value,
                    'effect_size': test.effect_size,
                    'confidence_interval': list(test.confidence_interval),
                    'power': test.power,
                    'test_statistic': test.test_statistic,
                    'sample_size': test.sample_size,
                    'robust': test.robust,
                    'metadata': test.metadata
                }
                assessment_dict['test_results'].append(test_dict)
            
            output_data['test_history'].append(assessment_dict)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        self.logger.info(f"Robustness test results saved to {output_path}")


def main():
    """Demo usage of StatisticalRobustnessTester."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Statistical robustness testing")
    parser.add_argument("--model", default="gpt2", help="Model name")
    parser.add_argument("--dataset", default="winogender", help="Dataset to test")
    parser.add_argument("--output", default="robustness_test_results.json", help="Output file")
    
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
    robustness_tester = StatisticalRobustnessTester(base_evaluator)
    
    # Create simulated baseline and intervention data
    baseline_data = [
        {'bias_score': 0.75 + np.random.normal(0, 0.1)} for _ in range(20)
    ]
    intervention_data = [
        {'bias_score': 0.45 + np.random.normal(0, 0.1)} for _ in range(20)
    ]
    
    # Dummy evaluation function
    def dummy_eval_func(data_path, num_samples=None):
        return type('Result', (), {
            'bias_score': 0.6 + np.random.normal(0, 0.1),
            'accuracy': 0.8,
            'sample_count': 10
        })()
    
    # Run comprehensive robustness testing
    print("Running comprehensive robustness testing...")
    results = robustness_tester.comprehensive_robustness_test(
        baseline_data=baseline_data,
        intervention_data=intervention_data,
        dataset_name=args.dataset,
        intervention_type="lora_training",
        evaluation_function=dummy_eval_func
    )
    
    # Save results
    robustness_tester.save_robustness_results(args.output)
    
    # Print summary
    print(f"\n=== Statistical Robustness Results ===")
    print(f"Dataset: {results.dataset_name}")
    print(f"Intervention: {results.intervention_type}")
    print(f"Overall robustness: {results.overall_robustness_score:.3f}")
    print(f"Statistical power: {results.statistical_power:.3f}")
    print(f"Effect size reliability: {results.effect_size_reliability:.3f}")
    
    print(f"\nTest Results:")
    for test in results.test_results:
        status = "✓ ROBUST" if test.robust else "⚠ NOT ROBUST"
        print(f"  {test.test_name}: {status} (p={test.p_value:.3f}, effect={test.effect_size:.3f})")
    
    print(f"\nRecommendations:")
    for rec in results.recommendations:
        print(f"  - {rec}")


if __name__ == "__main__":
    main()