#!/usr/bin/env python3
"""
Longitudinal Intervention Monitoring for Phase 4: Validation & Robustness
Tracks actual intervention effects over time with real persistence analysis.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import time
from scipy import stats
from collections import defaultdict, deque
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


@dataclass
class InterventionSnapshot:
    """Snapshot of intervention effects at a specific time."""
    timestamp: datetime
    intervention_type: str  # "lora", "steering", "combined"
    model_state: str  # "baseline", "trained", "steered"
    bias_scores: Dict[str, float]
    accuracy_scores: Dict[str, float]
    intervention_strength: float
    sample_count: int
    evaluation_metrics: Dict[str, Any]
    model_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PersistenceAnalysis:
    """Analysis of intervention persistence over time."""
    intervention_type: str
    time_span_hours: float
    initial_effect: float
    final_effect: float
    persistence_score: float  # 0-1, higher is more persistent
    decay_rate: float  # Effect decay per hour
    stability_metrics: Dict[str, float]
    trend_analysis: Dict[str, Any]
    statistical_significance: Dict[str, float]


@dataclass
class LongitudinalResults:
    """Complete longitudinal monitoring results."""
    monitoring_period: Tuple[datetime, datetime]
    total_snapshots: int
    interventions_tracked: List[str]
    persistence_analyses: Dict[str, PersistenceAnalysis]
    drift_detection: Dict[str, Any]
    robustness_over_time: Dict[str, float]
    recommendations: List[str]
    metadata: Dict[str, Any]


class LongitudinalInterventionMonitor:
    """
    Monitor intervention effects over time to assess persistence and drift.
    Provides real analysis of how bias mitigation maintains effectiveness.
    """
    
    def __init__(self, base_evaluator, logger: Optional[logging.Logger] = None):
        """
        Initialize longitudinal intervention monitor.
        
        Args:
            base_evaluator: Base bias evaluator for taking snapshots
            logger: Optional logger
        """
        self.base_evaluator = base_evaluator
        self.logger = logger or logging.getLogger(__name__)
        
        # Monitoring state
        self.snapshots = deque(maxlen=1000)  # Keep last 1000 snapshots
        self.intervention_history = defaultdict(list)
        self.baseline_references = {}
        
        # Configuration
        self.min_snapshots_for_analysis = 5
        self.drift_threshold = 0.1  # 10% change indicates drift
        self.persistence_threshold = 0.7  # 70% retention is "persistent"
        
        # Time tracking
        self.monitoring_start = datetime.now()
        self.last_snapshot_time = None
        
        self.logger.info("Initialized LongitudinalInterventionMonitor")
    
    def take_snapshot(self,
                     dataset_path: str,
                     dataset_name: str,
                     evaluation_function: callable,
                     intervention_type: str = "unknown",
                     model_state: str = "unknown",
                     intervention_strength: float = 1.0,
                     num_samples: Optional[int] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> InterventionSnapshot:
        """
        Take a snapshot of current intervention effects.
        
        Args:
            dataset_path: Path to evaluation dataset
            dataset_name: Name of dataset
            evaluation_function: Function to evaluate bias
            intervention_type: Type of intervention applied
            model_state: Current model state
            intervention_strength: Strength of intervention
            num_samples: Number of samples to evaluate
            metadata: Additional metadata
            
        Returns:
            InterventionSnapshot with current state
        """
        self.logger.info(f"Taking longitudinal snapshot: {intervention_type} on {dataset_name}")
        
        timestamp = datetime.now()
        
        try:
            # Evaluate current state
            eval_result = evaluation_function(dataset_path, num_samples)
            
            # Extract metrics
            bias_scores = {
                'primary_bias_score': eval_result.bias_score,
                'secondary_metrics': {}
            }
            
            accuracy_scores = {
                'overall_accuracy': eval_result.accuracy
            }
            
            # Extract additional metrics from metadata
            if hasattr(eval_result, 'metadata') and eval_result.metadata:
                for key, value in eval_result.metadata.items():
                    if isinstance(value, (int, float)):
                        if any(bias_term in key.lower() for bias_term in ['bias', 'score', 'rate']):
                            bias_scores['secondary_metrics'][key] = float(value)
                        elif 'accuracy' in key.lower():
                            accuracy_scores[key] = float(value)
            
            # Create snapshot
            snapshot = InterventionSnapshot(
                timestamp=timestamp,
                intervention_type=intervention_type,
                model_state=model_state,
                bias_scores=bias_scores,
                accuracy_scores=accuracy_scores,
                intervention_strength=intervention_strength,
                sample_count=eval_result.sample_count,
                evaluation_metrics={
                    'confidence_interval': getattr(eval_result, 'confidence_interval', (0.0, 0.0)),
                    'statistical_significance': getattr(eval_result, 'statistical_significance', {}),
                    'individual_scores': getattr(eval_result, 'individual_scores', [])
                },
                model_metadata=metadata or {}
            )
            
            # Store snapshot
            self.snapshots.append(snapshot)
            self.intervention_history[intervention_type].append(snapshot)
            self.last_snapshot_time = timestamp
            
            # Update baseline if this is a baseline measurement
            if model_state == "baseline":
                self.baseline_references[dataset_name] = snapshot
            
            self.logger.info(f"Snapshot captured: bias={eval_result.bias_score:.4f}, accuracy={eval_result.accuracy:.4f}")
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Failed to take snapshot: {e}")
            raise
    
    def analyze_persistence(self, 
                          intervention_type: str,
                          time_window_hours: Optional[float] = None) -> Optional[PersistenceAnalysis]:
        """
        Analyze persistence of intervention effects over time.
        
        Args:
            intervention_type: Type of intervention to analyze
            time_window_hours: Analysis window (None for all available data)
            
        Returns:
            PersistenceAnalysis or None if insufficient data
        """
        snapshots = self.intervention_history.get(intervention_type, [])
        
        if len(snapshots) < self.min_snapshots_for_analysis:
            self.logger.warning(f"Insufficient snapshots for {intervention_type}: {len(snapshots)}")
            return None
        
        # Filter by time window if specified
        if time_window_hours:
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            snapshots = [s for s in snapshots if s.timestamp >= cutoff_time]
        
        if len(snapshots) < 2:
            return None
        
        # Sort by timestamp
        snapshots = sorted(snapshots, key=lambda x: x.timestamp)
        
        # Extract time series data
        timestamps = [(s.timestamp - snapshots[0].timestamp).total_seconds() / 3600 for s in snapshots]  # Hours
        bias_scores = [s.bias_scores['primary_bias_score'] for s in snapshots]
        accuracy_scores = [s.accuracy_scores['overall_accuracy'] for s in snapshots]
        
        # Calculate persistence metrics
        initial_effect = bias_scores[0]
        final_effect = bias_scores[-1]
        time_span = timestamps[-1]
        
        # Compute decay rate (linear regression)
        if len(timestamps) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps, bias_scores)
            decay_rate = abs(slope)  # Absolute decay per hour
        else:
            decay_rate = abs(final_effect - initial_effect) / max(time_span, 1e-6)
        
        # Persistence score (how much effect remains)
        if initial_effect != 0:
            persistence_score = min(1.0, abs(final_effect) / abs(initial_effect))
        else:
            persistence_score = 1.0 if abs(final_effect) < 0.01 else 0.0
        
        # Stability metrics
        stability_metrics = self._compute_stability_metrics(bias_scores, timestamps)
        
        # Trend analysis
        trend_analysis = self._analyze_trends(timestamps, bias_scores, accuracy_scores)
        
        # Statistical significance of persistence
        statistical_significance = self._test_persistence_significance(bias_scores, timestamps)
        
        return PersistenceAnalysis(
            intervention_type=intervention_type,
            time_span_hours=time_span,
            initial_effect=initial_effect,
            final_effect=final_effect,
            persistence_score=persistence_score,
            decay_rate=decay_rate,
            stability_metrics=stability_metrics,
            trend_analysis=trend_analysis,
            statistical_significance=statistical_significance
        )
    
    def detect_drift(self, dataset_name: str, lookback_hours: float = 24.0) -> Dict[str, Any]:
        """
        Detect bias drift over recent time period.
        
        Args:
            dataset_name: Dataset to analyze for drift
            lookback_hours: How far back to look for drift detection
            
        Returns:
            Dictionary with drift analysis results
        """
        # Get recent snapshots
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        recent_snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_time]
        
        if len(recent_snapshots) < 3:
            return {
                'drift_detected': False,
                'reason': 'insufficient_data',
                'snapshots_analyzed': len(recent_snapshots)
            }
        
        # Get baseline reference
        baseline = self.baseline_references.get(dataset_name)
        if not baseline:
            return {
                'drift_detected': False,
                'reason': 'no_baseline_reference',
                'snapshots_analyzed': len(recent_snapshots)
            }
        
        # Analyze drift
        recent_bias_scores = [s.bias_scores['primary_bias_score'] for s in recent_snapshots]
        baseline_score = baseline.bias_scores['primary_bias_score']
        
        # Statistical tests for drift
        mean_recent = np.mean(recent_bias_scores)
        drift_magnitude = abs(mean_recent - baseline_score)
        drift_percentage = drift_magnitude / (abs(baseline_score) + 1e-8) * 100
        
        # Trend test
        timestamps = [(s.timestamp - recent_snapshots[0].timestamp).total_seconds() / 3600 
                     for s in recent_snapshots]
        
        if len(timestamps) > 2:
            slope, _, r_value, p_value, _ = stats.linregress(timestamps, recent_bias_scores)
            trend_significant = p_value < 0.05
        else:
            slope, r_value, p_value, trend_significant = 0.0, 0.0, 1.0, False
        
        # Drift detection logic
        drift_detected = (
            drift_magnitude > self.drift_threshold or
            drift_percentage > 10.0 or
            (trend_significant and abs(slope) > 0.01)
        )
        
        return {
            'drift_detected': drift_detected,
            'drift_magnitude': float(drift_magnitude),
            'drift_percentage': float(drift_percentage),
            'baseline_score': float(baseline_score),
            'recent_mean_score': float(mean_recent),
            'trend_slope': float(slope),
            'trend_r_squared': float(r_value**2),
            'trend_p_value': float(p_value),
            'trend_significant': trend_significant,
            'snapshots_analyzed': len(recent_snapshots),
            'time_span_hours': lookback_hours,
            'recommendation': self._get_drift_recommendation(drift_detected, drift_magnitude, trend_significant)
        }
    
    def generate_longitudinal_report(self, 
                                   analysis_period_hours: Optional[float] = None) -> LongitudinalResults:
        """
        Generate comprehensive longitudinal monitoring report.
        
        Args:
            analysis_period_hours: Period to analyze (None for full history)
            
        Returns:
            LongitudinalResults with complete analysis
        """
        self.logger.info("Generating longitudinal monitoring report")
        
        # Filter snapshots by time period
        if analysis_period_hours:
            cutoff_time = datetime.now() - timedelta(hours=analysis_period_hours)
            analysis_snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_time]
        else:
            analysis_snapshots = list(self.snapshots)
        
        if not analysis_snapshots:
            raise ValueError("No snapshots available for analysis")
        
        # Determine analysis period
        start_time = min(s.timestamp for s in analysis_snapshots)
        end_time = max(s.timestamp for s in analysis_snapshots)
        
        # Analyze persistence for each intervention type
        intervention_types = set(s.intervention_type for s in analysis_snapshots)
        persistence_analyses = {}
        
        for intervention_type in intervention_types:
            analysis = self.analyze_persistence(intervention_type, analysis_period_hours)
            if analysis:
                persistence_analyses[intervention_type] = analysis
        
        # Detect drift for all datasets
        datasets = set(s.model_metadata.get('dataset_name', 'unknown') for s in analysis_snapshots)
        drift_detection = {}
        
        for dataset in datasets:
            if dataset != 'unknown':
                drift_detection[dataset] = self.detect_drift(dataset, analysis_period_hours or 24.0)
        
        # Compute robustness over time
        robustness_metrics = self._compute_robustness_over_time(analysis_snapshots)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(persistence_analyses, drift_detection, robustness_metrics)
        
        return LongitudinalResults(
            monitoring_period=(start_time, end_time),
            total_snapshots=len(analysis_snapshots),
            interventions_tracked=list(intervention_types),
            persistence_analyses=persistence_analyses,
            drift_detection=drift_detection,
            robustness_over_time=robustness_metrics,
            recommendations=recommendations,
            metadata={
                'analysis_period_hours': analysis_period_hours,
                'min_snapshots_required': self.min_snapshots_for_analysis,
                'drift_threshold': self.drift_threshold,
                'persistence_threshold': self.persistence_threshold
            }
        )
    
    def _compute_stability_metrics(self, values: List[float], timestamps: List[float]) -> Dict[str, float]:
        """Compute stability metrics for a time series."""
        if len(values) < 2:
            return {}
        
        # Coefficient of variation
        cv = np.std(values) / (np.mean(values) + 1e-8)
        
        # Maximum excursion from mean
        mean_val = np.mean(values)
        max_deviation = max(abs(v - mean_val) for v in values)
        
        # Autocorrelation (if enough points)
        autocorr = 0.0
        if len(values) > 5:
            autocorr = np.corrcoef(values[:-1], values[1:])[0, 1]
            autocorr = autocorr if not np.isnan(autocorr) else 0.0
        
        return {
            'coefficient_of_variation': float(cv),
            'max_deviation_from_mean': float(max_deviation),
            'autocorrelation': float(autocorr),
            'stability_score': float(1.0 / (1.0 + cv))  # Higher is more stable
        }
    
    def _analyze_trends(self, timestamps: List[float], 
                       bias_scores: List[float], 
                       accuracy_scores: List[float]) -> Dict[str, Any]:
        """Analyze trends in the time series data."""
        if len(timestamps) < 3:
            return {}
        
        # Linear trends
        bias_slope, bias_intercept, bias_r, bias_p, _ = stats.linregress(timestamps, bias_scores)
        acc_slope, acc_intercept, acc_r, acc_p, _ = stats.linregress(timestamps, accuracy_scores)
        
        # Trend direction
        bias_trend = "increasing" if bias_slope > 0.001 else "decreasing" if bias_slope < -0.001 else "stable"
        acc_trend = "increasing" if acc_slope > 0.001 else "decreasing" if acc_slope < -0.001 else "stable"
        
        return {
            'bias_trend': {
                'direction': bias_trend,
                'slope': float(bias_slope),
                'r_squared': float(bias_r**2),
                'p_value': float(bias_p),
                'significant': bias_p < 0.05
            },
            'accuracy_trend': {
                'direction': acc_trend,
                'slope': float(acc_slope),
                'r_squared': float(acc_r**2),
                'p_value': float(acc_p),
                'significant': acc_p < 0.05
            }
        }
    
    def _test_persistence_significance(self, values: List[float], timestamps: List[float]) -> Dict[str, float]:
        """Test statistical significance of persistence."""
        if len(values) < 3:
            return {}
        
        # Test if final values are significantly different from initial
        initial_third = values[:len(values)//3]
        final_third = values[-len(values)//3:]
        
        if len(initial_third) > 1 and len(final_third) > 1:
            t_stat, p_value = stats.ttest_ind(initial_third, final_third)
        else:
            t_stat, p_value = 0.0, 1.0
        
        # Test for monotonic trend
        if len(values) > 5:
            tau, tau_p = stats.kendalltau(timestamps, values)
        else:
            tau, tau_p = 0.0, 1.0
        
        return {
            'initial_vs_final_p_value': float(p_value),
            'initial_vs_final_significant': p_value < 0.05,
            'kendall_tau': float(tau),
            'kendall_p_value': float(tau_p),
            'monotonic_trend': tau_p < 0.05
        }
    
    def _compute_robustness_over_time(self, snapshots: List[InterventionSnapshot]) -> Dict[str, float]:
        """Compute robustness metrics over time."""
        if len(snapshots) < 2:
            return {}
        
        # Group by intervention type
        intervention_groups = defaultdict(list)
        for snapshot in snapshots:
            intervention_groups[snapshot.intervention_type].append(snapshot)
        
        robustness_metrics = {}
        
        for intervention_type, group_snapshots in intervention_groups.items():
            if len(group_snapshots) >= 2:
                bias_scores = [s.bias_scores['primary_bias_score'] for s in group_snapshots]
                accuracy_scores = [s.accuracy_scores['overall_accuracy'] for s in group_snapshots]
                
                # Temporal consistency
                bias_cv = np.std(bias_scores) / (np.mean(bias_scores) + 1e-8)
                acc_cv = np.std(accuracy_scores) / (np.mean(accuracy_scores) + 1e-8)
                
                robustness_metrics[f'{intervention_type}_bias_consistency'] = float(1.0 / (1.0 + bias_cv))
                robustness_metrics[f'{intervention_type}_accuracy_consistency'] = float(1.0 / (1.0 + acc_cv))
        
        return robustness_metrics
    
    def _get_drift_recommendation(self, drift_detected: bool, drift_magnitude: float, trend_significant: bool) -> str:
        """Generate recommendation based on drift analysis."""
        if not drift_detected:
            return "No action required - intervention effects are stable"
        
        if drift_magnitude > 0.2:
            return "URGENT: Significant bias drift detected - consider retraining or strengthening intervention"
        elif trend_significant:
            return "WARNING: Trending bias drift detected - monitor closely and consider intervention adjustment"
        else:
            return "CAUTION: Moderate drift detected - increase monitoring frequency"
    
    def _generate_recommendations(self, 
                                persistence_analyses: Dict[str, PersistenceAnalysis],
                                drift_detection: Dict[str, Any],
                                robustness_metrics: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations from longitudinal analysis."""
        recommendations = []
        
        # Persistence-based recommendations
        for intervention_type, analysis in persistence_analyses.items():
            if analysis.persistence_score < self.persistence_threshold:
                recommendations.append(
                    f"Consider strengthening {intervention_type} intervention "
                    f"(persistence: {analysis.persistence_score:.2f})"
                )
            
            if analysis.decay_rate > 0.05:  # High decay rate
                recommendations.append(
                    f"High decay rate detected for {intervention_type} "
                    f"({analysis.decay_rate:.4f}/hour) - consider periodic reapplication"
                )
        
        # Drift-based recommendations
        for dataset, drift_info in drift_detection.items():
            if drift_info['drift_detected']:
                recommendations.append(drift_info['recommendation'])
        
        # Robustness-based recommendations
        for metric, value in robustness_metrics.items():
            if 'consistency' in metric and value < 0.7:
                intervention = metric.split('_')[0]
                recommendations.append(
                    f"Low temporal consistency for {intervention} - investigate variability sources"
                )
        
        if not recommendations:
            recommendations.append("All interventions showing good stability and persistence")
        
        return recommendations
    
    def save_monitoring_data(self, output_path: str):
        """Save all monitoring data and snapshots."""
        output_data = {
            'monitoring_start': self.monitoring_start.isoformat(),
            'last_snapshot': self.last_snapshot_time.isoformat() if self.last_snapshot_time else None,
            'snapshots': [],
            'intervention_history': {},
            'baseline_references': {},
            'configuration': {
                'min_snapshots_for_analysis': self.min_snapshots_for_analysis,
                'drift_threshold': self.drift_threshold,
                'persistence_threshold': self.persistence_threshold
            }
        }
        
        # Convert snapshots to serializable format
        for snapshot in self.snapshots:
            snapshot_dict = {
                'timestamp': snapshot.timestamp.isoformat(),
                'intervention_type': snapshot.intervention_type,
                'model_state': snapshot.model_state,
                'bias_scores': snapshot.bias_scores,
                'accuracy_scores': snapshot.accuracy_scores,
                'intervention_strength': snapshot.intervention_strength,
                'sample_count': snapshot.sample_count,
                'evaluation_metrics': snapshot.evaluation_metrics,
                'model_metadata': snapshot.model_metadata
            }
            output_data['snapshots'].append(snapshot_dict)
        
        # Convert intervention history
        for intervention_type, snapshots in self.intervention_history.items():
            output_data['intervention_history'][intervention_type] = [
                s.timestamp.isoformat() for s in snapshots
            ]
        
        # Convert baseline references
        for dataset, snapshot in self.baseline_references.items():
            output_data['baseline_references'][dataset] = {
                'timestamp': snapshot.timestamp.isoformat(),
                'bias_score': snapshot.bias_scores['primary_bias_score']
            }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        self.logger.info(f"Monitoring data saved to {output_path}")


def main():
    """Demo usage of LongitudinalInterventionMonitor."""
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="Longitudinal intervention monitoring")
    parser.add_argument("--model", default="gpt2", help="Model name")
    parser.add_argument("--dataset", default="winogender", help="Dataset to monitor")
    parser.add_argument("--snapshots", type=int, default=5, help="Number of snapshots to take")
    parser.add_argument("--interval", type=int, default=60, help="Interval between snapshots (seconds)")
    parser.add_argument("--output", default="longitudinal_monitoring.json", help="Output file")
    
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
    monitor = LongitudinalInterventionMonitor(base_evaluator)
    
    # Determine evaluation function
    dataset_path = f"datasets/{args.dataset}"
    if args.dataset == "winogender":
        eval_func = base_evaluator.evaluate_winogender
    elif args.dataset == "truthfulqa":
        eval_func = base_evaluator.evaluate_truthfulqa
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Take baseline snapshot
    print("Taking baseline snapshot...")
    monitor.take_snapshot(
        dataset_path=dataset_path,
        dataset_name=args.dataset,
        evaluation_function=eval_func,
        intervention_type="baseline",
        model_state="baseline",
        num_samples=5
    )
    
    # Simulate intervention snapshots over time
    print(f"Taking {args.snapshots} intervention snapshots with {args.interval}s intervals...")
    for i in range(args.snapshots):
        time.sleep(args.interval)
        
        monitor.take_snapshot(
            dataset_path=dataset_path,
            dataset_name=args.dataset,
            evaluation_function=eval_func,
            intervention_type="simulated_intervention",
            model_state="post_intervention",
            intervention_strength=1.0 - (i * 0.1),  # Simulate decay
            num_samples=5,
            metadata={'snapshot_number': i + 1}
        )
        
        print(f"Snapshot {i + 1}/{args.snapshots} completed")
    
    # Generate report
    print("Generating longitudinal report...")
    report = monitor.generate_longitudinal_report()
    
    # Save results
    monitor.save_monitoring_data(args.output)
    
    # Print summary
    print(f"\n=== Longitudinal Monitoring Results ===")
    print(f"Monitoring period: {report.monitoring_period[0]} to {report.monitoring_period[1]}")
    print(f"Total snapshots: {report.total_snapshots}")
    print(f"Interventions tracked: {report.interventions_tracked}")
    
    for intervention, analysis in report.persistence_analyses.items():
        print(f"\n{intervention} Analysis:")
        print(f"  Persistence score: {analysis.persistence_score:.4f}")
        print(f"  Decay rate: {analysis.decay_rate:.6f}/hour")
        print(f"  Initial effect: {analysis.initial_effect:.4f}")
        print(f"  Final effect: {analysis.final_effect:.4f}")
    
    print(f"\nRecommendations:")
    for rec in report.recommendations:
        print(f"  - {rec}")


if __name__ == "__main__":
    main()