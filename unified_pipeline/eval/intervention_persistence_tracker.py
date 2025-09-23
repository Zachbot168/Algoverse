#!/usr/bin/env python3
"""
Comprehensive Intervention Persistence Tracker for Phase 4: Validation & Robustness
Advanced tracking system for monitoring long-term intervention effectiveness and decay patterns.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from pathlib import Path
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import time
from scipy import stats
from scipy.optimize import curve_fit
from collections import defaultdict, deque
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pickle

# Suppress warnings
warnings.filterwarnings('ignore')


@dataclass
class PersistenceSnapshot:
    """Extended snapshot with persistence-specific metrics."""
    timestamp: datetime
    intervention_type: str
    model_state: str
    persistence_phase: str  # "immediate", "short_term", "medium_term", "long_term"
    effectiveness_score: float
    decay_rate: float
    stability_metrics: Dict[str, float]
    environmental_factors: Dict[str, Any]
    intervention_strength: float
    bias_scores: Dict[str, float]
    accuracy_scores: Dict[str, float]
    sample_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecayModel:
    """Model for intervention decay patterns."""
    model_type: str  # "exponential", "linear", "polynomial", "custom"
    parameters: Dict[str, float]
    r_squared: float
    prediction_accuracy: float
    half_life: Optional[float]  # Time for 50% effectiveness loss
    confidence_interval: Tuple[float, float]
    model_equation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PersistenceAnalysis:
    """Comprehensive persistence analysis results."""
    intervention_type: str
    analysis_period: Tuple[datetime, datetime]
    total_snapshots: int
    decay_model: DecayModel
    persistence_phases: Dict[str, Dict[str, float]]
    critical_thresholds: Dict[str, float]
    maintenance_recommendations: List[str]
    reapplication_schedule: Dict[str, Any]
    risk_factors: Dict[str, float]
    resilience_score: float
    long_term_viability: bool
    metadata: Dict[str, Any]


class InterventionPersistenceTracker:
    """
    Advanced persistence tracking system that monitors intervention effectiveness
    over extended periods and provides predictive decay modeling.
    """
    
    def __init__(self, base_evaluator, logger: Optional[logging.Logger] = None):
        """
        Initialize intervention persistence tracker.
        
        Args:
            base_evaluator: Base bias evaluator
            logger: Optional logger
        """
        self.base_evaluator = base_evaluator
        self.logger = logger or logging.getLogger(__name__)
        
        # Extended tracking storage
        self.persistence_snapshots = deque(maxlen=10000)  # Store more history
        self.intervention_timelines = defaultdict(list)
        self.decay_models = {}
        self.threshold_alerts = defaultdict(list)
        
        # Persistence phase definitions (in hours)
        self.phase_definitions = {
            'immediate': (0, 2),      # 0-2 hours
            'short_term': (2, 24),    # 2-24 hours  
            'medium_term': (24, 168), # 1-7 days
            'long_term': (168, 720)   # 1-4 weeks
        }
        
        # Critical thresholds
        self.effectiveness_threshold = 0.7  # Below this = intervention failing
        self.decay_rate_alert = 0.1         # Above this = rapid decay
        self.stability_threshold = 0.8      # Below this = unstable
        
        # Configuration
        self.min_snapshots_for_modeling = 10
        self.prediction_horizon_hours = 168  # 1 week ahead
        self.auto_alert_enabled = True
        
        self.logger.info("Initialized InterventionPersistenceTracker")
    
    def track_persistence_snapshot(self,
                                 dataset_path: str,
                                 dataset_name: str,
                                 evaluation_function: Callable,
                                 intervention_type: str,
                                 model_state: str = "post_intervention",
                                 intervention_strength: float = 1.0,
                                 environmental_factors: Optional[Dict[str, Any]] = None,
                                 num_samples: Optional[int] = None) -> PersistenceSnapshot:
        """
        Take a comprehensive persistence tracking snapshot.
        
        Args:
            dataset_path: Path to evaluation dataset
            dataset_name: Name of dataset
            evaluation_function: Function to evaluate bias
            intervention_type: Type of intervention being tracked
            model_state: Current model state
            intervention_strength: Current intervention strength
            environmental_factors: External factors that might affect persistence
            num_samples: Number of samples to evaluate
            
        Returns:
            PersistenceSnapshot with comprehensive metrics
        """
        self.logger.info(f"Taking persistence snapshot for {intervention_type}")
        
        timestamp = datetime.now()
        
        try:
            # Get current evaluation
            eval_result = evaluation_function(dataset_path, num_samples)
            
            # Determine persistence phase
            persistence_phase = self._determine_persistence_phase(intervention_type, timestamp)
            
            # Compute effectiveness score
            effectiveness_score = self._compute_effectiveness_score(eval_result, intervention_type)
            
            # Compute current decay rate
            decay_rate = self._compute_current_decay_rate(intervention_type, effectiveness_score)
            
            # Compute stability metrics
            stability_metrics = self._compute_stability_metrics(intervention_type, eval_result)
            
            # Extract bias and accuracy scores
            bias_scores = {
                'primary_bias_score': eval_result.bias_score,
                'normalized_score': self._normalize_bias_score(eval_result.bias_score)
            }
            
            accuracy_scores = {
                'overall_accuracy': eval_result.accuracy,
                'accuracy_retention': self._compute_accuracy_retention(eval_result.accuracy, intervention_type)
            }
            
            # Create snapshot
            snapshot = PersistenceSnapshot(
                timestamp=timestamp,
                intervention_type=intervention_type,
                model_state=model_state,
                persistence_phase=persistence_phase,
                effectiveness_score=effectiveness_score,
                decay_rate=decay_rate,
                stability_metrics=stability_metrics,
                environmental_factors=environmental_factors or {},
                intervention_strength=intervention_strength,
                bias_scores=bias_scores,
                accuracy_scores=accuracy_scores,
                sample_count=eval_result.sample_count,
                metadata={
                    'dataset_name': dataset_name,
                    'evaluation_metadata': getattr(eval_result, 'metadata', {}),
                    'snapshot_id': len(self.persistence_snapshots)
                }
            )
            
            # Store snapshot
            self.persistence_snapshots.append(snapshot)
            self.intervention_timelines[intervention_type].append(snapshot)
            
            # Check for alerts
            if self.auto_alert_enabled:
                self._check_persistence_alerts(snapshot)
            
            self.logger.info(f"Persistence snapshot completed: effectiveness={effectiveness_score:.3f}, decay_rate={decay_rate:.4f}")
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Failed to take persistence snapshot: {e}")
            raise
    
    def analyze_long_term_persistence(self,
                                    intervention_type: str,
                                    analysis_period_hours: Optional[float] = None) -> PersistenceAnalysis:
        """
        Perform comprehensive long-term persistence analysis.
        
        Args:
            intervention_type: Type of intervention to analyze
            analysis_period_hours: Analysis window (None for all available data)
            
        Returns:
            PersistenceAnalysis with comprehensive insights
        """
        self.logger.info(f"Analyzing long-term persistence for {intervention_type}")
        
        # Get relevant snapshots
        snapshots = self._get_relevant_snapshots(intervention_type, analysis_period_hours)
        
        if len(snapshots) < self.min_snapshots_for_modeling:
            raise ValueError(f"Insufficient snapshots for analysis: {len(snapshots)} < {self.min_snapshots_for_modeling}")
        
        # Build decay model
        decay_model = self._build_decay_model(snapshots)
        
        # Analyze persistence phases
        persistence_phases = self._analyze_persistence_phases(snapshots)
        
        # Identify critical thresholds
        critical_thresholds = self._identify_critical_thresholds(snapshots, decay_model)
        
        # Generate maintenance recommendations
        maintenance_recommendations = self._generate_maintenance_recommendations(
            snapshots, decay_model, critical_thresholds
        )
        
        # Create reapplication schedule
        reapplication_schedule = self._create_reapplication_schedule(decay_model, critical_thresholds)
        
        # Assess risk factors
        risk_factors = self._assess_risk_factors(snapshots)
        
        # Compute resilience score
        resilience_score = self._compute_resilience_score(snapshots, decay_model)
        
        # Determine long-term viability
        long_term_viability = self._assess_long_term_viability(decay_model, resilience_score)
        
        # Create analysis period
        analysis_period = (snapshots[0].timestamp, snapshots[-1].timestamp)
        
        return PersistenceAnalysis(
            intervention_type=intervention_type,
            analysis_period=analysis_period,
            total_snapshots=len(snapshots),
            decay_model=decay_model,
            persistence_phases=persistence_phases,
            critical_thresholds=critical_thresholds,
            maintenance_recommendations=maintenance_recommendations,
            reapplication_schedule=reapplication_schedule,
            risk_factors=risk_factors,
            resilience_score=resilience_score,
            long_term_viability=long_term_viability,
            metadata={
                'analysis_timestamp': datetime.now().isoformat(),
                'analysis_period_hours': analysis_period_hours,
                'model_confidence': decay_model.r_squared
            }
        )
    
    def _determine_persistence_phase(self, intervention_type: str, current_time: datetime) -> str:
        """Determine which persistence phase we're in."""
        intervention_snapshots = self.intervention_timelines.get(intervention_type, [])
        
        if not intervention_snapshots:
            return "immediate"
        
        # Find the most recent intervention application
        intervention_start = intervention_snapshots[0].timestamp
        hours_elapsed = (current_time - intervention_start).total_seconds() / 3600
        
        for phase, (start_hours, end_hours) in self.phase_definitions.items():
            if start_hours <= hours_elapsed < end_hours:
                return phase
        
        return "long_term"  # Beyond defined phases
    
    def _compute_effectiveness_score(self, eval_result, intervention_type: str) -> float:
        """Compute current intervention effectiveness score."""
        # Get baseline reference if available
        baseline_score = self._get_baseline_reference(intervention_type)
        
        if baseline_score is None:
            # Use simple normalized score
            return 1.0 - eval_result.bias_score
        
        # Compute improvement ratio
        current_score = eval_result.bias_score
        improvement = (baseline_score - current_score) / (baseline_score + 1e-8)
        
        # Normalize to 0-1 scale
        effectiveness = max(0.0, min(1.0, improvement))
        return effectiveness
    
    def _compute_current_decay_rate(self, intervention_type: str, current_effectiveness: float) -> float:
        """Compute current decay rate based on recent trend."""
        snapshots = self.intervention_timelines.get(intervention_type, [])
        
        if len(snapshots) < 3:
            return 0.0
        
        # Get recent snapshots (last 5 or all if fewer)
        recent_snapshots = snapshots[-5:]
        
        # Extract time and effectiveness data
        times = [(s.timestamp - recent_snapshots[0].timestamp).total_seconds() / 3600 
                for s in recent_snapshots]
        effectiveness_scores = [s.effectiveness_score for s in recent_snapshots]
        
        # Compute decay rate using linear regression
        if len(times) > 1:
            try:
                slope, _, _, _, _ = stats.linregress(times, effectiveness_scores)
                decay_rate = abs(slope) if slope < 0 else 0.0  # Only consider decay (negative slope)
                return decay_rate
            except:
                return 0.0
        
        return 0.0
    
    def _compute_stability_metrics(self, intervention_type: str, eval_result) -> Dict[str, float]:
        """Compute stability metrics for current state."""
        snapshots = self.intervention_timelines.get(intervention_type, [])
        
        if len(snapshots) < 3:
            return {'stability_score': 1.0, 'variance_score': 0.0, 'trend_score': 1.0}
        
        # Get recent effectiveness scores
        recent_scores = [s.effectiveness_score for s in snapshots[-10:]]
        
        # Coefficient of variation (stability measure)
        cv = np.std(recent_scores) / (np.mean(recent_scores) + 1e-8)
        stability_score = 1.0 / (1.0 + cv)
        
        # Variance score
        variance_score = np.var(recent_scores)
        
        # Trend score (consistency of direction)
        if len(recent_scores) >= 5:
            differences = np.diff(recent_scores)
            trend_consistency = 1.0 - (np.std(differences) / (np.mean(np.abs(differences)) + 1e-8))
            trend_score = max(0.0, trend_consistency)
        else:
            trend_score = 1.0
        
        return {
            'stability_score': stability_score,
            'variance_score': variance_score,
            'trend_score': trend_score
        }
    
    def _normalize_bias_score(self, bias_score: float) -> float:
        """Normalize bias score to 0-1 scale."""
        # Simple normalization assuming bias scores are between 0 and 1
        return max(0.0, min(1.0, bias_score))
    
    def _compute_accuracy_retention(self, current_accuracy: float, intervention_type: str) -> float:
        """Compute how well accuracy is retained compared to baseline."""
        # Get baseline accuracy if available
        baseline_accuracy = self._get_baseline_accuracy(intervention_type)
        
        if baseline_accuracy is None:
            return current_accuracy
        
        # Compute retention ratio
        retention = current_accuracy / (baseline_accuracy + 1e-8)
        return min(1.0, retention)  # Cap at 100% retention
    
    def _get_baseline_reference(self, intervention_type: str) -> Optional[float]:
        """Get baseline bias score for comparison."""
        # Look for baseline snapshots
        for snapshot in self.persistence_snapshots:
            if (snapshot.intervention_type == intervention_type and 
                snapshot.model_state == "baseline"):
                return snapshot.bias_scores.get('primary_bias_score')
        
        return None
    
    def _get_baseline_accuracy(self, intervention_type: str) -> Optional[float]:
        """Get baseline accuracy for comparison."""
        for snapshot in self.persistence_snapshots:
            if (snapshot.intervention_type == intervention_type and 
                snapshot.model_state == "baseline"):
                return snapshot.accuracy_scores.get('overall_accuracy')
        
        return None
    
    def _check_persistence_alerts(self, snapshot: PersistenceSnapshot):
        """Check for persistence-related alerts."""
        alerts = []
        
        # Effectiveness threshold alert
        if snapshot.effectiveness_score < self.effectiveness_threshold:
            alerts.append({
                'type': 'low_effectiveness',
                'message': f"Effectiveness below threshold: {snapshot.effectiveness_score:.3f} < {self.effectiveness_threshold}",
                'severity': 'high',
                'timestamp': snapshot.timestamp
            })
        
        # High decay rate alert
        if snapshot.decay_rate > self.decay_rate_alert:
            alerts.append({
                'type': 'rapid_decay',
                'message': f"Rapid decay detected: {snapshot.decay_rate:.4f}/hour",
                'severity': 'medium',
                'timestamp': snapshot.timestamp
            })
        
        # Stability alert
        stability_score = snapshot.stability_metrics.get('stability_score', 1.0)
        if stability_score < self.stability_threshold:
            alerts.append({
                'type': 'instability',
                'message': f"Stability below threshold: {stability_score:.3f} < {self.stability_threshold}",
                'severity': 'medium',
                'timestamp': snapshot.timestamp
            })
        
        # Store alerts
        for alert in alerts:
            self.threshold_alerts[snapshot.intervention_type].append(alert)
            self.logger.warning(f"PERSISTENCE ALERT: {alert['message']}")
    
    def _get_relevant_snapshots(self, intervention_type: str, 
                              analysis_period_hours: Optional[float]) -> List[PersistenceSnapshot]:
        """Get snapshots relevant for analysis."""
        snapshots = self.intervention_timelines.get(intervention_type, [])
        
        if analysis_period_hours is None:
            return snapshots
        
        cutoff_time = datetime.now() - timedelta(hours=analysis_period_hours)
        return [s for s in snapshots if s.timestamp >= cutoff_time]
    
    def _build_decay_model(self, snapshots: List[PersistenceSnapshot]) -> DecayModel:
        """Build predictive decay model from snapshots."""
        if len(snapshots) < 3:
            return self._create_default_decay_model()
        
        # Extract time and effectiveness data
        start_time = snapshots[0].timestamp
        times = [(s.timestamp - start_time).total_seconds() / 3600 for s in snapshots]
        effectiveness = [s.effectiveness_score for s in snapshots]
        
        times = np.array(times)
        effectiveness = np.array(effectiveness)
        
        # Try different decay models
        models = {}
        
        # 1. Exponential decay model
        try:
            def exponential_decay(t, a, b, c):
                return a * np.exp(-b * t) + c
            
            popt_exp, _ = curve_fit(exponential_decay, times, effectiveness, 
                                  bounds=([0, 0, 0], [2, 1, 1]))
            pred_exp = exponential_decay(times, *popt_exp)
            r2_exp = 1 - np.sum((effectiveness - pred_exp)**2) / np.sum((effectiveness - np.mean(effectiveness))**2)
            
            models['exponential'] = {
                'parameters': {'a': popt_exp[0], 'b': popt_exp[1], 'c': popt_exp[2]},
                'r_squared': r2_exp,
                'predictions': pred_exp,
                'equation': f"f(t) = {popt_exp[0]:.3f} * exp(-{popt_exp[1]:.3f} * t) + {popt_exp[2]:.3f}"
            }
        except:
            pass
        
        # 2. Linear decay model
        try:
            slope, intercept, r_value, _, _ = stats.linregress(times, effectiveness)
            pred_linear = slope * times + intercept
            
            models['linear'] = {
                'parameters': {'slope': slope, 'intercept': intercept},
                'r_squared': r_value**2,
                'predictions': pred_linear,
                'equation': f"f(t) = {slope:.4f} * t + {intercept:.3f}"
            }
        except:
            pass
        
        # 3. Polynomial decay model
        try:
            poly_features = PolynomialFeatures(degree=2)
            times_poly = poly_features.fit_transform(times.reshape(-1, 1))
            
            reg = LinearRegression()
            reg.fit(times_poly, effectiveness)
            pred_poly = reg.predict(times_poly)
            
            r2_poly = reg.score(times_poly, effectiveness)
            
            models['polynomial'] = {
                'parameters': {'coefficients': reg.coef_.tolist(), 'intercept': reg.intercept_},
                'r_squared': r2_poly,
                'predictions': pred_poly,
                'equation': f"f(t) = {reg.coef_[2]:.4f} * t² + {reg.coef_[1]:.4f} * t + {reg.intercept_:.3f}"
            }
        except:
            pass
        
        # Select best model
        if not models:
            return self._create_default_decay_model()
        
        best_model_name = max(models.keys(), key=lambda k: models[k]['r_squared'])
        best_model = models[best_model_name]
        
        # Compute half-life for exponential model
        half_life = None
        if best_model_name == 'exponential' and best_model['parameters']['b'] > 0:
            half_life = np.log(2) / best_model['parameters']['b']
        
        # Compute prediction accuracy
        prediction_accuracy = best_model['r_squared']
        
        # Compute confidence interval (simplified)
        residuals = effectiveness - best_model['predictions']
        confidence_interval = (
            np.mean(residuals) - 1.96 * np.std(residuals),
            np.mean(residuals) + 1.96 * np.std(residuals)
        )
        
        return DecayModel(
            model_type=best_model_name,
            parameters=best_model['parameters'],
            r_squared=best_model['r_squared'],
            prediction_accuracy=prediction_accuracy,
            half_life=half_life,
            confidence_interval=confidence_interval,
            model_equation=best_model['equation'],
            metadata={
                'n_snapshots': len(snapshots),
                'time_span_hours': times[-1] - times[0],
                'all_models_tested': list(models.keys())
            }
        )
    
    def _create_default_decay_model(self) -> DecayModel:
        """Create default decay model when insufficient data."""
        return DecayModel(
            model_type="default",
            parameters={'decay_rate': 0.01},
            r_squared=0.0,
            prediction_accuracy=0.0,
            half_life=None,
            confidence_interval=(0.0, 0.0),
            model_equation="f(t) = 1.0 - 0.01 * t",
            metadata={'insufficient_data': True}
        )
    
    def _analyze_persistence_phases(self, snapshots: List[PersistenceSnapshot]) -> Dict[str, Dict[str, float]]:
        """Analyze performance in different persistence phases."""
        phase_analysis = {}
        
        for phase in self.phase_definitions:
            phase_snapshots = [s for s in snapshots if s.persistence_phase == phase]
            
            if phase_snapshots:
                effectiveness_scores = [s.effectiveness_score for s in phase_snapshots]
                decay_rates = [s.decay_rate for s in phase_snapshots]
                
                phase_analysis[phase] = {
                    'count': len(phase_snapshots),
                    'mean_effectiveness': np.mean(effectiveness_scores),
                    'std_effectiveness': np.std(effectiveness_scores),
                    'mean_decay_rate': np.mean(decay_rates),
                    'stability_score': 1.0 - (np.std(effectiveness_scores) / (np.mean(effectiveness_scores) + 1e-8))
                }
            else:
                phase_analysis[phase] = {
                    'count': 0,
                    'mean_effectiveness': 0.0,
                    'std_effectiveness': 0.0,
                    'mean_decay_rate': 0.0,
                    'stability_score': 0.0
                }
        
        return phase_analysis
    
    def _identify_critical_thresholds(self, snapshots: List[PersistenceSnapshot], 
                                    decay_model: DecayModel) -> Dict[str, float]:
        """Identify critical thresholds for intervention maintenance."""
        effectiveness_scores = [s.effectiveness_score for s in snapshots]
        
        thresholds = {
            'minimum_effectiveness': self.effectiveness_threshold,
            'intervention_failure': 0.5,  # 50% of original effectiveness
            'maintenance_trigger': 0.8,   # 80% of original effectiveness
            'critical_decay_rate': self.decay_rate_alert
        }
        
        # Adaptive thresholds based on data
        if effectiveness_scores:
            max_effectiveness = max(effectiveness_scores)
            thresholds['optimal_effectiveness'] = max_effectiveness * 0.95
            thresholds['degraded_effectiveness'] = max_effectiveness * 0.7
        
        return thresholds
    
    def _generate_maintenance_recommendations(self, snapshots: List[PersistenceSnapshot],
                                            decay_model: DecayModel,
                                            thresholds: Dict[str, float]) -> List[str]:
        """Generate maintenance recommendations based on analysis."""
        recommendations = []
        
        current_effectiveness = snapshots[-1].effectiveness_score if snapshots else 0.0
        current_decay_rate = snapshots[-1].decay_rate if snapshots else 0.0
        
        # Effectiveness-based recommendations
        if current_effectiveness < thresholds['minimum_effectiveness']:
            recommendations.append("URGENT: Intervention effectiveness below minimum threshold - immediate reapplication required")
        elif current_effectiveness < thresholds['maintenance_trigger']:
            recommendations.append("WARNING: Intervention effectiveness declining - schedule maintenance intervention")
        
        # Decay rate recommendations
        if current_decay_rate > thresholds['critical_decay_rate']:
            recommendations.append("ALERT: High decay rate detected - consider strengthening intervention or increasing frequency")
        
        # Model-based recommendations
        if decay_model.r_squared > 0.7:
            if decay_model.half_life and decay_model.half_life < 24:
                recommendations.append(f"Short half-life detected ({decay_model.half_life:.1f} hours) - consider daily reapplication")
            elif decay_model.half_life and decay_model.half_life < 168:
                recommendations.append(f"Medium half-life detected ({decay_model.half_life:.1f} hours) - consider weekly reapplication")
        
        # Phase-based recommendations
        long_term_snapshots = [s for s in snapshots if s.persistence_phase == "long_term"]
        if long_term_snapshots:
            long_term_effectiveness = np.mean([s.effectiveness_score for s in long_term_snapshots])
            if long_term_effectiveness > 0.8:
                recommendations.append("✓ Good long-term persistence - current maintenance schedule adequate")
        
        if not recommendations:
            recommendations.append("✓ Intervention persistence is stable - continue current monitoring schedule")
        
        return recommendations
    
    def _create_reapplication_schedule(self, decay_model: DecayModel, 
                                     thresholds: Dict[str, float]) -> Dict[str, Any]:
        """Create optimal reapplication schedule."""
        schedule = {
            'strategy': 'adaptive',
            'base_interval_hours': 168,  # Weekly default
            'trigger_thresholds': thresholds,
            'recommended_intervals': {}
        }
        
        # Adjust based on decay model
        if decay_model.half_life:
            if decay_model.half_life < 12:
                schedule['strategy'] = 'frequent'
                schedule['base_interval_hours'] = 6
            elif decay_model.half_life < 48:
                schedule['strategy'] = 'daily'
                schedule['base_interval_hours'] = 24
            elif decay_model.half_life < 336:  # 2 weeks
                schedule['strategy'] = 'weekly'
                schedule['base_interval_hours'] = 168
            else:
                schedule['strategy'] = 'biweekly'
                schedule['base_interval_hours'] = 336
        
        # Phase-specific intervals
        schedule['recommended_intervals'] = {
            'immediate': 2,    # Monitor every 2 hours initially
            'short_term': 6,   # Every 6 hours
            'medium_term': 24, # Daily
            'long_term': schedule['base_interval_hours']
        }
        
        return schedule
    
    def _assess_risk_factors(self, snapshots: List[PersistenceSnapshot]) -> Dict[str, float]:
        """Assess risk factors for intervention failure."""
        risk_factors = {}
        
        if not snapshots:
            return risk_factors
        
        effectiveness_scores = [s.effectiveness_score for s in snapshots]
        decay_rates = [s.decay_rate for s in snapshots]
        
        # Trend risk
        if len(effectiveness_scores) >= 3:
            recent_trend = np.mean(effectiveness_scores[-3:]) - np.mean(effectiveness_scores[:3])
            risk_factors['declining_trend'] = max(0.0, -recent_trend)
        
        # Volatility risk
        effectiveness_cv = np.std(effectiveness_scores) / (np.mean(effectiveness_scores) + 1e-8)
        risk_factors['volatility'] = min(1.0, effectiveness_cv)
        
        # Decay acceleration risk
        if len(decay_rates) >= 3:
            recent_decay = np.mean(decay_rates[-3:])
            early_decay = np.mean(decay_rates[:3])
            risk_factors['decay_acceleration'] = max(0.0, recent_decay - early_decay)
        
        # Environmental risk factors
        environmental_changes = []
        for snapshot in snapshots:
            if snapshot.environmental_factors:
                environmental_changes.append(len(snapshot.environmental_factors))
        
        if environmental_changes:
            risk_factors['environmental_instability'] = np.std(environmental_changes) / (np.mean(environmental_changes) + 1e-8)
        
        return risk_factors
    
    def _compute_resilience_score(self, snapshots: List[PersistenceSnapshot], 
                                decay_model: DecayModel) -> float:
        """Compute overall resilience score."""
        if not snapshots:
            return 0.0
        
        factors = []
        
        # Stability factor
        effectiveness_scores = [s.effectiveness_score for s in snapshots]
        stability = 1.0 - (np.std(effectiveness_scores) / (np.mean(effectiveness_scores) + 1e-8))
        factors.append(max(0.0, stability))
        
        # Model quality factor
        model_quality = decay_model.r_squared if decay_model.r_squared > 0 else 0.0
        factors.append(model_quality)
        
        # Persistence factor
        current_effectiveness = snapshots[-1].effectiveness_score
        factors.append(current_effectiveness)
        
        # Recovery factor (how well it maintains effectiveness over time)
        if len(snapshots) > 5:
            recent_avg = np.mean([s.effectiveness_score for s in snapshots[-5:]])
            early_avg = np.mean([s.effectiveness_score for s in snapshots[:5]])
            recovery = min(1.0, recent_avg / (early_avg + 1e-8))
            factors.append(recovery)
        
        return np.mean(factors)
    
    def _assess_long_term_viability(self, decay_model: DecayModel, resilience_score: float) -> bool:
        """Assess if intervention is viable long-term."""
        criteria = []
        
        # Model quality
        criteria.append(decay_model.r_squared > 0.5)
        
        # Resilience
        criteria.append(resilience_score > 0.6)
        
        # Half-life (if available)
        if decay_model.half_life:
            criteria.append(decay_model.half_life > 24)  # At least 24 hours
        
        # Model type preference
        if decay_model.model_type in ['exponential', 'polynomial']:
            criteria.append(True)
        else:
            criteria.append(False)
        
        # Require majority of criteria to be met
        return sum(criteria) >= len(criteria) // 2
    
    def save_persistence_data(self, output_path: str):
        """Save all persistence tracking data."""
        output_data = {
            'persistence_snapshots': [],
            'intervention_timelines': {},
            'decay_models': {},
            'threshold_alerts': {},
            'configuration': {
                'phase_definitions': self.phase_definitions,
                'effectiveness_threshold': self.effectiveness_threshold,
                'decay_rate_alert': self.decay_rate_alert,
                'stability_threshold': self.stability_threshold,
                'min_snapshots_for_modeling': self.min_snapshots_for_modeling
            }
        }
        
        # Convert snapshots
        for snapshot in self.persistence_snapshots:
            snapshot_dict = {
                'timestamp': snapshot.timestamp.isoformat(),
                'intervention_type': snapshot.intervention_type,
                'model_state': snapshot.model_state,
                'persistence_phase': snapshot.persistence_phase,
                'effectiveness_score': snapshot.effectiveness_score,
                'decay_rate': snapshot.decay_rate,
                'stability_metrics': snapshot.stability_metrics,
                'environmental_factors': snapshot.environmental_factors,
                'intervention_strength': snapshot.intervention_strength,
                'bias_scores': snapshot.bias_scores,
                'accuracy_scores': snapshot.accuracy_scores,
                'sample_count': snapshot.sample_count,
                'metadata': snapshot.metadata
            }
            output_data['persistence_snapshots'].append(snapshot_dict)
        
        # Convert timelines
        for intervention_type, snapshots in self.intervention_timelines.items():
            output_data['intervention_timelines'][intervention_type] = [
                s.timestamp.isoformat() for s in snapshots
            ]
        
        # Convert decay models
        for intervention_type, model in self.decay_models.items():
            output_data['decay_models'][intervention_type] = {
                'model_type': model.model_type,
                'parameters': model.parameters,
                'r_squared': model.r_squared,
                'prediction_accuracy': model.prediction_accuracy,
                'half_life': model.half_life,
                'confidence_interval': list(model.confidence_interval),
                'model_equation': model.model_equation,
                'metadata': model.metadata
            }
        
        # Convert alerts
        for intervention_type, alerts in self.threshold_alerts.items():
            output_data['threshold_alerts'][intervention_type] = [
                {
                    'type': alert['type'],
                    'message': alert['message'],
                    'severity': alert['severity'],
                    'timestamp': alert['timestamp'].isoformat()
                }
                for alert in alerts
            ]
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        self.logger.info(f"Persistence tracking data saved to {output_path}")


def main():
    """Demo usage of InterventionPersistenceTracker."""
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="Intervention persistence tracking")
    parser.add_argument("--model", default="gpt2", help="Model name")
    parser.add_argument("--dataset", default="winogender", help="Dataset to track")
    parser.add_argument("--intervention", default="lora_training", help="Intervention type")
    parser.add_argument("--snapshots", type=int, default=10, help="Number of snapshots to take")
    parser.add_argument("--interval", type=int, default=300, help="Interval between snapshots (seconds)")
    parser.add_argument("--output", default="persistence_tracking.json", help="Output file")
    
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
    tracker = InterventionPersistenceTracker(base_evaluator)
    
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
    tracker.track_persistence_snapshot(
        dataset_path=dataset_path,
        dataset_name=args.dataset,
        evaluation_function=eval_func,
        intervention_type=args.intervention,
        model_state="baseline",
        num_samples=5
    )
    
    # Simulate persistence tracking over time
    print(f"Tracking persistence with {args.snapshots} snapshots...")
    for i in range(args.snapshots):
        time.sleep(args.interval)
        
        # Simulate intervention decay
        intervention_strength = 1.0 - (i * 0.05)  # Gradual decay
        
        tracker.track_persistence_snapshot(
            dataset_path=dataset_path,
            dataset_name=args.dataset,
            evaluation_function=eval_func,
            intervention_type=args.intervention,
            model_state="post_intervention",
            intervention_strength=intervention_strength,
            environmental_factors={'time_elapsed': i * args.interval},
            num_samples=5
        )
        
        print(f"Snapshot {i + 1}/{args.snapshots} completed")
    
    # Perform comprehensive analysis
    print("Performing long-term persistence analysis...")
    analysis = tracker.analyze_long_term_persistence(args.intervention)
    
    # Save results
    tracker.save_persistence_data(args.output)
    
    # Print summary
    print(f"\n=== Persistence Analysis Results ===")
    print(f"Intervention: {analysis.intervention_type}")
    print(f"Analysis period: {analysis.analysis_period[0]} to {analysis.analysis_period[1]}")
    print(f"Total snapshots: {analysis.total_snapshots}")
    print(f"Decay model: {analysis.decay_model.model_type} (R²={analysis.decay_model.r_squared:.3f})")
    if analysis.decay_model.half_life:
        print(f"Half-life: {analysis.decay_model.half_life:.1f} hours")
    print(f"Resilience score: {analysis.resilience_score:.3f}")
    print(f"Long-term viable: {analysis.long_term_viability}")
    
    print(f"\nMaintenance Recommendations:")
    for rec in analysis.maintenance_recommendations:
        print(f"  - {rec}")


if __name__ == "__main__":
    main()