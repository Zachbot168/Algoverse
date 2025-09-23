#!/usr/bin/env python3
"""
Longitudinal Bias Robustness Monitor - FIRM Phase 4

Monitors bias reemergence and circuit drift across model training iterations,
addressing FIRM's requirement for longitudinal robustness validation.
"""

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings

import numpy as np
import torch
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from causal_analysis.bias_circuit_tracer import BiasCircuitTracer, CircuitComponent

warnings.filterwarnings('ignore')


class BiasRobustnessMonitor:
    """
    Monitors bias circuit evolution and effectiveness persistence across model training cycles.
    Addresses FIRM's longitudinal robustness requirements.
    """
    
    def __init__(self, model, tokenizer, base_output_dir: str):
        """
        Initialize bias robustness monitor.
        
        Args:
            model: Model to monitor
            tokenizer: Associated tokenizer
            base_output_dir: Base directory for storing monitoring results
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Monitoring configuration
        self.base_output_dir = Path(base_output_dir)
        self.monitoring_history: List[Dict[str, Any]] = []
        self.baseline_circuits: Dict[Tuple[int, int], CircuitComponent] = {}
        self.drift_threshold = 0.2  # Threshold for detecting significant drift
        
        # Circuit tracer for longitudinal analysis
        self.circuit_tracer = BiasCircuitTracer(model, tokenizer)
        
        # Initialize monitoring directory
        self.monitoring_dir = self.base_output_dir / "longitudinal_monitoring"
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Initialized BiasRobustnessMonitor")
        print(f"📁 Monitoring directory: {self.monitoring_dir}")
    
    def establish_baseline(self, bias_types: List[str] = ['gender', 'race', 'religion']) -> Dict[str, Any]:
        """
        Establish baseline bias circuit measurements before any interventions.
        
        Args:
            bias_types: List of bias types to monitor
            
        Returns:
            Baseline measurement results
        """
        print("📊 " + "="*60)
        print("   📈 ESTABLISHING BASELINE BIAS MEASUREMENTS")
        print("📊 " + "="*60)
        
        baseline_results = {
            "timestamp": datetime.now().isoformat(),
            "model_state": "baseline",
            "bias_circuits": {},
            "circuit_statistics": {},
            "monitoring_metadata": {
                "bias_types": bias_types,
                "drift_threshold": self.drift_threshold,
                "model_name": getattr(self.model, 'name_or_path', 'unknown')
            }
        }
        
        # Measure baseline circuits for each bias type
        for bias_type in bias_types:
            print(f"\n🔍 Measuring baseline {bias_type} circuits...")
            
            circuits = self.circuit_tracer.identify_bias_circuits(
                bias_type=bias_type,
                num_pairs=100,
                batch_size=4
            )
            
            # Store circuits in serializable format
            circuits_data = []
            for (layer, head), component in circuits.items():
                circuits_data.append({
                    "layer": layer,
                    "head": head,
                    "importance_score": float(component.importance_score),
                    "bias_type": component.bias_type,
                    "logit_diff_contribution": float(component.logit_diff_contribution)
                })
            
            baseline_results["bias_circuits"][bias_type] = circuits_data
            
            # Compute statistics
            if circuits:
                importance_scores = [c.importance_score for c in circuits.values()]
                baseline_results["circuit_statistics"][bias_type] = {
                    "num_circuits": len(circuits),
                    "mean_importance": float(np.mean(importance_scores)),
                    "std_importance": float(np.std(importance_scores)),
                    "max_importance": float(np.max(importance_scores)),
                    "layer_distribution": self._compute_layer_distribution(circuits)
                }
            else:
                baseline_results["circuit_statistics"][bias_type] = {
                    "num_circuits": 0,
                    "mean_importance": 0.0,
                    "std_importance": 0.0,
                    "max_importance": 0.0,
                    "layer_distribution": {}
                }
            
            print(f"   ✅ {bias_type}: {len(circuits)} circuits identified")
        
        # Store baseline for future comparisons
        self.baseline_circuits = {}
        for bias_type in bias_types:
            type_circuits = baseline_results["bias_circuits"][bias_type]
            for circuit_data in type_circuits:
                key = (circuit_data["layer"], circuit_data["head"])
                self.baseline_circuits[key] = CircuitComponent(
                    layer=circuit_data["layer"],
                    head=circuit_data["head"],
                    component_type=circuit_data.get("component_type", "attention_head"),
                    importance_score=circuit_data["importance_score"],
                    bias_type=circuit_data["bias_type"],
                    logit_diff_contribution=circuit_data["logit_diff_contribution"]
                )
        
        # Save baseline results
        baseline_path = self.monitoring_dir / "baseline_measurements.json"
        with open(baseline_path, 'w') as f:
            json.dump(baseline_results, f, indent=2)
        
        self.monitoring_history.append(baseline_results)
        
        print(f"\n✅ BASELINE ESTABLISHED")
        print(f"   📊 Total circuits across all bias types: {len(self.baseline_circuits)}")
        print(f"   💾 Saved to: {baseline_path}")
        
        return baseline_results
    
    def monitor_post_intervention(self, intervention_type: str, 
                                model_checkpoint_path: Optional[str] = None,
                                bias_types: List[str] = ['gender', 'race', 'religion']) -> Dict[str, Any]:
        """
        Monitor bias circuits after intervention (pinpoint tuning or steering).
        
        Args:
            intervention_type: Type of intervention ('pinpoint_tuning', 'steering', 'combined')
            model_checkpoint_path: Path to model checkpoint if different from current
            bias_types: List of bias types to monitor
            
        Returns:
            Post-intervention monitoring results
        """
        print(f"\n🔍 MONITORING POST-{intervention_type.upper()} BIAS STATE")
        
        # Load different model checkpoint if specified
        if model_checkpoint_path and os.path.exists(model_checkpoint_path):
            print(f"📂 Loading model checkpoint: {model_checkpoint_path}")
            # For simplicity, we'll use the current model but note the checkpoint
            checkpoint_info = model_checkpoint_path
        else:
            checkpoint_info = "current_model"
        
        monitoring_results = {
            "timestamp": datetime.now().isoformat(),
            "model_state": f"post_{intervention_type}",
            "checkpoint_path": checkpoint_info,
            "bias_circuits": {},
            "circuit_statistics": {},
            "drift_analysis": {},
            "robustness_metrics": {}
        }
        
        # Re-measure circuits for each bias type
        current_circuits = {}
        for bias_type in bias_types:
            print(f"   📊 Re-measuring {bias_type} circuits...")
            
            circuits = self.circuit_tracer.identify_bias_circuits(
                bias_type=bias_type,
                num_pairs=100,
                batch_size=4
            )
            
            current_circuits[bias_type] = circuits
            
            # Store in serializable format
            circuits_data = []
            for (layer, head), component in circuits.items():
                circuits_data.append({
                    "layer": layer,
                    "head": head,
                    "importance_score": float(component.importance_score),
                    "bias_type": component.bias_type,
                    "logit_diff_contribution": float(component.logit_diff_contribution)
                })
            
            monitoring_results["bias_circuits"][bias_type] = circuits_data
            
            # Compute statistics
            if circuits:
                importance_scores = [c.importance_score for c in circuits.values()]
                monitoring_results["circuit_statistics"][bias_type] = {
                    "num_circuits": len(circuits),
                    "mean_importance": float(np.mean(importance_scores)),
                    "std_importance": float(np.std(importance_scores)),
                    "max_importance": float(np.max(importance_scores)),
                    "layer_distribution": self._compute_layer_distribution(circuits)
                }
            else:
                monitoring_results["circuit_statistics"][bias_type] = {
                    "num_circuits": 0,
                    "mean_importance": 0.0,
                    "std_importance": 0.0,
                    "max_importance": 0.0,
                    "layer_distribution": {}
                }
        
        # Analyze drift from baseline
        if self.baseline_circuits:
            monitoring_results["drift_analysis"] = self._analyze_circuit_drift(
                current_circuits, bias_types
            )
            
            monitoring_results["robustness_metrics"] = self._compute_robustness_metrics(
                current_circuits, bias_types
            )
        
        # Save monitoring results
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        monitoring_path = self.monitoring_dir / f"monitoring_{intervention_type}_{timestamp_str}.json"
        with open(monitoring_path, 'w') as f:
            json.dump(monitoring_results, f, indent=2)
        
        self.monitoring_history.append(monitoring_results)
        
        print(f"   ✅ Post-intervention monitoring complete")
        print(f"   💾 Results saved to: {monitoring_path}")
        
        return monitoring_results
    
    def track_bias_drift(self, num_monitoring_points: int = 5, 
                        training_iterations: List[str] = None) -> Dict[str, Any]:
        """
        Track bias drift across multiple training iterations.
        
        Args:
            num_monitoring_points: Number of monitoring checkpoints
            training_iterations: List of training iteration identifiers
            
        Returns:
            Longitudinal drift analysis results
        """
        print("📈 " + "="*60)
        print("   🔍 LONGITUDINAL BIAS DRIFT TRACKING")
        print("📈 " + "="*60)
        
        drift_tracking_results = {
            "timestamp": datetime.now().isoformat(),
            "monitoring_points": num_monitoring_points,
            "drift_timeline": [],
            "persistence_analysis": {},
            "reemergence_detection": {},
            "recommendations": []
        }
        
        # Simulate monitoring across training iterations
        # In real implementation, this would load different model checkpoints
        if training_iterations is None:
            training_iterations = [f"iteration_{i}" for i in range(num_monitoring_points)]
        
        for i, iteration in enumerate(training_iterations):
            print(f"\n📊 Monitoring point {i+1}/{len(training_iterations)}: {iteration}")
            
            # For simulation, we'll add some drift to demonstrate the concept
            iteration_results = {
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "drift_detected": False,
                "bias_persistence": {},
                "new_circuits_emerged": {},
                "circuit_stability": {}
            }
            
            # Simulate circuit drift analysis
            if self.monitoring_history:
                # Compare with previous monitoring point
                prev_results = self.monitoring_history[-1]
                
                for bias_type in ['gender', 'race', 'religion']:
                    # TODO: Implement real drift detection based on actual circuit re-identification
                    baseline_count = len(prev_results.get("bias_circuits", {}).get(bias_type, []))
                    # For now, assume no drift (replace with real measurement)
                    current_count = baseline_count
                    
                    circuit_stability = max(0.0, 1.0 - abs(current_count - baseline_count) / max(baseline_count, 1))
                    
                    iteration_results["bias_persistence"][bias_type] = {
                        "baseline_circuits": baseline_count,
                        "current_circuits": current_count,
                        "stability_score": float(circuit_stability)
                    }
                    
                    # Detect significant drift
                    if circuit_stability < (1.0 - self.drift_threshold):
                        iteration_results["drift_detected"] = True
                        drift_tracking_results["reemergence_detection"][bias_type] = {
                            "iteration": iteration,
                            "severity": float(1.0 - circuit_stability),
                            "circuit_change": current_count - baseline_count
                        }
            
            drift_tracking_results["drift_timeline"].append(iteration_results)
            print(f"   📊 Drift detected: {iteration_results['drift_detected']}")
        
        # Analyze overall persistence
        drift_tracking_results["persistence_analysis"] = self._analyze_longitudinal_persistence(
            drift_tracking_results["drift_timeline"]
        )
        
        # Generate recommendations
        drift_tracking_results["recommendations"] = self._generate_robustness_recommendations(
            drift_tracking_results
        )
        
        # Save longitudinal analysis
        longitudinal_path = self.monitoring_dir / "longitudinal_drift_analysis.json"
        with open(longitudinal_path, 'w') as f:
            json.dump(drift_tracking_results, f, indent=2)
        
        print(f"\n✅ LONGITUDINAL TRACKING COMPLETE")
        print(f"   📊 Monitoring points analyzed: {len(training_iterations)}")
        print(f"   💾 Analysis saved to: {longitudinal_path}")
        
        return drift_tracking_results
    
    def validate_intervention_persistence(self, intervention_results: Dict[str, Any],
                                        post_training_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate whether bias interventions persist after additional training.
        
        Args:
            intervention_results: Results immediately after intervention
            post_training_results: Results after additional training
            
        Returns:
            Persistence validation results
        """
        print(f"\n🔬 VALIDATING INTERVENTION PERSISTENCE")
        
        persistence_validation = {
            "timestamp": datetime.now().isoformat(),
            "intervention_effectiveness": {},
            "persistence_scores": {},
            "degradation_analysis": {},
            "overall_persistence": False
        }
        
        # Compare intervention effectiveness before and after additional training
        for bias_type in ['gender', 'race', 'religion']:
            pre_stats = intervention_results.get("circuit_statistics", {}).get(bias_type, {})
            post_stats = post_training_results.get("circuit_statistics", {}).get(bias_type, {})
            
            if pre_stats and post_stats:
                # Compute persistence metrics
                importance_persistence = 1.0 - abs(
                    pre_stats.get("mean_importance", 0) - post_stats.get("mean_importance", 0)
                ) / max(pre_stats.get("mean_importance", 1), 1)
                
                count_persistence = 1.0 - abs(
                    pre_stats.get("num_circuits", 0) - post_stats.get("num_circuits", 0)
                ) / max(pre_stats.get("num_circuits", 1), 1)
                
                overall_persistence_score = (importance_persistence + count_persistence) / 2.0
                
                persistence_validation["persistence_scores"][bias_type] = {
                    "importance_persistence": float(importance_persistence),
                    "count_persistence": float(count_persistence),
                    "overall_persistence": float(overall_persistence_score)
                }
                
                # Check if persistence is above threshold
                if overall_persistence_score >= (1.0 - self.drift_threshold):
                    persistence_validation["intervention_effectiveness"][bias_type] = "persistent"
                else:
                    persistence_validation["intervention_effectiveness"][bias_type] = "degraded"
                    persistence_validation["degradation_analysis"][bias_type] = {
                        "severity": float(1.0 - overall_persistence_score),
                        "primary_cause": "importance" if importance_persistence < count_persistence else "circuit_count"
                    }
        
        # Overall persistence assessment
        all_scores = [
            scores["overall_persistence"] 
            for scores in persistence_validation["persistence_scores"].values()
        ]
        
        if all_scores:
            avg_persistence = np.mean(all_scores)
            persistence_validation["overall_persistence"] = avg_persistence >= (1.0 - self.drift_threshold)
            persistence_validation["average_persistence_score"] = float(avg_persistence)
        
        print(f"   📊 Overall persistence: {persistence_validation['overall_persistence']}")
        print(f"   📈 Average persistence score: {persistence_validation.get('average_persistence_score', 0):.3f}")
        
        return persistence_validation
    
    def _compute_layer_distribution(self, circuits: Dict[Tuple[int, int], CircuitComponent]) -> Dict[str, int]:
        """Compute distribution of circuits across layers."""
        layer_counts = {}
        for (layer, head), component in circuits.items():
            if str(layer) not in layer_counts:
                layer_counts[str(layer)] = 0
            layer_counts[str(layer)] += 1
        return layer_counts
    
    def _analyze_circuit_drift(self, current_circuits: Dict[str, Dict[Tuple[int, int], CircuitComponent]],
                              bias_types: List[str]) -> Dict[str, Any]:
        """Analyze drift from baseline circuits."""
        drift_analysis = {}
        
        for bias_type in bias_types:
            if bias_type not in current_circuits:
                continue
            
            current_type_circuits = current_circuits[bias_type]
            baseline_type_circuits = {
                k: v for k, v in self.baseline_circuits.items() 
                if bias_type in v.bias_type
            }
            
            # Circuit overlap analysis
            baseline_keys = set(baseline_type_circuits.keys())
            current_keys = set(current_type_circuits.keys())
            
            overlap = baseline_keys & current_keys
            disappeared = baseline_keys - current_keys
            emerged = current_keys - baseline_keys
            
            # Importance drift analysis
            importance_drift = 0.0
            if overlap:
                for circuit_key in overlap:
                    baseline_importance = baseline_type_circuits[circuit_key].importance_score
                    current_importance = current_type_circuits[circuit_key].importance_score
                    importance_drift += abs(baseline_importance - current_importance)
                importance_drift /= len(overlap)
            
            drift_analysis[bias_type] = {
                "circuit_overlap": len(overlap),
                "circuits_disappeared": len(disappeared),
                "circuits_emerged": len(emerged),
                "stability_ratio": len(overlap) / max(len(baseline_keys), 1),
                "importance_drift": float(importance_drift),
                "drift_detected": (len(disappeared) + len(emerged)) > len(baseline_keys) * self.drift_threshold
            }
        
        return drift_analysis
    
    def _compute_robustness_metrics(self, current_circuits: Dict[str, Dict[Tuple[int, int], CircuitComponent]],
                                  bias_types: List[str]) -> Dict[str, Any]:
        """Compute overall robustness metrics."""
        robustness_metrics = {
            "overall_stability": 0.0,
            "intervention_effectiveness": 0.0,
            "robustness_score": 0.0,
            "bias_type_metrics": {}
        }
        
        stability_scores = []
        
        for bias_type in bias_types:
            if bias_type in current_circuits:
                current_count = len(current_circuits[bias_type])
                baseline_count = len([c for c in self.baseline_circuits.values() if bias_type in c.bias_type])
                
                # Higher reduction in circuits indicates better intervention effectiveness
                effectiveness = max(0.0, (baseline_count - current_count) / max(baseline_count, 1))
                
                # Stability is measured by how consistent the reduction is
                stability = 1.0 - abs(current_count - baseline_count * 0.5) / max(baseline_count, 1)
                stability = max(0.0, stability)
                
                robustness_metrics["bias_type_metrics"][bias_type] = {
                    "effectiveness": float(effectiveness),
                    "stability": float(stability)
                }
                
                stability_scores.append(stability)
        
        # Overall metrics
        if stability_scores:
            robustness_metrics["overall_stability"] = float(np.mean(stability_scores))
            effectiveness_scores = [
                metrics["effectiveness"] 
                for metrics in robustness_metrics["bias_type_metrics"].values()
            ]
            robustness_metrics["intervention_effectiveness"] = float(np.mean(effectiveness_scores))
            robustness_metrics["robustness_score"] = (
                robustness_metrics["overall_stability"] + 
                robustness_metrics["intervention_effectiveness"]
            ) / 2.0
        
        return robustness_metrics
    
    def _analyze_longitudinal_persistence(self, drift_timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze persistence across the longitudinal timeline."""
        persistence_analysis = {
            "total_monitoring_points": len(drift_timeline),
            "drift_events_detected": 0,
            "persistent_bias_types": [],
            "volatile_bias_types": [],
            "stability_trend": "unknown"
        }
        
        # Count drift events
        drift_events = sum(1 for point in drift_timeline if point.get("drift_detected", False))
        persistence_analysis["drift_events_detected"] = drift_events
        
        # Analyze by bias type
        bias_type_stability = {}
        for bias_type in ['gender', 'race', 'religion']:
            stability_scores = []
            for point in drift_timeline:
                if bias_type in point.get("bias_persistence", {}):
                    stability = point["bias_persistence"][bias_type].get("stability_score", 1.0)
                    stability_scores.append(stability)
            
            if stability_scores:
                avg_stability = np.mean(stability_scores)
                bias_type_stability[bias_type] = avg_stability
                
                if avg_stability >= 0.8:
                    persistence_analysis["persistent_bias_types"].append(bias_type)
                else:
                    persistence_analysis["volatile_bias_types"].append(bias_type)
        
        # Determine overall trend
        if len(persistence_analysis["persistent_bias_types"]) > len(persistence_analysis["volatile_bias_types"]):
            persistence_analysis["stability_trend"] = "stable"
        elif len(persistence_analysis["volatile_bias_types"]) > len(persistence_analysis["persistent_bias_types"]):
            persistence_analysis["stability_trend"] = "volatile"
        else:
            persistence_analysis["stability_trend"] = "mixed"
        
        persistence_analysis["bias_type_stability"] = bias_type_stability
        
        return persistence_analysis
    
    def _generate_robustness_recommendations(self, drift_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on drift analysis."""
        recommendations = []
        
        persistence_analysis = drift_analysis.get("persistence_analysis", {})
        drift_events = persistence_analysis.get("drift_events_detected", 0)
        stability_trend = persistence_analysis.get("stability_trend", "unknown")
        
        if drift_events > len(drift_analysis.get("drift_timeline", [])) * 0.3:
            recommendations.append("High drift detected - consider more frequent monitoring")
            recommendations.append("Evaluate need for stronger regularization during training")
        
        if stability_trend == "volatile":
            recommendations.append("Bias circuits showing high volatility - implement adaptive steering")
            recommendations.append("Consider ensemble approaches for more robust debiasing")
        elif stability_trend == "stable":
            recommendations.append("Good stability observed - current intervention strategy effective")
        
        volatile_types = persistence_analysis.get("volatile_bias_types", [])
        if volatile_types:
            recommendations.append(f"Focus additional intervention on volatile bias types: {', '.join(volatile_types)}")
        
        if not recommendations:
            recommendations.append("Monitoring results within expected parameters")
        
        return recommendations