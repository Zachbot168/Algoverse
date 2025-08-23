#!/usr/bin/env python3
"""
Bias Comparison Analyzer
Analyzes before/after bias mitigation results to quantify improvements.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime


class BiasComparisonAnalyzer:
    """Analyzes bias evaluation results before and after mitigation to quantify improvements."""
    
    def __init__(self):
        self.pre_mitigation_results: Optional[Dict[str, Any]] = None
        self.post_mitigation_results: Optional[Dict[str, Any]] = None
        self.comparison_metrics: Dict[str, Any] = {}
    
    def load_results(self, pre_mitigation_path: str, post_mitigation_path: str) -> bool:
        """Load before and after mitigation results from JSON files."""
        try:
            # Load pre-mitigation results
            if os.path.exists(pre_mitigation_path):
                with open(pre_mitigation_path, 'r') as f:
                    self.pre_mitigation_results = json.load(f)
                print(f"✅ Loaded pre-mitigation results from {pre_mitigation_path}")
            else:
                print(f"❌ Pre-mitigation results not found: {pre_mitigation_path}")
                return False
            
            # Load post-mitigation results  
            if os.path.exists(post_mitigation_path):
                with open(post_mitigation_path, 'r') as f:
                    self.post_mitigation_results = json.load(f)
                print(f"✅ Loaded post-mitigation results from {post_mitigation_path}")
            else:
                print(f"❌ Post-mitigation results not found: {post_mitigation_path}")
                return False
                
            return True
            
        except Exception as e:
            print(f"❌ Error loading results: {e}")
            return False
    
    def extract_bias_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Extract bias metrics from evaluation results."""
        metrics = {}
        
        # Look for dataset-specific results
        for key, value in results.items():
            if isinstance(value, dict):
                # Check if this is a dataset result
                if 'bias_score' in value:
                    metrics[f"{key}_bias_score"] = float(value['bias_score'])
                elif 'main_metric' in value:
                    metrics[f"{key}_main_metric"] = float(value['main_metric'])
                elif 'stereotype_score' in value:
                    metrics[f"{key}_stereotype_score"] = float(value['stereotype_score'])
                elif 'accuracy' in value:
                    metrics[f"{key}_accuracy"] = float(value['accuracy'])
                
                # Look for bias type specific scores
                if 'bias_types' in value and isinstance(value['bias_types'], dict):
                    for bias_type, score in value['bias_types'].items():
                        if isinstance(score, (int, float)):
                            metrics[f"{key}_{bias_type}_bias"] = float(score)
        
        return metrics
    
    def calculate_improvements(self) -> Dict[str, Any]:
        """Calculate bias reduction improvements."""
        if not self.pre_mitigation_results or not self.post_mitigation_results:
            raise ValueError("Both pre and post mitigation results must be loaded")
        
        # Extract metrics from both results
        pre_metrics = self.extract_bias_metrics(self.pre_mitigation_results)
        post_metrics = self.extract_bias_metrics(self.post_mitigation_results)
        
        improvements = {
            "timestamp": datetime.now().isoformat(),
            "datasets_compared": [],
            "overall_improvement": 0.0,
            "dataset_improvements": {},
            "bias_type_improvements": {},
            "significant_reductions": [],
            "areas_for_improvement": [],
            "summary": {}
        }
        
        # Calculate dataset-level improvements
        dataset_names = set()
        for metric_name in pre_metrics.keys():
            dataset_name = metric_name.split('_')[0]
            dataset_names.add(dataset_name)
        
        total_improvement = 0.0
        valid_comparisons = 0
        
        for dataset in dataset_names:
            dataset_improvements = {}
            
            # Find matching metrics for this dataset
            for pre_metric, pre_value in pre_metrics.items():
                if pre_metric.startswith(dataset + '_'):
                    post_metric = pre_metric
                    if post_metric in post_metrics:
                        post_value = post_metrics[post_metric]
                        
                        # Calculate improvement (lower bias scores are better)
                        if pre_value > 0:  # Avoid division by zero
                            improvement = ((pre_value - post_value) / pre_value) * 100
                            dataset_improvements[pre_metric.replace(f"{dataset}_", "")] = {
                                "before": pre_value,
                                "after": post_value,
                                "improvement_percent": improvement,
                                "absolute_change": pre_value - post_value
                            }
                            
                            total_improvement += improvement
                            valid_comparisons += 1
                            
                            # Track significant improvements
                            if improvement > 10:  # >10% improvement
                                improvements["significant_reductions"].append({
                                    "dataset": dataset,
                                    "metric": pre_metric.replace(f"{dataset}_", ""),
                                    "improvement_percent": improvement,
                                    "before": pre_value,
                                    "after": post_value
                                })
                            elif improvement < -5:  # Degradation >5%
                                improvements["areas_for_improvement"].append({
                                    "dataset": dataset,
                                    "metric": pre_metric.replace(f"{dataset}_", ""),
                                    "degradation_percent": abs(improvement),
                                    "before": pre_value,
                                    "after": post_value
                                })
            
            if dataset_improvements:
                improvements["dataset_improvements"][dataset] = dataset_improvements
                improvements["datasets_compared"].append(dataset)
        
        # Calculate overall improvement
        if valid_comparisons > 0:
            improvements["overall_improvement"] = total_improvement / valid_comparisons
        
        # Generate summary
        improvements["summary"] = {
            "total_datasets_analyzed": len(improvements["datasets_compared"]),
            "metrics_compared": valid_comparisons,
            "average_improvement": improvements["overall_improvement"],
            "significant_improvements": len(improvements["significant_reductions"]),
            "areas_needing_attention": len(improvements["areas_for_improvement"]),
            "overall_assessment": self._get_overall_assessment(improvements["overall_improvement"])
        }
        
        self.comparison_metrics = improvements
        return improvements
    
    def _get_overall_assessment(self, improvement: float) -> str:
        """Get overall assessment based on improvement percentage."""
        if improvement >= 20:
            return "Excellent bias reduction achieved"
        elif improvement >= 10:
            return "Good bias reduction achieved"
        elif improvement >= 5:
            return "Moderate bias reduction achieved"
        elif improvement >= 0:
            return "Minimal bias reduction achieved"
        else:
            return "Bias increased - mitigation ineffective"
    
    def generate_comparison_report(self, output_dir: str) -> str:
        """Generate comprehensive comparison report."""
        if not self.comparison_metrics:
            raise ValueError("Must calculate improvements first")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate markdown report
        report_content = self._generate_markdown_report()
        report_path = os.path.join(output_dir, "bias_reduction_analysis.md")
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        # Save detailed JSON results
        json_path = os.path.join(output_dir, "bias_comparison_metrics.json")
        with open(json_path, 'w') as f:
            json.dump(self.comparison_metrics, f, indent=2)
        
        print(f"📊 Bias reduction report generated: {report_path}")
        print(f"📈 Detailed metrics saved: {json_path}")
        
        return report_path
    
    def _generate_markdown_report(self) -> str:
        """Generate markdown comparison report."""
        metrics = self.comparison_metrics
        summary = metrics["summary"]
        
        report = f"""# Bias Mitigation Effectiveness Analysis

**Analysis Date:** {metrics['timestamp']}  
**Overall Improvement:** {metrics['overall_improvement']:.2f}%  
**Assessment:** {summary['overall_assessment']}

## Executive Summary

- **Datasets Analyzed:** {summary['total_datasets_analyzed']}
- **Metrics Compared:** {summary['metrics_compared']}
- **Significant Improvements:** {summary['significant_improvements']}
- **Areas Needing Attention:** {summary['areas_needing_attention']}

## Dataset-Level Analysis

"""
        
        for dataset, improvements in metrics["dataset_improvements"].items():
            report += f"### {dataset}\n\n"
            
            for metric, data in improvements.items():
                improvement_icon = "✅" if data["improvement_percent"] > 0 else "❌"
                report += f"- {improvement_icon} **{metric}:** {data['before']:.3f} → {data['after']:.3f} "
                report += f"({data['improvement_percent']:+.1f}%)\n"
            
            report += "\n"
        
        # Significant reductions
        if metrics["significant_reductions"]:
            report += "## 🎯 Significant Bias Reductions (>10%)\n\n"
            for reduction in metrics["significant_reductions"]:
                report += f"- **{reduction['dataset']} ({reduction['metric']}):** "
                report += f"{reduction['before']:.3f} → {reduction['after']:.3f} "
                report += f"({reduction['improvement_percent']:+.1f}%)\n"
            report += "\n"
        
        # Areas for improvement
        if metrics["areas_for_improvement"]:
            report += "## ⚠️ Areas for Improvement (degraded >5%)\n\n"
            for issue in metrics["areas_for_improvement"]:
                report += f"- **{issue['dataset']} ({issue['metric']}):** "
                report += f"{issue['before']:.3f} → {issue['after']:.3f} "
                report += f"({issue['degradation_percent']:+.1f}% degradation)\n"
            report += "\n"
        
        # Recommendations
        report += "## 📋 Recommendations\n\n"
        
        if metrics["overall_improvement"] > 10:
            report += "✅ **Fairsteer mitigation is effective** - consider deploying for production use.\n\n"
        elif metrics["overall_improvement"] > 0:
            report += "⚠️ **Modest improvements achieved** - consider fine-tuning steering vectors or additional mitigation techniques.\n\n"
        else:
            report += "❌ **Mitigation ineffective** - review steering vector computation and consider alternative approaches.\n\n"
        
        if len(metrics["areas_for_improvement"]) > 0:
            report += f"🔧 **Address {len(metrics['areas_for_improvement'])} degraded metrics** by adjusting steering parameters.\n\n"
        
        report += "---\n"
        report += "*Generated by Algoverse Bias Mitigation Pipeline*\n"
        
        return report
    
    def print_summary(self):
        """Print concise summary of bias reduction analysis."""
        if not self.comparison_metrics:
            print("❌ No comparison metrics available. Run calculate_improvements() first.")
            return
        
        summary = self.comparison_metrics["summary"]
        
        print("\n" + "="*60)
        print("BIAS REDUCTION ANALYSIS SUMMARY")
        print("="*60)
        print(f"📊 Overall Improvement: {self.comparison_metrics['overall_improvement']:+.2f}%")
        print(f"🎯 Assessment: {summary['overall_assessment']}")
        print(f"📈 Datasets Analyzed: {summary['total_datasets_analyzed']}")
        print(f"✅ Significant Improvements: {summary['significant_improvements']}")
        print(f"⚠️  Areas for Improvement: {summary['areas_needing_attention']}")
        print("="*60)