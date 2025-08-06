#!/usr/bin/env python3
"""
Detailed Results Analyzer for Per-Dataset Statistics
Provides granular analysis of bias mitigation effectiveness across individual datasets
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class DetailedResultsAnalyzer:
    """Analyzes and compares results at the individual dataset level."""
    
    def __init__(self):
        self.baseline_results = None
        self.post_mitigation_results = None
        
    def load_results(self, baseline_file: str, post_mitigation_file: str):
        """Load baseline and post-mitigation results."""
        with open(baseline_file, 'r') as f:
            self.baseline_results = json.load(f)
        with open(post_mitigation_file, 'r') as f:
            self.post_mitigation_results = json.load(f)
            
    def extract_per_dataset_metrics(self, results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Extract detailed metrics for each dataset."""
        dataset_metrics = {}
        
        if 'evaluation_results' in results:
            for dataset_name, dataset_result in results['evaluation_results'].items():
                if 'metrics' in dataset_result:
                    metrics = dataset_result['metrics']
                    metadata = dataset_result.get('metadata', {})
                    
                    # Extract key metrics
                    dataset_metrics[dataset_name] = {
                        'bias_score': metrics.get(f'{dataset_name}_bias_score', 0.0),
                        'accuracy': metrics.get(f'{dataset_name}_accuracy', 0.0),
                        'stereotype_score': metrics.get(f'{dataset_name}_stereotype_score', 0.0),
                        'preference_gap': metrics.get(f'{dataset_name}_preference_gap', 0.0),
                        'total_samples': metadata.get('total_samples', 0),
                        'bias_types': metadata.get('bias_types', []),
                        'evaluation_mode': metadata.get('evaluation_mode', 'unknown'),
                        'success_rate': metrics.get(f'{dataset_name}_success_rate', 0.0)
                    }
                    
        return dataset_metrics
        
    def generate_per_dataset_comparison(self) -> pd.DataFrame:
        """Generate detailed per-dataset comparison."""
        baseline_metrics = self.extract_per_dataset_metrics(self.baseline_results)
        post_mitigation_metrics = self.extract_per_dataset_metrics(self.post_mitigation_results)
        
        comparison_data = []
        
        for dataset_name in baseline_metrics.keys():
            if dataset_name in post_mitigation_metrics:
                baseline = baseline_metrics[dataset_name]
                post_mit = post_mitigation_metrics[dataset_name]
                
                # Calculate improvements
                bias_reduction = baseline['bias_score'] - post_mit['bias_score']
                accuracy_change = post_mit['accuracy'] - baseline['accuracy']
                stereotype_reduction = baseline['stereotype_score'] - post_mit['stereotype_score']
                
                comparison_data.append({
                    'Dataset': dataset_name,
                    'Bias_Type': ', '.join(baseline['bias_types']),
                    'Evaluation_Mode': baseline['evaluation_mode'],
                    'Samples': baseline['total_samples'],
                    
                    # Baseline scores
                    'Baseline_Bias': baseline['bias_score'],
                    'Baseline_Accuracy': baseline['accuracy'],
                    'Baseline_Stereotype': baseline['stereotype_score'],
                    
                    # Post-mitigation scores  
                    'Post_Mit_Bias': post_mit['bias_score'],
                    'Post_Mit_Accuracy': post_mit['accuracy'],
                    'Post_Mit_Stereotype': post_mit['stereotype_score'],
                    
                    # Improvements
                    'Bias_Reduction': bias_reduction,
                    'Accuracy_Change': accuracy_change,
                    'Stereotype_Reduction': stereotype_reduction,
                    'Bias_Reduction_Pct': (bias_reduction / baseline['bias_score'] * 100) if baseline['bias_score'] > 0 else 0,
                    
                    # Success indicators
                    'Bias_Improved': bias_reduction > 0,
                    'Accuracy_Maintained': accuracy_change >= -0.05,  # Allow 5% degradation
                    'Overall_Success': (bias_reduction > 0) and (accuracy_change >= -0.05)
                })
                
        return pd.DataFrame(comparison_data)
        
    def generate_detailed_report(self, output_file: str = "detailed_results_analysis.md"):
        """Generate a comprehensive markdown report."""
        df = self.generate_per_dataset_comparison()
        
        report = []
        report.append("# Detailed Per-Dataset Bias Mitigation Analysis\\n")
        
        # Overall summary
        total_datasets = len(df)
        successful_datasets = df['Overall_Success'].sum()
        bias_improved_datasets = df['Bias_Improved'].sum()
        accuracy_maintained_datasets = df['Accuracy_Maintained'].sum()
        
        report.append("## 📊 Overall Summary\\n")
        report.append(f"- **Total Datasets Evaluated:** {total_datasets}")
        report.append(f"- **Datasets with Bias Reduction:** {bias_improved_datasets}/{total_datasets} ({bias_improved_datasets/total_datasets*100:.1f}%)")
        report.append(f"- **Datasets with Maintained Accuracy:** {accuracy_maintained_datasets}/{total_datasets} ({accuracy_maintained_datasets/total_datasets*100:.1f}%)")
        report.append(f"- **Overall Successful Datasets:** {successful_datasets}/{total_datasets} ({successful_datasets/total_datasets*100:.1f}%)\\n")
        
        # Top performers
        report.append("## 🏆 Top Performing Datasets (Highest Bias Reduction)\\n")
        top_performers = df.nlargest(5, 'Bias_Reduction_Pct')[['Dataset', 'Bias_Type', 'Bias_Reduction_Pct', 'Accuracy_Change']]
        report.append(top_performers.to_markdown(index=False))
        report.append("\\n")
        
        # Detailed per-dataset breakdown
        report.append("## 📋 Detailed Per-Dataset Results\\n")
        
        for bias_type in df['Bias_Type'].unique():
            if pd.isna(bias_type):
                continue
                
            report.append(f"### {bias_type} Bias\\n")
            bias_df = df[df['Bias_Type'].str.contains(bias_type.split(',')[0], na=False)].copy()
            
            if len(bias_df) > 0:
                # Sort by bias reduction
                bias_df = bias_df.sort_values('Bias_Reduction_Pct', ascending=False)
                
                for _, row in bias_df.iterrows():
                    dataset = row['Dataset']
                    success_icon = "✅" if row['Overall_Success'] else "❌"
                    bias_icon = "⬇️" if row['Bias_Improved'] else "⬆️"
                    acc_icon = "✅" if row['Accuracy_Maintained'] else "⚠️"
                    
                    report.append(f"#### {success_icon} {dataset}")
                    report.append(f"- **Evaluation Mode:** {row['Evaluation_Mode']}")
                    report.append(f"- **Samples:** {row['Samples']}")
                    report.append(f"- **Bias Score:** {row['Baseline_Bias']:.3f} → {row['Post_Mit_Bias']:.3f} {bias_icon} ({row['Bias_Reduction_Pct']:+.1f}%)")
                    report.append(f"- **Accuracy:** {row['Baseline_Accuracy']:.3f} → {row['Post_Mit_Accuracy']:.3f} {acc_icon} ({row['Accuracy_Change']:+.3f})")
                    
                    if row['Baseline_Stereotype'] > 0:
                        report.append(f"- **Stereotype Score:** {row['Baseline_Stereotype']:.3f} → {row['Post_Mit_Stereotype']:.3f} ({row['Stereotype_Reduction']:+.3f})")
                    report.append("")
                    
        # Problem datasets
        problem_datasets = df[~df['Overall_Success']]
        if len(problem_datasets) > 0:
            report.append("## ⚠️ Datasets Needing Attention\\n")
            for _, row in problem_datasets.iterrows():
                report.append(f"### {row['Dataset']}")
                
                issues = []
                if not row['Bias_Improved']:
                    issues.append(f"Bias increased by {abs(row['Bias_Reduction_Pct']):.1f}%")
                if not row['Accuracy_Maintained']:
                    issues.append(f"Accuracy dropped by {abs(row['Accuracy_Change']):.1f}%")
                    
                report.append(f"- **Issues:** {', '.join(issues)}")
                report.append(f"- **Bias Type:** {row['Bias_Type']}")
                report.append(f"- **Evaluation Mode:** {row['Evaluation_Mode']}")
                report.append("")
                
        # Save report
        with open(output_file, 'w') as f:
            f.write("\\n".join(report))
            
        return "\\n".join(report)
        
    def generate_csv_export(self, output_file: str = "detailed_results_data.csv"):
        """Export detailed results to CSV for further analysis."""
        df = self.generate_per_dataset_comparison()
        df.to_csv(output_file, index=False)
        print(f"📊 Detailed results exported to: {output_file}")
        
    def create_visualization(self, output_file: str = "bias_reduction_visualization.png"):
        """Create visualization of per-dataset improvements."""
        df = self.generate_per_dataset_comparison()
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Bias reduction by dataset
        datasets = df['Dataset']
        bias_reduction = df['Bias_Reduction_Pct']
        colors = ['green' if x > 0 else 'red' for x in bias_reduction]
        
        ax1.barh(datasets, bias_reduction, color=colors, alpha=0.7)
        ax1.set_xlabel('Bias Reduction (%)')
        ax1.set_title('Bias Reduction by Dataset')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Plot 2: Accuracy change vs bias reduction
        ax2.scatter(df['Bias_Reduction_Pct'], df['Accuracy_Change'] * 100, 
                   c=df['Overall_Success'].map({True: 'green', False: 'red'}), alpha=0.7)
        ax2.set_xlabel('Bias Reduction (%)')
        ax2.set_ylabel('Accuracy Change (%)')
        ax2.set_title('Bias Reduction vs Accuracy Trade-off')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        
        # Add quadrant labels
        ax2.text(0.05, 0.95, 'Ideal\\n(Bias↓, Acc↑)', transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        ax2.text(0.05, 0.05, 'Acceptable\\n(Bias↓, Acc→)', transform=ax2.transAxes,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📈 Visualization saved to: {output_file}")


def main():
    """Example usage of detailed results analyzer."""
    analyzer = DetailedResultsAnalyzer()
    
    # Example - you'd replace these with actual result files
    # analyzer.load_results("baseline_results.json", "post_mitigation_results.json")
    # 
    # # Generate detailed analysis
    # report = analyzer.generate_detailed_report()
    # analyzer.generate_csv_export()
    # analyzer.create_visualization()
    
    print("DetailedResultsAnalyzer ready for use!")
    print("Usage:")
    print("  analyzer = DetailedResultsAnalyzer()")
    print("  analyzer.load_results('baseline.json', 'post_mitigation.json')")
    print("  analyzer.generate_detailed_report()")


if __name__ == "__main__":
    main()