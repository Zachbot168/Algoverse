"""
Statistical Analysis Module for FIRM Bias Mitigation Research

This module provides comprehensive statistical analysis tools for bias evaluation
results across the 4-variant FIRM framework (baseline, fairsteer, sycophancy, FIRM).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, kruskal, mannwhitneyu, friedmanchisquare
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class FIRMStatisticalAnalyzer:
    """
    Comprehensive statistical analysis for FIRM bias mitigation results.
    
    Provides statistical tests, effect size calculations, confidence intervals,
    and visualization for bias evaluation across multiple techniques and datasets.
    """
    
    def __init__(self, results_directory: str = None):
        """
        Initialize the statistical analyzer.
        
        Args:
            results_directory: Path to directory containing evaluation results
        """
        self.results_dir = Path(results_directory) if results_directory else Path(".")
        self.techniques = ['baseline', 'fairsteer', 'sycophancy', 'firm']
        self.datasets = [
            'CrowsPairs', 'StereoSet', 'WinoBias', 'WinoGender', 'BBQ', 
            'SEAT', 'BOLD', 'BiosBias', 'TruthfulQA', 'SycophancyEval'
        ]
        
    def load_evaluation_results(self, file_path: str) -> Dict:
        """Load evaluation results from JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def extract_dataset_scores(self, results: Dict) -> pd.DataFrame:
        """
        Extract dataset scores into a structured DataFrame.
        
        Handles both 4-variant format and single evaluation format.
        
        Returns:
            DataFrame with columns: technique, dataset, score, model, seed
        """
        data = []
        
        # Check if this is a 4-variant comparison format
        if any(technique in results for technique in self.techniques):
            # 4-variant format: {'baseline': [...], 'fairsteer': [...], etc.}
            for technique in self.techniques:
                if technique in results:
                    for seed_data in results[technique]:
                        scores = seed_data.get('dataset_scores', {})
                        for dataset, score in scores.items():
                            data.append({
                                'technique': technique,
                                'dataset': dataset,
                                'score': score,
                                'model': seed_data.get('model', 'unknown'),
                                'seed': seed_data.get('seed', 0)
                            })
        else:
            # Single evaluation format: unified pipeline results
            technique = 'baseline'  # Default technique name
            model = results.get('model_name', 'unknown')
            seed = results.get('seed', 0)
            
            # Extract scores from dataset_results or aggregated_metrics
            if 'dataset_results' in results:
                for dataset, dataset_data in results['dataset_results'].items():
                    # Get the main metric for this dataset
                    if 'metrics' in dataset_data:
                        metrics = dataset_data['metrics']
                        
                        # Find the primary score metric for each dataset
                        score = None
                        if dataset == 'CrowsPairs':
                            score = metrics.get('crows_pairs_accuracy', metrics.get('crows_pairs_bias_score'))
                        elif dataset == 'StereoSet':
                            score = metrics.get('stereoset_bias_score')
                        elif dataset == 'WinoBias':
                            score = metrics.get('winobias_accuracy')
                        elif dataset == 'WinoGender':
                            score = metrics.get('winogender_accuracy')
                        elif dataset == 'BBQ':
                            score = metrics.get('bbq_accuracy')
                        elif dataset == 'SEAT':
                            score = metrics.get('seat_effect_size')
                        elif dataset == 'BOLD':
                            score = metrics.get('bold_bias_score')
                        elif dataset == 'BiosBias':
                            score = metrics.get('biosbias_accuracy')
                        elif dataset == 'TruthfulQA':
                            score = metrics.get('truthfulqa_accuracy')
                        elif dataset == 'SycophancyEval':
                            score = metrics.get('sycophancy_score')
                        
                        # If no specific metric found, try to find any score-like metric
                        if score is None:
                            for key, value in metrics.items():
                                if 'score' in key.lower() or 'accuracy' in key.lower():
                                    score = value
                                    break
                        
                        if score is not None:
                            data.append({
                                'technique': technique,
                                'dataset': dataset,
                                'score': score,
                                'model': model,
                                'seed': seed
                            })
            
            # Also check aggregated_metrics.dataset_summary for additional data
            elif 'aggregated_metrics' in results and 'dataset_summary' in results['aggregated_metrics']:
                for dataset, dataset_data in results['aggregated_metrics']['dataset_summary'].items():
                    score = dataset_data.get('main_metric')
                    if score is not None:
                        data.append({
                            'technique': technique,
                            'dataset': dataset,
                            'score': score,
                            'model': model,
                            'seed': seed
                        })
        
        return pd.DataFrame(data)
    
    def compute_statistical_significance(self, df: pd.DataFrame) -> Dict:
        """
        Compute statistical significance tests across techniques.
        
        Returns:
            Dictionary containing statistical test results
        """
        results = {}
        
        for dataset in self.datasets:
            dataset_data = df[df['dataset'] == dataset]
            if len(dataset_data) < 2:
                continue
                
            results[dataset] = {}
            
            # Get scores for each technique
            technique_scores = {}
            for technique in self.techniques:
                scores = dataset_data[dataset_data['technique'] == technique]['score'].values
                if len(scores) > 0:
                    technique_scores[technique] = scores
            
            # Pairwise comparisons
            results[dataset]['pairwise'] = {}
            techniques = list(technique_scores.keys())
            
            for i in range(len(techniques)):
                for j in range(i+1, len(techniques)):
                    tech1, tech2 = techniques[i], techniques[j]
                    scores1, scores2 = technique_scores[tech1], technique_scores[tech2]
                    
                    # Choose appropriate test based on data
                    if len(scores1) >= 30 and len(scores2) >= 30:
                        # Large sample: use t-test
                        statistic, p_value = ttest_rel(scores1, scores2) if len(scores1) == len(scores2) else stats.ttest_ind(scores1, scores2)
                        test_type = "t-test"
                    else:
                        # Small sample: use Mann-Whitney U
                        statistic, p_value = mannwhitneyu(scores1, scores2, alternative='two-sided')
                        test_type = "Mann-Whitney U"
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(scores1)-1)*np.var(scores1) + (len(scores2)-1)*np.var(scores2)) / (len(scores1)+len(scores2)-2))
                    cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std if pooled_std > 0 else 0
                    
                    results[dataset]['pairwise'][f'{tech1}_vs_{tech2}'] = {
                        'statistic': float(statistic),
                        'p_value': float(p_value),
                        'effect_size': float(cohens_d),
                        'test_type': test_type,
                        'significant': p_value < 0.05,
                        'mean_diff': float(np.mean(scores1) - np.mean(scores2))
                    }
            
            # Overall ANOVA/Kruskal-Wallis test
            if len(technique_scores) >= 3:
                score_groups = [scores for scores in technique_scores.values()]
                try:
                    if all(len(group) >= 5 for group in score_groups):
                        # Use ANOVA for larger samples
                        f_stat, p_val = stats.f_oneway(*score_groups)
                        test_name = "ANOVA"
                    else:
                        # Use Kruskal-Wallis for smaller samples
                        h_stat, p_val = kruskal(*score_groups)
                        test_name = "Kruskal-Wallis"
                        f_stat = h_stat
                    
                    results[dataset]['overall'] = {
                        'test': test_name,
                        'statistic': float(f_stat),
                        'p_value': float(p_val),
                        'significant': p_val < 0.05
                    }
                except Exception as e:
                    results[dataset]['overall'] = {'error': str(e)}
        
        return results
    
    def compute_confidence_intervals(self, df: pd.DataFrame, confidence_level: float = 0.95) -> Dict:
        """
        Compute confidence intervals for each technique-dataset combination.
        """
        results = {}
        alpha = 1 - confidence_level
        
        for technique in self.techniques:
            results[technique] = {}
            tech_data = df[df['technique'] == technique]
            
            for dataset in self.datasets:
                dataset_scores = tech_data[tech_data['dataset'] == dataset]['score'].values
                
                if len(dataset_scores) > 1:
                    mean_score = np.mean(dataset_scores)
                    std_score = np.std(dataset_scores, ddof=1)
                    n = len(dataset_scores)
                    
                    # t-distribution critical value
                    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
                    margin_error = t_crit * (std_score / np.sqrt(n))
                    
                    results[technique][dataset] = {
                        'mean': float(mean_score),
                        'std': float(std_score),
                        'n': int(n),
                        'ci_lower': float(mean_score - margin_error),
                        'ci_upper': float(mean_score + margin_error),
                        'margin_error': float(margin_error)
                    }
                elif len(dataset_scores) == 1:
                    results[technique][dataset] = {
                        'mean': float(dataset_scores[0]),
                        'std': 0.0,
                        'n': 1,
                        'ci_lower': float(dataset_scores[0]),
                        'ci_upper': float(dataset_scores[0]),
                        'margin_error': 0.0
                    }
        
        return results
    
    def compute_effect_sizes(self, df: pd.DataFrame) -> Dict:
        """
        Compute effect sizes (Cohen's d) for all pairwise comparisons.
        """
        results = {}
        
        for dataset in self.datasets:
            dataset_data = df[df['dataset'] == dataset]
            results[dataset] = {}
            
            techniques = dataset_data['technique'].unique()
            
            for i, tech1 in enumerate(techniques):
                for tech2 in techniques[i+1:]:
                    scores1 = dataset_data[dataset_data['technique'] == tech1]['score'].values
                    scores2 = dataset_data[dataset_data['technique'] == tech2]['score'].values
                    
                    if len(scores1) > 0 and len(scores2) > 0:
                        # Cohen's d
                        pooled_std = np.sqrt(((len(scores1)-1)*np.var(scores1, ddof=1) + 
                                            (len(scores2)-1)*np.var(scores2, ddof=1)) / 
                                           (len(scores1)+len(scores2)-2))
                        
                        cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std if pooled_std > 0 else 0
                        
                        # Interpretation
                        if abs(cohens_d) < 0.2:
                            interpretation = "negligible"
                        elif abs(cohens_d) < 0.5:
                            interpretation = "small"
                        elif abs(cohens_d) < 0.8:
                            interpretation = "medium"
                        else:
                            interpretation = "large"
                        
                        results[dataset][f'{tech1}_vs_{tech2}'] = {
                            'cohens_d': float(cohens_d),
                            'interpretation': interpretation,
                            'mean1': float(np.mean(scores1)),
                            'mean2': float(np.mean(scores2)),
                            'std1': float(np.std(scores1, ddof=1)),
                            'std2': float(np.std(scores2, ddof=1))
                        }
        
        return results
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive summary statistics.
        """
        summary = {}
        
        for technique in self.techniques:
            tech_data = df[df['technique'] == technique]
            summary[technique] = {}
            
            for dataset in self.datasets:
                dataset_scores = tech_data[tech_data['dataset'] == dataset]['score'].values
                
                if len(dataset_scores) > 0:
                    summary[technique][dataset] = {
                        'count': int(len(dataset_scores)),
                        'mean': float(np.mean(dataset_scores)),
                        'median': float(np.median(dataset_scores)),
                        'std': float(np.std(dataset_scores, ddof=1)) if len(dataset_scores) > 1 else 0.0,
                        'min': float(np.min(dataset_scores)),
                        'max': float(np.max(dataset_scores)),
                        'q25': float(np.percentile(dataset_scores, 25)),
                        'q75': float(np.percentile(dataset_scores, 75)),
                        'iqr': float(np.percentile(dataset_scores, 75) - np.percentile(dataset_scores, 25)),
                        'cv': float(np.std(dataset_scores, ddof=1) / np.mean(dataset_scores)) if np.mean(dataset_scores) != 0 and len(dataset_scores) > 1 else 0.0
                    }
        
        return summary
    
    def run_complete_analysis(self, results_file: str) -> Dict:
        """
        Run complete statistical analysis pipeline.
        
        Args:
            results_file: Path to evaluation results JSON file
            
        Returns:
            Dictionary containing all statistical analysis results
        """
        print("🔬 Running Complete Statistical Analysis...")
        
        # Load data
        results = self.load_evaluation_results(results_file)
        df = self.extract_dataset_scores(results)
        
        print(f"📊 Loaded {len(df)} data points across {df['technique'].nunique()} techniques")
        
        # Run analyses
        analysis_results = {
            'summary_statistics': self.generate_summary_statistics(df),
            'confidence_intervals': self.compute_confidence_intervals(df),
            'effect_sizes': self.compute_effect_sizes(df),
            'statistical_significance': self.compute_statistical_significance(df),
            'data_info': {
                'total_observations': len(df),
                'techniques': list(df['technique'].unique()),
                'datasets': list(df['dataset'].unique()),
                'models': list(df['model'].unique())
            }
        }
        
        print("✅ Statistical analysis complete!")
        return analysis_results
    
    def save_analysis_results(self, analysis_results: Dict, output_file: str):
        """Save analysis results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        print(f"💾 Analysis results saved to: {output_file}")

def main():
    """Example usage of the statistical analyzer."""
    
    # Initialize analyzer
    analyzer = FIRMStatisticalAnalyzer()
    
    # Look for recent results files
    results_file = None
    
    # Check for recent unified pipeline results
    unified_runs = Path("../unified_pipeline/unified_pipeline_runs")
    if unified_runs.exists():
        run_dirs = sorted([d for d in unified_runs.iterdir() if d.is_dir()], reverse=True)
        for run_dir in run_dirs:
            potential_file = run_dir / "evaluation" / "baseline" / "evaluation_results.json"
            if potential_file.exists():
                results_file = str(potential_file)
                break
    
    # Fallback to example file
    if not results_file:
        results_file = "evaluation_results.json"
    
    if Path(results_file).exists():
        print(f"📊 Analyzing results from: {results_file}")
        
        # Run complete analysis
        results = analyzer.run_complete_analysis(results_file)
        
        # Save results
        output_file = "statistical_analysis_results.json"
        analyzer.save_analysis_results(results, output_file)
        
        # Print key findings
        print("\n📈 KEY STATISTICAL FINDINGS:")
        print("="*50)
        
        # Data overview
        data_info = results['data_info']
        print(f"\n📋 Data Overview:")
        print(f"   Total observations: {data_info['total_observations']}")
        print(f"   Techniques analyzed: {', '.join(data_info['techniques'])}")
        print(f"   Datasets evaluated: {', '.join(data_info['datasets'])}")
        print(f"   Models: {', '.join(data_info['models'])}")
        
        # Summary by technique
        for technique in data_info['techniques']:
            if technique in results['summary_statistics']:
                print(f"\n🎯 {technique.upper()} Summary:")
                tech_stats = results['summary_statistics'][technique]
                dataset_means = [stats['mean'] for stats in tech_stats.values() if 'mean' in stats]
                if dataset_means:
                    print(f"   Average across datasets: {np.mean(dataset_means):.3f}")
                    print(f"   Standard deviation: {np.std(dataset_means):.3f}")
                    if len(tech_stats) > 1:
                        print(f"   Best performing dataset: {max(tech_stats.items(), key=lambda x: x[1].get('mean', 0))[0]}")
                        print(f"   Worst performing dataset: {min(tech_stats.items(), key=lambda x: x[1].get('mean', 1))[0]}")
        
        # Significance findings (only for multi-technique comparisons)
        if len(data_info['techniques']) > 1:
            sig_results = results['statistical_significance']
            significant_comparisons = []
            
            for dataset, tests in sig_results.items():
                if 'pairwise' in tests:
                    for comparison, result in tests['pairwise'].items():
                        if result.get('significant', False):
                            significant_comparisons.append((dataset, comparison, result['p_value'], result['effect_size']))
            
            print(f"\n📊 Found {len(significant_comparisons)} statistically significant differences (p < 0.05)")
            
            for dataset, comparison, p_val, effect_size in significant_comparisons[:5]:  # Show top 5
                print(f"   {dataset}: {comparison} (p={p_val:.4f}, Cohen's d={effect_size:.3f})")
        else:
            print(f"\n📊 Single technique analysis - no statistical comparisons performed")
            print("   To compare techniques, run 4-variant evaluation with run_integrated_pipeline.py")
    
    else:
        print(f"❌ Results file not found: {results_file}")
        print("📝 To generate results, run:")
        print("   cd ../unified_pipeline")
        print("   CUDA_VISIBLE_DEVICES=0 python run_unified_pipeline.py --model-config configs/models/qwen2.5-3b-instruct.yaml --suite quick_evaluation")
        print("   Then rerun this analysis.")

if __name__ == "__main__":
    main()