"""
Advanced Results Analysis for FIRM Bias Mitigation Research

This module provides comprehensive analysis tools for interpreting bias evaluation
results, including trend analysis, comparative effectiveness, and research insights.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class FIRMResultsAnalyzer:
    """
    Advanced analysis tools for FIRM bias mitigation results.
    
    Provides trend analysis, clustering, dimensionality reduction,
    and comprehensive research insights.
    """
    
    def __init__(self):
        """Initialize results analyzer."""
        self.techniques = ['baseline', 'fairsteer', 'sycophancy', 'firm']
        self.datasets = [
            'CrowsPairs', 'StereoSet', 'WinoBias', 'WinoGender', 'BBQ', 
            'SEAT', 'BOLD', 'BiosBias', 'TruthfulQA', 'SycophancyEval'
        ]
        
        # Dataset categorization by bias type
        self.bias_categories = {
            'stereotypical': ['CrowsPairs', 'StereoSet', 'SEAT'],
            'gender': ['WinoBias', 'WinoGender', 'BiosBias'],
            'demographic': ['BBQ', 'BOLD'],
            'truthfulness': ['TruthfulQA', 'SycophancyEval']
        }
        
        # Score interpretation (higher = better or lower = better)
        self.score_direction = {
            'CrowsPairs': 'higher_better',    # Anti-stereotypical preference
            'WinoBias': 'higher_better',      # Accuracy
            'WinoGender': 'higher_better',    # Accuracy
            'BBQ': 'higher_better',           # Unbiased accuracy
            'TruthfulQA': 'higher_better',    # Truthfulness
            'BiosBias': 'higher_better',      # Unbiased accuracy
            'StereoSet': 'lower_better',      # Stereotype bias
            'SEAT': 'lower_better',           # Implicit association
            'BOLD': 'lower_better',           # Sentiment bias
            'SycophancyEval': 'lower_better'  # Sycophantic agreement
        }
    
    def load_and_structure_data(self, results_file: str) -> pd.DataFrame:
        """Load evaluation results and create structured DataFrame."""
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        data = []
        for technique in self.techniques:
            if technique in results:
                for seed_data in results[technique]:
                    scores = seed_data.get('dataset_scores', {})
                    for dataset, score in scores.items():
                        data.append({
                            'technique': technique,
                            'dataset': dataset,
                            'score': score,
                            'normalized_score': self.normalize_score(dataset, score),
                            'model': seed_data.get('model', 'unknown'),
                            'seed': seed_data.get('seed', 0),
                            'bias_category': self.get_bias_category(dataset)
                        })
        
        return pd.DataFrame(data)
    
    def normalize_score(self, dataset: str, score: float) -> float:
        """
        Normalize scores so higher values always indicate better performance.
        """
        if self.score_direction.get(dataset) == 'lower_better':
            # For lower-is-better metrics, return 1 - score (assuming scores are 0-1)
            return 1 - score if score <= 1 else -score
        else:
            return score
    
    def get_bias_category(self, dataset: str) -> str:
        """Get bias category for a dataset."""
        for category, datasets in self.bias_categories.items():
            if dataset in datasets:
                return category
        return 'other'
    
    def compute_technique_effectiveness(self, df: pd.DataFrame) -> Dict:
        """
        Compute comprehensive effectiveness metrics for each technique.
        """
        effectiveness = {}
        
        for technique in self.techniques:
            tech_data = df[df['technique'] == technique]
            
            if len(tech_data) == 0:
                continue
            
            # Overall metrics
            overall_scores = tech_data['normalized_score']
            
            effectiveness[technique] = {
                'overall': {
                    'mean': float(overall_scores.mean()),
                    'median': float(overall_scores.median()),
                    'std': float(overall_scores.std()),
                    'min': float(overall_scores.min()),
                    'max': float(overall_scores.max()),
                    'iqr': float(overall_scores.quantile(0.75) - overall_scores.quantile(0.25)),
                    'cv': float(overall_scores.std() / overall_scores.mean()) if overall_scores.mean() != 0 else 0
                }
            }
            
            # Category-wise analysis
            effectiveness[technique]['by_category'] = {}
            for category in self.bias_categories.keys():
                cat_data = tech_data[tech_data['bias_category'] == category]
                if len(cat_data) > 0:
                    cat_scores = cat_data['normalized_score']
                    effectiveness[technique]['by_category'][category] = {
                        'mean': float(cat_scores.mean()),
                        'std': float(cat_scores.std()),
                        'n_datasets': len(cat_scores)
                    }
            
            # Dataset-wise analysis
            effectiveness[technique]['by_dataset'] = {}
            for dataset in self.datasets:
                dataset_data = tech_data[tech_data['dataset'] == dataset]
                if len(dataset_data) > 0:
                    dataset_scores = dataset_data['normalized_score']
                    effectiveness[technique]['by_dataset'][dataset] = {
                        'mean': float(dataset_scores.mean()),
                        'std': float(dataset_scores.std()),
                        'n_samples': len(dataset_scores)
                    }
        
        return effectiveness
    
    def identify_best_techniques(self, df: pd.DataFrame) -> Dict:
        """
        Identify best-performing techniques across different criteria.
        """
        results = {
            'overall_best': {},
            'category_best': {},
            'dataset_best': {},
            'consistency_best': {}
        }
        
        # Overall best technique
        technique_means = df.groupby('technique')['normalized_score'].mean()
        results['overall_best'] = {
            'technique': technique_means.idxmax(),
            'score': float(technique_means.max()),
            'ranking': technique_means.sort_values(ascending=False).to_dict()
        }
        
        # Best by bias category
        for category in self.bias_categories.keys():
            cat_data = df[df['bias_category'] == category]
            if len(cat_data) > 0:
                cat_means = cat_data.groupby('technique')['normalized_score'].mean()
                results['category_best'][category] = {
                    'technique': cat_means.idxmax(),
                    'score': float(cat_means.max()),
                    'ranking': cat_means.sort_values(ascending=False).to_dict()
                }
        
        # Best by dataset
        for dataset in self.datasets:
            dataset_data = df[df['dataset'] == dataset]
            if len(dataset_data) > 0:
                dataset_means = dataset_data.groupby('technique')['normalized_score'].mean()
                results['dataset_best'][dataset] = {
                    'technique': dataset_means.idxmax(),
                    'score': float(dataset_means.max()),
                    'ranking': dataset_means.sort_values(ascending=False).to_dict()
                }
        
        # Most consistent technique (lowest coefficient of variation)
        technique_cv = df.groupby('technique')['normalized_score'].agg(['mean', 'std'])
        technique_cv['cv'] = technique_cv['std'] / technique_cv['mean']
        results['consistency_best'] = {
            'technique': technique_cv['cv'].idxmin(),
            'cv': float(technique_cv['cv'].min()),
            'ranking': technique_cv['cv'].sort_values().to_dict()
        }
        
        return results
    
    def perform_clustering_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Perform clustering analysis to identify technique and dataset patterns.
        """
        # Create technique-dataset matrix
        pivot_data = df.groupby(['technique', 'dataset'])['normalized_score'].mean().unstack(fill_value=0)
        
        if pivot_data.empty:
            return {'error': 'No data available for clustering'}
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pivot_data.T)  # Transpose for dataset clustering
        
        # K-means clustering of datasets
        n_clusters = min(4, len(pivot_data.columns))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        dataset_clusters = kmeans.fit_predict(scaled_data)
        
        # Hierarchical clustering for techniques
        technique_linkage = linkage(pivot_data.values, method='ward')
        
        clustering_results = {
            'dataset_clusters': {
                dataset: int(cluster) 
                for dataset, cluster in zip(pivot_data.columns, dataset_clusters)
            },
            'cluster_characteristics': {},
            'technique_similarity': technique_linkage.tolist(),
            'technique_order': pivot_data.index.tolist()
        }
        
        # Analyze cluster characteristics
        for cluster_id in range(n_clusters):
            cluster_datasets = [
                dataset for dataset, cluster in clustering_results['dataset_clusters'].items()
                if cluster == cluster_id
            ]
            
            if cluster_datasets:
                cluster_data = df[df['dataset'].isin(cluster_datasets)]
                cluster_stats = cluster_data.groupby('technique')['normalized_score'].mean()
                
                clustering_results['cluster_characteristics'][f'cluster_{cluster_id}'] = {
                    'datasets': cluster_datasets,
                    'best_technique': cluster_stats.idxmax(),
                    'technique_scores': cluster_stats.to_dict(),
                    'cluster_size': len(cluster_datasets)
                }
        
        return clustering_results
    
    def analyze_improvement_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze improvement patterns relative to baseline.
        """
        baseline_scores = df[df['technique'] == 'baseline'].set_index('dataset')['normalized_score']
        
        improvement_analysis = {
            'technique_improvements': {},
            'dataset_improvements': {},
            'improvement_stats': {}
        }
        
        # Calculate improvements for each technique
        for technique in ['fairsteer', 'sycophancy', 'firm']:
            tech_data = df[df['technique'] == technique]
            
            improvements = []
            for dataset in tech_data['dataset'].unique():
                tech_score = tech_data[tech_data['dataset'] == dataset]['normalized_score'].mean()
                baseline_score = baseline_scores.get(dataset, np.nan)
                
                if not np.isnan(baseline_score):
                    improvement = tech_score - baseline_score
                    improvements.append({
                        'dataset': dataset,
                        'baseline_score': float(baseline_score),
                        'technique_score': float(tech_score),
                        'improvement': float(improvement),
                        'relative_improvement': float(improvement / baseline_score) if baseline_score != 0 else 0
                    })
            
            if improvements:
                improvement_values = [imp['improvement'] for imp in improvements]
                improvement_analysis['technique_improvements'][technique] = {
                    'improvements': improvements,
                    'mean_improvement': float(np.mean(improvement_values)),
                    'median_improvement': float(np.median(improvement_values)),
                    'std_improvement': float(np.std(improvement_values)),
                    'positive_improvements': sum(1 for imp in improvement_values if imp > 0),
                    'total_comparisons': len(improvement_values),
                    'success_rate': sum(1 for imp in improvement_values if imp > 0) / len(improvement_values)
                }
        
        # Dataset-wise improvement analysis
        for dataset in self.datasets:
            dataset_improvements = {}
            baseline_score = baseline_scores.get(dataset, np.nan)
            
            if not np.isnan(baseline_score):
                for technique in ['fairsteer', 'sycophancy', 'firm']:
                    tech_data = df[(df['technique'] == technique) & (df['dataset'] == dataset)]
                    if len(tech_data) > 0:
                        tech_score = tech_data['normalized_score'].mean()
                        improvement = tech_score - baseline_score
                        dataset_improvements[technique] = {
                            'score': float(tech_score),
                            'improvement': float(improvement),
                            'relative_improvement': float(improvement / baseline_score) if baseline_score != 0 else 0
                        }
                
                if dataset_improvements:
                    best_technique = max(dataset_improvements.items(), 
                                       key=lambda x: x[1]['improvement'])
                    
                    improvement_analysis['dataset_improvements'][dataset] = {
                        'baseline_score': float(baseline_score),
                        'technique_results': dataset_improvements,
                        'best_technique': best_technique[0],
                        'best_improvement': best_technique[1]['improvement']
                    }
        
        return improvement_analysis
    
    def generate_research_insights(self, df: pd.DataFrame, effectiveness: Dict, 
                                 clustering: Dict, improvements: Dict) -> List[str]:
        """
        Generate actionable research insights from the analysis.
        """
        insights = []
        
        # Overall performance insights
        best_overall = effectiveness.get('overall_best', {}).get('technique')
        if best_overall:
            insights.append(f"🏆 **{best_overall.upper()}** shows the best overall performance across all bias types")
        
        # Category-specific insights
        category_winners = {}
        for category, result in effectiveness.get('category_best', {}).items():
            winner = result.get('technique')
            if winner:
                category_winners[winner] = category_winners.get(winner, []) + [category]
        
        for technique, categories in category_winners.items():
            if len(categories) > 1:
                insights.append(f"🎯 **{technique.upper()}** excels in {', '.join(categories)} bias types")
            else:
                insights.append(f"🎯 **{technique.upper()}** is most effective for {categories[0]} bias")
        
        # Consistency insights
        most_consistent = effectiveness.get('consistency_best', {}).get('technique')
        if most_consistent:
            insights.append(f"📊 **{most_consistent.upper()}** shows the most consistent performance across datasets")
        
        # Improvement patterns
        for technique, data in improvements.get('technique_improvements', {}).items():
            success_rate = data.get('success_rate', 0)
            mean_improvement = data.get('mean_improvement', 0)
            
            if success_rate > 0.8:
                insights.append(f"✅ **{technique.upper()}** improves over baseline in {success_rate:.0%} of cases")
            elif success_rate > 0.5:
                insights.append(f"⚖️ **{technique.upper()}** shows mixed results (improves in {success_rate:.0%} of cases)")
            else:
                insights.append(f"⚠️ **{technique.upper()}** may not be consistently effective (improves in {success_rate:.0%} of cases)")
            
            if mean_improvement > 0.1:
                insights.append(f"📈 **{technique.upper()}** provides substantial bias reduction (Δ = {mean_improvement:+.3f})")
            elif mean_improvement > 0.05:
                insights.append(f"📈 **{technique.upper()}** provides moderate bias reduction (Δ = {mean_improvement:+.3f})")
        
        # Clustering insights
        if 'cluster_characteristics' in clustering:
            for cluster_name, cluster_data in clustering['cluster_characteristics'].items():
                datasets = cluster_data.get('datasets', [])
                best_tech = cluster_data.get('best_technique')
                
                if len(datasets) > 1 and best_tech:
                    dataset_list = ', '.join(datasets)
                    insights.append(f"🔍 **{best_tech.upper()}** is particularly effective for: {dataset_list}")
        
        # Dataset-specific insights
        challenging_datasets = []
        easy_datasets = []
        
        for dataset, data in improvements.get('dataset_improvements', {}).items():
            best_improvement = data.get('best_improvement', 0)
            
            if best_improvement < 0.02:  # Very small improvement
                challenging_datasets.append(dataset)
            elif best_improvement > 0.1:  # Large improvement
                easy_datasets.append(dataset)
        
        if challenging_datasets:
            insights.append(f"🚨 **Challenging datasets** requiring further research: {', '.join(challenging_datasets)}")
        
        if easy_datasets:
            insights.append(f"✨ **High-impact datasets** showing strong bias reduction: {', '.join(easy_datasets)}")
        
        # Methodology insights
        techniques_with_data = df['technique'].unique()
        if len(techniques_with_data) >= 3:
            insights.append("📋 **Recommendation**: Focus on head-to-head comparisons between top 2-3 techniques")
        
        if 'firm' in techniques_with_data:
            firm_data = improvements.get('technique_improvements', {}).get('firm', {})
            if firm_data.get('success_rate', 0) > 0.7:
                insights.append("🧠 **FIRM methodology** shows promise as a comprehensive bias mitigation approach")
            else:
                insights.append("🧠 **FIRM methodology** requires further optimization for consistent effectiveness")
        
        # Statistical insights
        total_comparisons = sum(
            data.get('total_comparisons', 0) 
            for data in improvements.get('technique_improvements', {}).values()
        )
        
        if total_comparisons < 30:
            insights.append("⚠️ **Sample size**: Consider increasing evaluation runs for more robust statistical analysis")
        elif total_comparisons > 100:
            insights.append("✅ **Sample size**: Sufficient data for robust statistical conclusions")
        
        return insights
    
    def create_comprehensive_analysis_report(self, results_file: str, save_path: str = None) -> Dict:
        """
        Generate comprehensive analysis report with all insights.
        """
        print("🔬 Generating comprehensive FIRM analysis report...")
        
        # Load and analyze data
        df = self.load_and_structure_data(results_file)
        effectiveness = self.compute_technique_effectiveness(df)
        clustering = self.perform_clustering_analysis(df)
        improvements = self.analyze_improvement_patterns(df)
        insights = self.generate_research_insights(df, effectiveness, clustering, improvements)
        
        # Compile full report
        report = {
            'data_summary': {
                'total_observations': len(df),
                'techniques': list(df['technique'].unique()),
                'datasets': list(df['dataset'].unique()),
                'bias_categories': list(df['bias_category'].unique())
            },
            'technique_effectiveness': effectiveness,
            'best_techniques': self.identify_best_techniques(df),
            'clustering_analysis': clustering,
            'improvement_patterns': improvements,
            'research_insights': insights,
            'metadata': {
                'analysis_date': pd.Timestamp.now().isoformat(),
                'total_insights': len(insights)
            }
        }
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"📊 Comprehensive analysis report saved to: {save_path}")
        
        # Print key insights
        print("\n🎯 KEY RESEARCH INSIGHTS:")
        print("=" * 50)
        for i, insight in enumerate(insights[:10], 1):  # Show top 10 insights
            print(f"{i:2d}. {insight}")
        
        if len(insights) > 10:
            print(f"\n... and {len(insights) - 10} more insights in the full report")
        
        return report

def main():
    """Example usage of the results analyzer."""
    
    # Initialize analyzer
    analyzer = FIRMResultsAnalyzer()
    
    # Example results file
    results_file = "evaluation_results.json"
    
    if Path(results_file).exists():
        print("🔬 FIRM Results Analysis")
        print("=" * 30)
        
        # Generate comprehensive report
        report = analyzer.create_comprehensive_analysis_report(
            results_file, 
            "comprehensive_analysis_report.json"
        )
        
        # Show summary statistics
        data_summary = report['data_summary']
        print(f"\n📊 Analysis Summary:")
        print(f"   Total observations: {data_summary['total_observations']}")
        print(f"   Techniques analyzed: {', '.join(data_summary['techniques'])}")
        print(f"   Datasets included: {len(data_summary['datasets'])}")
        print(f"   Research insights generated: {report['metadata']['total_insights']}")
        
        # Show best techniques
        best_techniques = report['best_techniques']
        print(f"\n🏆 Best Techniques:")
        print(f"   Overall: {best_techniques['overall_best']['technique']}")
        print(f"   Most consistent: {best_techniques['consistency_best']['technique']}")
        
        print("\n✅ Comprehensive analysis complete!")
        
    else:
        print(f"❌ Results file not found: {results_file}")
        print("Please provide evaluation results to analyze.")

if __name__ == "__main__":
    main()