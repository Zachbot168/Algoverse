"""
Advanced Visualization Tools for FIRM Bias Mitigation Research

This module provides comprehensive visualization capabilities for analyzing
bias evaluation results across the 4-variant FIRM framework.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

class FIRMVisualizationTools:
    """
    Advanced visualization tools for FIRM bias mitigation analysis.
    
    Provides statistical plots, comparative visualizations, and interactive
    dashboards for understanding bias evaluation results.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize visualization tools with default figure size."""
        self.figsize = figsize
        self.techniques = ['baseline', 'fairsteer', 'sycophancy', 'firm']
        self.datasets = [
            'CrowsPairs', 'StereoSet', 'WinoBias', 'WinoGender', 'BBQ', 
            'SEAT', 'BOLD', 'BiosBias', 'TruthfulQA', 'SycophancyEval'
        ]
        
        # Color scheme for techniques
        self.technique_colors = {
            'baseline': '#FF6B6B',     # Red
            'fairsteer': '#4ECDC4',    # Teal
            'sycophancy': '#45B7D1',   # Blue
            'firm': '#96CEB4'          # Green
        }
        
    def load_data(self, results_file: str) -> pd.DataFrame:
        """Load and structure evaluation data."""
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
                            'model': seed_data.get('model', 'unknown'),
                            'seed': seed_data.get('seed', 0)
                        })
        
        return pd.DataFrame(data)
    
    def create_technique_comparison_plot(self, df: pd.DataFrame, save_path: str = None) -> plt.Figure:
        """
        Create comprehensive technique comparison visualization.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('FIRM Framework: Comprehensive Technique Comparison', fontsize=16, fontweight='bold')
        
        # 1. Box plot comparison
        ax1 = axes[0, 0]
        df_pivot = df.pivot_table(index=['dataset', 'seed'], columns='technique', values='score').reset_index()
        df_melt = df_pivot.melt(id_vars=['dataset', 'seed'], var_name='technique', value_name='score')
        
        sns.boxplot(data=df_melt, x='technique', y='score', ax=ax1, palette=self.technique_colors)
        ax1.set_title('Distribution of Bias Scores by Technique', fontweight='bold')
        ax1.set_ylabel('Bias Score')
        ax1.set_xlabel('Technique')
        ax1.grid(True, alpha=0.3)
        
        # 2. Heatmap of mean scores
        ax2 = axes[0, 1]
        heatmap_data = df.groupby(['technique', 'dataset'])['score'].mean().unstack(fill_value=0)
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax2, cbar_kws={'label': 'Bias Score'})
        ax2.set_title('Mean Bias Scores Heatmap', fontweight='bold')
        ax2.set_xlabel('Dataset')
        ax2.set_ylabel('Technique')
        
        # 3. Radar chart for technique profiles
        ax3 = axes[1, 0]
        technique_means = df.groupby('technique')['score'].agg(['mean', 'std']).reset_index()
        
        x_pos = np.arange(len(technique_means))
        bars = ax3.bar(x_pos, technique_means['mean'], yerr=technique_means['std'], 
                      capsize=5, color=[self.technique_colors[t] for t in technique_means['technique']],
                      alpha=0.7, edgecolor='black')
        
        ax3.set_xlabel('Technique')
        ax3.set_ylabel('Mean Bias Score ± SD')
        ax3.set_title('Overall Performance with Error Bars', fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(technique_means['technique'], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean_val in zip(bars, technique_means['mean']):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Dataset-specific comparison
        ax4 = axes[1, 1]
        dataset_improvement = []
        
        for dataset in self.datasets:
            dataset_data = df[df['dataset'] == dataset]
            if len(dataset_data) > 0:
                baseline_scores = dataset_data[dataset_data['technique'] == 'baseline']['score']
                firm_scores = dataset_data[dataset_data['technique'] == 'firm']['score']
                
                if len(baseline_scores) > 0 and len(firm_scores) > 0:
                    improvement = np.mean(firm_scores) - np.mean(baseline_scores)
                    dataset_improvement.append((dataset, improvement))
        
        if dataset_improvement:
            datasets, improvements = zip(*dataset_improvement)
            colors = ['green' if imp > 0 else 'red' for imp in improvements]
            
            bars = ax4.barh(range(len(datasets)), improvements, color=colors, alpha=0.7)
            ax4.set_yticks(range(len(datasets)))
            ax4.set_yticklabels(datasets)
            ax4.set_xlabel('FIRM vs Baseline Score Difference')
            ax4.set_title('FIRM Improvement over Baseline', fontweight='bold')
            ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            ax4.grid(True, alpha=0.3)
            
            # Add value labels
            for i, (bar, imp) in enumerate(zip(bars, improvements)):
                ax4.text(imp + 0.001 if imp >= 0 else imp - 0.001, i,
                        f'{imp:+.3f}', ha='left' if imp >= 0 else 'right', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Technique comparison plot saved to: {save_path}")
        
        return fig
    
    def create_statistical_significance_plot(self, analysis_results: Dict, save_path: str = None) -> plt.Figure:
        """
        Create visualization for statistical significance results.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Statistical Significance Analysis', fontsize=16, fontweight='bold')
        
        # Extract significance data
        sig_results = analysis_results.get('statistical_significance', {})
        
        # 1. P-value heatmap
        ax1 = axes[0, 0]
        p_value_matrix = []
        comparison_labels = []
        datasets_with_data = []
        
        for dataset, tests in sig_results.items():
            if 'pairwise' in tests and tests['pairwise']:
                datasets_with_data.append(dataset)
                row = []
                if not comparison_labels:  # Initialize labels from first dataset
                    comparison_labels = list(tests['pairwise'].keys())
                
                for comparison in comparison_labels:
                    if comparison in tests['pairwise']:
                        row.append(tests['pairwise'][comparison]['p_value'])
                    else:
                        row.append(1.0)  # No significance
                p_value_matrix.append(row)
        
        if p_value_matrix:
            p_value_df = pd.DataFrame(p_value_matrix, 
                                     index=datasets_with_data, 
                                     columns=comparison_labels)
            
            sns.heatmap(p_value_df, annot=True, fmt='.4f', cmap='RdYlBu', 
                       ax=ax1, cbar_kws={'label': 'p-value'})
            ax1.set_title('P-values for Pairwise Comparisons', fontweight='bold')
            ax1.set_xlabel('Technique Comparison')
            ax1.set_ylabel('Dataset')
        
        # 2. Effect size visualization
        ax2 = axes[0, 1]
        effect_sizes = analysis_results.get('effect_sizes', {})
        
        effect_matrix = []
        for dataset in datasets_with_data:
            if dataset in effect_sizes:
                row = []
                for comparison in comparison_labels:
                    if comparison in effect_sizes[dataset]:
                        row.append(abs(effect_sizes[dataset][comparison]['cohens_d']))
                    else:
                        row.append(0.0)
                effect_matrix.append(row)
        
        if effect_matrix:
            effect_df = pd.DataFrame(effect_matrix,
                                   index=datasets_with_data,
                                   columns=comparison_labels)
            
            sns.heatmap(effect_df, annot=True, fmt='.3f', cmap='viridis',
                       ax=ax2, cbar_kws={'label': "Cohen's d (absolute)"})
            ax2.set_title('Effect Sizes (Cohen\'s d)', fontweight='bold')
            ax2.set_xlabel('Technique Comparison')
            ax2.set_ylabel('Dataset')
        
        # 3. Significance summary
        ax3 = axes[1, 0]
        
        # Count significant results per comparison
        significance_counts = {}
        for comparison in comparison_labels:
            count = 0
            for dataset in datasets_with_data:
                if (dataset in sig_results and 
                    'pairwise' in sig_results[dataset] and
                    comparison in sig_results[dataset]['pairwise'] and
                    sig_results[dataset]['pairwise'][comparison]['significant']):
                    count += 1
            significance_counts[comparison] = count
        
        if significance_counts:
            comparisons, counts = zip(*significance_counts.items())
            bars = ax3.bar(range(len(comparisons)), counts, 
                          color='skyblue', alpha=0.7, edgecolor='navy')
            ax3.set_xlabel('Technique Comparison')
            ax3.set_ylabel('Number of Significant Results')
            ax3.set_title('Significant Differences Count (p < 0.05)', fontweight='bold')
            ax3.set_xticks(range(len(comparisons)))
            ax3.set_xticklabels(comparisons, rotation=45, ha='right')
            ax3.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, count in zip(bars, counts):
                ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom', fontweight='bold')
        
        # 4. Confidence intervals plot
        ax4 = axes[1, 1]
        ci_data = analysis_results.get('confidence_intervals', {})
        
        if ci_data:
            # Select a representative dataset for CI visualization
            representative_dataset = 'CrowsPairs'  # or choose dynamically
            
            techniques_ci = []
            means = []
            ci_lowers = []
            ci_uppers = []
            
            for technique in self.techniques:
                if (technique in ci_data and 
                    representative_dataset in ci_data[technique]):
                    
                    ci_info = ci_data[technique][representative_dataset]
                    techniques_ci.append(technique)
                    means.append(ci_info['mean'])
                    ci_lowers.append(ci_info['ci_lower'])
                    ci_uppers.append(ci_info['ci_upper'])
            
            if techniques_ci:
                x_pos = np.arange(len(techniques_ci))
                
                # Plot means with error bars
                ax4.errorbar(x_pos, means, 
                           yerr=[np.array(means) - np.array(ci_lowers),
                                 np.array(ci_uppers) - np.array(means)],
                           fmt='o', capsize=5, capthick=2, markersize=8,
                           color='darkred', ecolor='red', alpha=0.7)
                
                ax4.set_xlabel('Technique')
                ax4.set_ylabel('Bias Score')
                ax4.set_title(f'95% Confidence Intervals - {representative_dataset}', fontweight='bold')
                ax4.set_xticks(x_pos)
                ax4.set_xticklabels(techniques_ci, rotation=45)
                ax4.grid(True, alpha=0.3)
                
                # Add mean value labels
                for i, (tech, mean_val) in enumerate(zip(techniques_ci, means)):
                    ax4.text(i, mean_val + 0.02, f'{mean_val:.3f}',
                            ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 Statistical significance plot saved to: {save_path}")
        
        return fig
    
    def create_interactive_dashboard(self, df: pd.DataFrame, save_path: str = None) -> go.Figure:
        """
        Create interactive Plotly dashboard for bias evaluation results.
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Technique Performance Overview', 'Dataset-wise Comparison',
                          'Score Distribution', 'Technique Radar Chart'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "polar"}]]
        )
        
        # 1. Technique performance overview (bar chart)
        technique_means = df.groupby('technique')['score'].agg(['mean', 'std']).reset_index()
        
        for i, technique in enumerate(technique_means['technique']):
            fig.add_trace(
                go.Bar(
                    x=[technique],
                    y=[technique_means.iloc[i]['mean']],
                    error_y=dict(type='data', array=[technique_means.iloc[i]['std']]),
                    name=technique,
                    marker_color=self.technique_colors.get(technique, 'gray'),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # 2. Dataset-wise comparison (line plot)
        for technique in self.techniques:
            tech_data = df[df['technique'] == technique]
            dataset_means = tech_data.groupby('dataset')['score'].mean()
            
            fig.add_trace(
                go.Scatter(
                    x=dataset_means.index,
                    y=dataset_means.values,
                    mode='lines+markers',
                    name=technique,
                    line=dict(color=self.technique_colors.get(technique, 'gray')),
                    showlegend=True
                ),
                row=1, col=2
            )
        
        # 3. Score distribution (violin plot)
        for technique in self.techniques:
            tech_scores = df[df['technique'] == technique]['score']
            
            fig.add_trace(
                go.Violin(
                    y=tech_scores,
                    name=technique,
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=self.technique_colors.get(technique, 'gray'),
                    opacity=0.6,
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 4. Radar chart for technique profiles
        technique_profiles = df.groupby('technique')['score'].agg(['mean', 'std', 'min', 'max'])
        
        categories = ['Mean Score', 'Consistency (1/std)', 'Min Score', 'Max Score']
        
        for technique in self.techniques:
            if technique in technique_profiles.index:
                profile = technique_profiles.loc[technique]
                
                # Normalize values for radar chart
                values = [
                    profile['mean'],
                    1/profile['std'] if profile['std'] > 0 else 1.0,  # Higher is better for consistency
                    profile['min'],
                    profile['max']
                ]
                
                fig.add_trace(
                    go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=technique,
                        line=dict(color=self.technique_colors.get(technique, 'gray')),
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title_text="FIRM Framework: Interactive Analysis Dashboard",
            title_x=0.5,
            title_font_size=16,
            height=800,
            showlegend=True
        )
        
        # Update subplot titles and axes
        fig.update_xaxes(title_text="Technique", row=1, col=1)
        fig.update_yaxes(title_text="Mean Bias Score", row=1, col=1)
        
        fig.update_xaxes(title_text="Dataset", row=1, col=2)
        fig.update_yaxes(title_text="Bias Score", row=1, col=2)
        
        fig.update_xaxes(title_text="Technique", row=2, col=1)
        fig.update_yaxes(title_text="Bias Score", row=2, col=1)
        
        if save_path:
            fig.write_html(save_path)
            print(f"📊 Interactive dashboard saved to: {save_path}")
        
        return fig
    
    def create_publication_ready_plot(self, df: pd.DataFrame, save_path: str = None) -> plt.Figure:
        """
        Create publication-ready figure with proper formatting.
        """
        # Set publication style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Calculate means and confidence intervals
        technique_stats = []
        for technique in self.techniques:
            tech_data = df[df['technique'] == technique]['score']
            if len(tech_data) > 0:
                mean_score = tech_data.mean()
                std_score = tech_data.std()
                n = len(tech_data)
                
                # 95% confidence interval
                ci = 1.96 * (std_score / np.sqrt(n)) if n > 1 else 0
                
                technique_stats.append({
                    'technique': technique,
                    'mean': mean_score,
                    'ci': ci,
                    'n': n
                })
        
        if technique_stats:
            techniques = [t['technique'] for t in technique_stats]
            means = [t['mean'] for t in technique_stats]
            cis = [t['ci'] for t in technique_stats]
            
            # Create bar plot with error bars
            bars = ax.bar(techniques, means, yerr=cis, capsize=5,
                         color=[self.technique_colors[t] for t in techniques],
                         alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Customize appearance
            ax.set_ylabel('Bias Score (Lower = Better)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Bias Mitigation Technique', fontsize=12, fontweight='bold')
            ax.set_title('FIRM Framework: Bias Mitigation Performance Comparison', 
                        fontsize=14, fontweight='bold', pad=20)
            
            # Add value labels on bars
            for bar, mean_val, ci_val in zip(bars, means, cis):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + ci_val + 0.01,
                       f'{mean_val:.3f}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=10)
            
            # Customize grid and spines
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add technique descriptions
            descriptions = {
                'baseline': 'No Intervention',
                'fairsteer': 'Steering Vectors',
                'sycophancy': 'Path Patching',
                'firm': 'FIRM (5-Phase)'
            }
            
            # Add legend with descriptions
            legend_elements = []
            for i, technique in enumerate(techniques):
                legend_elements.append(
                    Rectangle((0, 0), 1, 1, 
                             facecolor=self.technique_colors[technique],
                             label=f'{technique.title()}: {descriptions.get(technique, "")}')
                )
            
            ax.legend(handles=legend_elements, loc='upper right', 
                     frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"📰 Publication-ready plot saved to: {save_path}")
        
        return fig

def main():
    """Example usage of visualization tools."""
    
    # Initialize visualization tools
    viz = FIRMVisualizationTools()
    
    # Example data file
    results_file = "evaluation_results.json"
    
    if Path(results_file).exists():
        # Load data
        df = viz.load_data(results_file)
        print(f"📊 Loaded {len(df)} data points for visualization")
        
        # Create visualizations
        print("\n🎨 Creating visualizations...")
        
        # 1. Comprehensive comparison
        viz.create_technique_comparison_plot(df, "technique_comparison.png")
        
        # 2. Publication-ready plot
        viz.create_publication_ready_plot(df, "publication_figure.png")
        
        # 3. Interactive dashboard
        viz.create_interactive_dashboard(df, "interactive_dashboard.html")
        
        print("✅ All visualizations created successfully!")
        
    else:
        print(f"❌ Results file not found: {results_file}")
        print("Please provide evaluation results to create visualizations.")

if __name__ == "__main__":
    main()