"""
Experimental Design and Power Analysis for FIRM Bias Mitigation Research

This module provides tools for experimental design, power analysis, sample size calculation,
and statistical methodology selection for bias evaluation studies.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind, ttest_rel, mannwhitneyu, wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ExperimentalDesignAnalyzer:
    """
    Tools for experimental design and power analysis in bias mitigation research.
    
    Provides sample size calculation, power analysis, effect size estimation,
    and experimental design recommendations.
    """
    
    def __init__(self):
        """Initialize experimental design analyzer."""
        self.alpha_levels = [0.05, 0.01, 0.001]
        self.power_levels = [0.8, 0.9, 0.95]
        self.effect_sizes = {
            'small': 0.2,
            'medium': 0.5,
            'large': 0.8,
            'very_large': 1.2
        }
    
    def calculate_sample_size_ttest(self, effect_size: float, alpha: float = 0.05, 
                                   power: float = 0.8, two_sided: bool = True) -> int:
        """
        Calculate required sample size for t-test.
        
        Args:
            effect_size: Expected Cohen's d effect size
            alpha: Type I error rate (default 0.05)
            power: Desired statistical power (default 0.8)
            two_sided: Whether to use two-sided test (default True)
            
        Returns:
            Required sample size per group
        """
        # Z-scores for alpha and beta
        if two_sided:
            z_alpha = stats.norm.ppf(1 - alpha/2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)
        
        z_beta = stats.norm.ppf(power)
        
        # Sample size calculation
        n = ((z_alpha + z_beta) / effect_size) ** 2 * 2
        
        return max(3, int(np.ceil(n)))  # Minimum of 3 per group
    
    def calculate_power_analysis(self, n: int, effect_size: float, 
                                alpha: float = 0.05, two_sided: bool = True) -> float:
        """
        Calculate statistical power given sample size and effect size.
        
        Args:
            n: Sample size per group
            effect_size: Cohen's d effect size
            alpha: Type I error rate
            two_sided: Whether to use two-sided test
            
        Returns:
            Statistical power (0-1)
        """
        if two_sided:
            z_alpha = stats.norm.ppf(1 - alpha/2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)
        
        # Calculate z-score for beta
        z_beta = effect_size * np.sqrt(n/2) - z_alpha
        
        # Power is 1 - beta
        power = stats.norm.cdf(z_beta)
        
        return max(0, min(1, power))
    
    def calculate_minimum_detectable_effect(self, n: int, alpha: float = 0.05, 
                                          power: float = 0.8, two_sided: bool = True) -> float:
        """
        Calculate minimum detectable effect size given sample size and power.
        
        Args:
            n: Sample size per group
            alpha: Type I error rate
            power: Desired statistical power
            two_sided: Whether to use two-sided test
            
        Returns:
            Minimum detectable effect size (Cohen's d)
        """
        if two_sided:
            z_alpha = stats.norm.ppf(1 - alpha/2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)
        
        z_beta = stats.norm.ppf(power)
        
        # Minimum detectable effect
        mde = (z_alpha + z_beta) * np.sqrt(2/n)
        
        return mde
    
    def create_power_analysis_curves(self, save_path: str = None) -> plt.Figure:
        """
        Create power analysis curves for different effect sizes.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Power Analysis for FIRM Bias Mitigation Studies', fontsize=16, fontweight='bold')
        
        sample_sizes = np.arange(5, 101, 5)
        
        # 1. Power curves for different effect sizes
        ax1 = axes[0, 0]
        for effect_name, effect_size in self.effect_sizes.items():
            powers = [self.calculate_power_analysis(n, effect_size) for n in sample_sizes]
            ax1.plot(sample_sizes, powers, label=f'{effect_name.title()} (d={effect_size})', 
                    linewidth=2, marker='o', markersize=4)
        
        ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Power = 0.8')
        ax1.set_xlabel('Sample Size per Group')
        ax1.set_ylabel('Statistical Power')
        ax1.set_title('Power vs Sample Size by Effect Size', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # 2. Sample size requirements for different power levels
        ax2 = axes[0, 1]
        effect_sizes_range = np.linspace(0.1, 1.5, 50)
        
        for power in self.power_levels:
            sample_sizes_req = [self.calculate_sample_size_ttest(es, power=power) 
                               for es in effect_sizes_range]
            ax2.plot(effect_sizes_range, sample_sizes_req, 
                    label=f'Power = {power}', linewidth=2, marker='o', markersize=3)
        
        ax2.set_xlabel('Effect Size (Cohen\'s d)')
        ax2.set_ylabel('Required Sample Size per Group')
        ax2.set_title('Sample Size Requirements', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 200)
        
        # 3. Minimum detectable effect
        ax3 = axes[1, 0]
        for power in self.power_levels:
            mdes = [self.calculate_minimum_detectable_effect(n, power=power) 
                   for n in sample_sizes]
            ax3.plot(sample_sizes, mdes, 
                    label=f'Power = {power}', linewidth=2, marker='o', markersize=4)
        
        # Add effect size interpretation lines
        for effect_name, effect_size in self.effect_sizes.items():
            if effect_size <= 1.0:  # Only show reasonable effect sizes
                ax3.axhline(y=effect_size, linestyle=':', alpha=0.5, 
                           label=f'{effect_name.title()} effect')
        
        ax3.set_xlabel('Sample Size per Group')
        ax3.set_ylabel('Minimum Detectable Effect (Cohen\'s d)')
        ax3.set_title('Minimum Detectable Effect Size', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.5)
        
        # 4. Type I and Type II error visualization
        ax4 = axes[1, 1]
        alphas = np.array(self.alpha_levels)
        
        # Sample sizes for different alpha levels (medium effect size)
        medium_effect = self.effect_sizes['medium']
        sample_sizes_alpha = [self.calculate_sample_size_ttest(medium_effect, alpha=alpha) 
                             for alpha in alphas]
        
        x_pos = np.arange(len(alphas))
        bars = ax4.bar(x_pos, sample_sizes_alpha, alpha=0.7, color='skyblue', edgecolor='navy')
        
        ax4.set_xlabel('Alpha Level (Type I Error Rate)')
        ax4.set_ylabel('Required Sample Size')
        ax4.set_title(f'Sample Size vs Alpha Level\n(Medium Effect Size, Power=0.8)', 
                     fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f'{a:.3f}' for a in alphas])
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, size in zip(bars, sample_sizes_alpha):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{size}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Power analysis curves saved to: {save_path}")
        
        return fig
    
    def recommend_experimental_design(self, expected_effect_size: float = None,
                                     available_samples: int = None,
                                     desired_power: float = 0.8,
                                     alpha: float = 0.05) -> Dict:
        """
        Provide experimental design recommendations based on constraints.
        
        Args:
            expected_effect_size: Expected Cohen's d (if known)
            available_samples: Maximum available samples per group (if constrained)
            desired_power: Target statistical power
            alpha: Type I error rate
            
        Returns:
            Dictionary with recommendations
        """
        recommendations = {
            'design_type': 'between_subjects',  # Default for bias mitigation
            'alpha': alpha,
            'power_target': desired_power,
            'recommendations': []
        }
        
        # Case 1: Effect size known, optimize sample size
        if expected_effect_size is not None:
            required_n = self.calculate_sample_size_ttest(
                expected_effect_size, alpha=alpha, power=desired_power
            )
            
            recommendations['expected_effect_size'] = expected_effect_size
            recommendations['required_sample_size'] = required_n
            
            if available_samples is not None:
                if available_samples >= required_n:
                    recommendations['feasible'] = True
                    recommendations['recommendations'].append(
                        f"✅ Design is feasible with {available_samples} samples per group"
                    )
                else:
                    actual_power = self.calculate_power_analysis(
                        available_samples, expected_effect_size, alpha
                    )
                    recommendations['feasible'] = False
                    recommendations['actual_power'] = actual_power
                    recommendations['recommendations'].append(
                        f"⚠️ Only {actual_power:.2f} power with {available_samples} samples"
                    )
                    recommendations['recommendations'].append(
                        f"💡 Consider reducing alpha to {alpha/2:.3f} or accepting lower power"
                    )
            
            # Effect size interpretation
            if expected_effect_size < 0.2:
                recommendations['recommendations'].append(
                    "📏 Very small effect size - consider if practically meaningful"
                )
            elif expected_effect_size < 0.5:
                recommendations['recommendations'].append(
                    "📏 Small effect size - adequate for bias research"
                )
            elif expected_effect_size < 0.8:
                recommendations['recommendations'].append(
                    "📏 Medium effect size - good for demonstrating bias reduction"
                )
            else:
                recommendations['recommendations'].append(
                    "📏 Large effect size - strong bias mitigation expected"
                )
        
        # Case 2: Sample size constrained, determine detectable effect
        elif available_samples is not None:
            mde = self.calculate_minimum_detectable_effect(
                available_samples, alpha=alpha, power=desired_power
            )
            
            recommendations['available_samples'] = available_samples
            recommendations['minimum_detectable_effect'] = mde
            
            recommendations['recommendations'].append(
                f"🔍 Can detect effect sizes ≥ {mde:.3f} with {desired_power} power"
            )
            
            if mde > 0.8:
                recommendations['recommendations'].append(
                    "⚠️ Can only detect large effects - consider increasing sample size"
                )
            elif mde > 0.5:
                recommendations['recommendations'].append(
                    "📊 Can detect medium-to-large effects - reasonable for bias research"
                )
            else:
                recommendations['recommendations'].append(
                    "✅ Can detect small-to-medium effects - good sensitivity"
                )
        
        # General recommendations
        recommendations['recommendations'].extend([
            f"📋 Use {alpha} significance level (adjust for multiple comparisons)",
            f"🎯 Target {desired_power} statistical power",
            "🔄 Consider within-subjects design if possible (more powerful)",
            "📊 Plan for effect size estimation with confidence intervals",
            "🧪 Include manipulation checks for bias interventions",
            "📈 Pre-register analysis plan to avoid p-hacking"
        ])
        
        return recommendations
    
    def multiple_comparisons_adjustment(self, p_values: List[float], 
                                      method: str = 'bonferroni') -> Dict:
        """
        Apply multiple comparisons corrections.
        
        Args:
            p_values: List of p-values from multiple tests
            method: Correction method ('bonferroni', 'holm', 'fdr_bh')
            
        Returns:
            Dictionary with original and adjusted p-values
        """
        p_values = np.array(p_values)
        n_tests = len(p_values)
        
        if method == 'bonferroni':
            adjusted_p = p_values * n_tests
            adjusted_p = np.minimum(adjusted_p, 1.0)  # Cap at 1
            
        elif method == 'holm':
            # Holm-Bonferroni step-down method
            sorted_indices = np.argsort(p_values)
            adjusted_p = np.zeros_like(p_values)
            
            for i, idx in enumerate(sorted_indices):
                adjusted_p[idx] = p_values[idx] * (n_tests - i)
                if i > 0:
                    adjusted_p[idx] = max(adjusted_p[idx], 
                                        adjusted_p[sorted_indices[i-1]])
            
            adjusted_p = np.minimum(adjusted_p, 1.0)
            
        elif method == 'fdr_bh':
            # Benjamini-Hochberg FDR control
            sorted_indices = np.argsort(p_values)
            adjusted_p = np.zeros_like(p_values)
            
            for i in range(n_tests-1, -1, -1):
                idx = sorted_indices[i]
                adjusted_p[idx] = p_values[idx] * n_tests / (i + 1)
                if i < n_tests - 1:
                    adjusted_p[idx] = min(adjusted_p[idx], 
                                        adjusted_p[sorted_indices[i+1]])
            
            adjusted_p = np.minimum(adjusted_p, 1.0)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            'original_p_values': p_values.tolist(),
            'adjusted_p_values': adjusted_p.tolist(),
            'method': method,
            'n_tests': n_tests,
            'significant_original': (p_values < 0.05).sum(),
            'significant_adjusted': (adjusted_p < 0.05).sum()
        }
    
    def generate_experimental_report(self, design_params: Dict, save_path: str = None) -> str:
        """
        Generate a comprehensive experimental design report.
        
        Args:
            design_params: Dictionary with experimental parameters
            save_path: Optional path to save the report
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("# FIRM Bias Mitigation: Experimental Design Report")
        report.append("=" * 60)
        report.append("")
        
        # Study overview
        report.append("## Study Overview")
        report.append(f"- Research Question: Effectiveness of FIRM bias mitigation techniques")
        report.append(f"- Design Type: {design_params.get('design_type', 'Between-subjects')}")
        report.append(f"- Primary Analysis: Statistical comparison of bias scores")
        report.append("")
        
        # Statistical parameters
        report.append("## Statistical Parameters")
        report.append(f"- Significance Level (α): {design_params.get('alpha', 0.05)}")
        report.append(f"- Target Power (1-β): {design_params.get('power_target', 0.8)}")
        
        if 'expected_effect_size' in design_params:
            report.append(f"- Expected Effect Size (Cohen's d): {design_params['expected_effect_size']}")
        
        if 'required_sample_size' in design_params:
            report.append(f"- Required Sample Size per Group: {design_params['required_sample_size']}")
        
        if 'minimum_detectable_effect' in design_params:
            report.append(f"- Minimum Detectable Effect: {design_params['minimum_detectable_effect']:.3f}")
        
        report.append("")
        
        # Recommendations
        if 'recommendations' in design_params:
            report.append("## Recommendations")
            for rec in design_params['recommendations']:
                report.append(f"- {rec}")
            report.append("")
        
        # Multiple comparisons
        report.append("## Multiple Comparisons Strategy")
        report.append("- Primary Comparisons: FIRM vs Baseline, FairSteer vs Baseline")
        report.append("- Secondary Comparisons: All pairwise technique comparisons")
        report.append("- Correction Method: Holm-Bonferroni for family-wise error control")
        report.append("- Dataset-wise Analysis: Separate correction within each dataset")
        report.append("")
        
        # Data collection
        report.append("## Data Collection Protocol")
        report.append("- Evaluation Datasets: 10 bias benchmarks (CrowsPairs, StereoSet, etc.)")
        report.append("- Randomization: Random seed assignment for reproducibility")
        report.append("- Blinding: Automated evaluation (no human rater bias)")
        report.append("- Quality Control: Validation of model loading and dataset integrity")
        report.append("")
        
        # Analysis plan
        report.append("## Statistical Analysis Plan")
        report.append("### Primary Analysis")
        report.append("1. Descriptive statistics for each technique-dataset combination")
        report.append("2. Normality testing (Shapiro-Wilk for n<50, visual inspection)")
        report.append("3. Homogeneity of variance testing (Levene's test)")
        report.append("4. Primary hypothesis testing:")
        report.append("   - Parametric: Paired t-tests (if assumptions met)")
        report.append("   - Non-parametric: Wilcoxon signed-rank tests (if assumptions violated)")
        report.append("")
        
        report.append("### Secondary Analysis")
        report.append("1. Effect size calculation (Cohen's d) with confidence intervals")
        report.append("2. Equivalence testing for non-significant results")
        report.append("3. Bayesian analysis for evidence quantification")
        report.append("4. Sensitivity analysis for outliers")
        report.append("")
        
        # Interpretation guidelines
        report.append("## Effect Size Interpretation")
        report.append("- |d| < 0.2: Negligible effect")
        report.append("- 0.2 ≤ |d| < 0.5: Small effect")
        report.append("- 0.5 ≤ |d| < 0.8: Medium effect")
        report.append("- |d| ≥ 0.8: Large effect")
        report.append("")
        
        report.append("## Bias Score Interpretation")
        report.append("- CrowsPairs: Higher = less bias (anti-stereotypical preference)")
        report.append("- StereoSet: Lower = less bias (less stereotypical completion)")
        report.append("- WinoBias/WinoGender: Higher = better (accuracy without gender bias)")
        report.append("- BBQ: Higher = better (unbiased question answering)")
        report.append("- TruthfulQA: Higher = better (truthfulness vs sycophancy)")
        report.append("")
        
        # Reporting standards
        report.append("## Reporting Standards")
        report.append("- Follow APA guidelines for statistical reporting")
        report.append("- Report exact p-values (not just p < 0.05)")
        report.append("- Include effect sizes with confidence intervals")
        report.append("- Report assumption checking results")
        report.append("- Provide raw data and analysis code for reproducibility")
        report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"📋 Experimental design report saved to: {save_path}")
        
        return report_text

def main():
    """Example usage of experimental design tools."""
    
    # Initialize analyzer
    analyzer = ExperimentalDesignAnalyzer()
    
    print("🧪 FIRM Experimental Design Analysis")
    print("=" * 40)
    
    # Example 1: Known effect size, determine sample size
    print("\n📊 Scenario 1: Expected medium effect size")
    effect_size = 0.5
    recommendations = analyzer.recommend_experimental_design(
        expected_effect_size=effect_size,
        desired_power=0.8,
        alpha=0.05
    )
    
    print(f"Expected effect size: {effect_size}")
    print(f"Required sample size: {recommendations['required_sample_size']}")
    for rec in recommendations['recommendations'][:3]:
        print(f"  {rec}")
    
    # Example 2: Constrained sample size
    print("\n📊 Scenario 2: Limited to 20 samples per group")
    available_n = 20
    recommendations = analyzer.recommend_experimental_design(
        available_samples=available_n,
        desired_power=0.8,
        alpha=0.05
    )
    
    print(f"Available samples: {available_n}")
    print(f"Minimum detectable effect: {recommendations['minimum_detectable_effect']:.3f}")
    for rec in recommendations['recommendations'][:3]:
        print(f"  {rec}")
    
    # Example 3: Multiple comparisons
    print("\n🔬 Multiple Comparisons Example")
    p_values = [0.001, 0.02, 0.03, 0.08, 0.15]
    corrections = analyzer.multiple_comparisons_adjustment(p_values, method='holm')
    
    print("Original p-values:", [f"{p:.3f}" for p in corrections['original_p_values']])
    print("Holm-adjusted p-values:", [f"{p:.3f}" for p in corrections['adjusted_p_values']])
    print(f"Significant before correction: {corrections['significant_original']}")
    print(f"Significant after correction: {corrections['significant_adjusted']}")
    
    # Create power analysis plots
    print("\n📈 Creating power analysis visualizations...")
    analyzer.create_power_analysis_curves("power_analysis_curves.png")
    
    # Generate experimental report
    print("\n📋 Generating experimental design report...")
    design_params = {
        'design_type': 'Between-subjects comparison',
        'alpha': 0.05,
        'power_target': 0.8,
        'expected_effect_size': 0.5,
        'required_sample_size': 32,
        'recommendations': [
            "Use adequate sample size for medium effect detection",
            "Apply multiple comparisons correction",
            "Include effect size reporting with confidence intervals"
        ]
    }
    
    report = analyzer.generate_experimental_report(design_params, "experimental_design_report.md")
    
    print("✅ Experimental design analysis complete!")

if __name__ == "__main__":
    main()