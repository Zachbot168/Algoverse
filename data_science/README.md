# FIRM Data Science and Statistical Analysis Suite

A comprehensive collection of data science and statistical analysis tools for the FIRM (Fairness Interventions at Runtime and Model-training) bias mitigation research framework.

## Overview

This data science suite provides advanced statistical analysis, visualization, experimental design, and results interpretation tools specifically designed for bias mitigation research. All tools are built to handle the complexity of comparing multiple techniques (baseline, FairSteer, sycophancy, FIRM) across multiple bias evaluation datasets.

## Tools Included

### 1. Statistical Analyzer (`statistical_analyzer.py`)

Comprehensive statistical analysis including:
- **Significance Testing**: t-tests, Mann-Whitney U, ANOVA, Kruskal-Wallis
- **Effect Size Calculation**: Cohen's d with interpretation
- **Confidence Intervals**: 95% CI for all technique-dataset combinations
- **Multiple Comparisons**: Bonferroni, Holm, FDR corrections
- **Power Analysis**: Statistical power computation and recommendations

### 2. Visualization Tools (`visualization_tools.py`)

Advanced plotting and visualization including:
- **Comparative Plots**: Box plots, heatmaps, bar charts with error bars
- **Statistical Visualizations**: P-value heatmaps, effect size plots, confidence intervals
- **Interactive Dashboards**: Plotly-based interactive analysis dashboards
- **Publication-Ready Figures**: High-quality plots for papers and presentations
- **Technique Comparison**: Head-to-head performance visualizations

### 3. Experimental Design (`experimental_design.py`)

Research methodology and experimental design tools:
- **Sample Size Calculation**: Required n for different effect sizes and power levels
- **Power Analysis**: Statistical power given constraints
- **Effect Size Estimation**: Minimum detectable effects
- **Multiple Comparisons Planning**: Family-wise error rate control
- **Experimental Design Reports**: Comprehensive methodology documentation

### 4. Results Analyzer (`results_analyzer.py`)

Advanced analysis and research insights:
- **Technique Effectiveness**: Comprehensive performance metrics
- **Clustering Analysis**: Pattern identification across techniques and datasets
- **Improvement Patterns**: Detailed baseline comparison analysis
- **Research Insights**: Automated insight generation and recommendations
- **Comprehensive Reports**: Full analysis with actionable conclusions

## Quick Start

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn plotly
```

### Basic Usage Examples

#### 1. Statistical Analysis

```python
from statistical_analyzer import FIRMStatisticalAnalyzer

analyzer = FIRMStatisticalAnalyzer()
results = analyzer.run_complete_analysis("evaluation_results.json")
analyzer.save_analysis_results(results, "statistical_analysis.json")
```

#### 2. Create Visualizations

```python
from visualization_tools import FIRMVisualizationTools

viz = FIRMVisualizationTools()
df = viz.load_data("evaluation_results.json")
viz.create_technique_comparison_plot(df, "comparison.png")
viz.create_publication_ready_plot(df, "publication_figure.png")
viz.create_interactive_dashboard(df, "dashboard.html")
```

#### 3. Experimental Design Planning

```python
from experimental_design import ExperimentalDesignAnalyzer

design = ExperimentalDesignAnalyzer()
recommendations = design.recommend_experimental_design(
    expected_effect_size=0.5,
    desired_power=0.8,
    alpha=0.05
)
design.create_power_analysis_curves("power_curves.png")
design.generate_experimental_report(recommendations, "design_report.md")
```

#### 4. Comprehensive Results Analysis

```python
from results_analyzer import FIRMResultsAnalyzer

analyzer = FIRMResultsAnalyzer()
report = analyzer.create_comprehensive_analysis_report(
    "evaluation_results.json",
    "analysis_report.json"
)

for insight in report['research_insights']:
    print(f"- {insight}")
```

## Input Data Format

All tools expect evaluation results in JSON format:

```json
{
  "baseline": [
    {
      "model": "qwen2.5-3b-instruct",
      "seed": 42,
      "dataset_scores": {
        "CrowsPairs": 0.73,
        "StereoSet": 0.57,
        "WinoBias": 0.73,
        "BBQ": 0.44,
        "TruthfulQA": 0.96
      }
    }
  ],
  "fairsteer": [...],
  "sycophancy": [...],
  "firm": [...]
}
```

## Statistical Methodology

### Significance Testing
- **Parametric Tests**: t-tests for normally distributed data with equal variances
- **Non-parametric Tests**: Mann-Whitney U and Wilcoxon for non-normal data
- **Multiple Groups**: ANOVA or Kruskal-Wallis for comparing >2 techniques
- **Multiple Comparisons**: Holm-Bonferroni or FDR correction for family-wise error control

### Effect Size Interpretation
- **Cohen's d < 0.2**: Negligible effect
- **0.2 <= d < 0.5**: Small effect (still practically meaningful for bias reduction)
- **0.5 <= d < 0.8**: Medium effect (substantial bias reduction)
- **d >= 0.8**: Large effect (strong bias mitigation)

### Power Analysis Guidelines
- **Minimum Power**: 0.8 (80% chance of detecting true effects)
- **Preferred Power**: 0.9 (90% for important comparisons)
- **Sample Size**: Calculate based on smallest effect size of practical interest
- **Multiple Comparisons**: Adjust power calculations for planned comparisons

## Output Files

- **`statistical_analysis.json`**: Complete statistical analysis results
- **`technique_comparison.png`**: Comprehensive technique comparison plot
- **`publication_figure.png`**: Publication-ready figure
- **`interactive_dashboard.html`**: Interactive analysis dashboard
- **`power_analysis_curves.png`**: Power analysis visualizations
- **`experimental_design_report.md`**: Methodology documentation
- **`comprehensive_analysis_report.json`**: Full results analysis with insights

## Research Applications

### Bias Mitigation Evaluation
- Compare effectiveness of different bias reduction techniques
- Identify which approaches work best for specific bias types
- Quantify improvement over baseline approaches
- Assess consistency across different evaluation datasets

### Method Development
- Guide development of new bias mitigation techniques
- Identify areas where current methods are insufficient
- Optimize hyperparameters based on performance patterns
- Validate new approaches with proper statistical rigor

### Publication and Reporting
- Generate publication-ready figures and tables
- Provide comprehensive statistical analysis for peer review
- Document experimental methodology and design decisions
- Create reproducible analysis workflows

## Best Practices

### Data Collection
- Use multiple random seeds for robust evaluation
- Include baseline comparisons for all techniques
- Evaluate on diverse bias types and datasets
- Document all experimental parameters

### Statistical Analysis
- Check assumptions before applying parametric tests
- Report effect sizes with confidence intervals
- Apply appropriate multiple comparisons corrections
- Use equivalence testing for non-significant results

### Visualization
- Include error bars or confidence intervals on all plots
- Use consistent color schemes across figures
- Provide clear axis labels and legends
- Create both overview and detailed plots

### Interpretation
- Focus on practical significance, not just statistical significance
- Consider consistency across datasets and bias types
- Acknowledge limitations and potential confounds
- Provide actionable recommendations for future research
