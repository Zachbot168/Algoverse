# FIRM Data Science & Statistical Analysis Suite

A comprehensive collection of data science and statistical analysis tools for the FIRM (Fairness Interventions at Runtime and Model-training) bias mitigation research framework.

## 📊 **Overview**

This data science suite provides advanced statistical analysis, visualization, experimental design, and results interpretation tools specifically designed for bias mitigation research. All tools are built to handle the complexity of comparing multiple techniques (baseline, FairSteer, sycophancy, FIRM) across multiple bias evaluation datasets.

## 🛠️ **Tools Included**

### 1. **Statistical Analyzer** (`statistical_analyzer.py`)
Comprehensive statistical analysis including:
- **Significance Testing**: t-tests, Mann-Whitney U, ANOVA, Kruskal-Wallis
- **Effect Size Calculation**: Cohen's d with interpretation
- **Confidence Intervals**: 95% CI for all technique-dataset combinations  
- **Multiple Comparisons**: Bonferroni, Holm, FDR corrections
- **Power Analysis**: Statistical power computation and recommendations

### 2. **Visualization Tools** (`visualization_tools.py`)
Advanced plotting and visualization including:
- **Comparative Plots**: Box plots, heatmaps, bar charts with error bars
- **Statistical Visualizations**: P-value heatmaps, effect size plots, confidence intervals
- **Interactive Dashboards**: Plotly-based interactive analysis dashboards
- **Publication-Ready Figures**: High-quality plots for papers and presentations
- **Technique Comparison**: Head-to-head performance visualizations

### 3. **Experimental Design** (`experimental_design.py`)
Research methodology and experimental design tools:
- **Sample Size Calculation**: Required n for different effect sizes and power levels
- **Power Analysis**: Statistical power given constraints
- **Effect Size Estimation**: Minimum detectable effects
- **Multiple Comparisons Planning**: Family-wise error rate control
- **Experimental Design Reports**: Comprehensive methodology documentation

### 4. **Results Analyzer** (`results_analyzer.py`)
Advanced analysis and research insights:
- **Technique Effectiveness**: Comprehensive performance metrics
- **Clustering Analysis**: Pattern identification across techniques and datasets
- **Improvement Patterns**: Detailed baseline comparison analysis  
- **Research Insights**: Automated insight generation and recommendations
- **Comprehensive Reports**: Full analysis with actionable conclusions

## 🚀 **Quick Start**

### **Prerequisites**

```bash
# Install required packages
pip install numpy pandas matplotlib seaborn scipy scikit-learn plotly
```

### **Basic Usage Examples**

#### 1. **Statistical Analysis**
```python
from statistical_analyzer import FIRMStatisticalAnalyzer

# Initialize analyzer
analyzer = FIRMStatisticalAnalyzer()

# Run complete analysis
results = analyzer.run_complete_analysis("evaluation_results.json")

# Save results
analyzer.save_analysis_results(results, "statistical_analysis.json")
```

#### 2. **Create Visualizations**
```python
from visualization_tools import FIRMVisualizationTools

# Initialize visualization tools
viz = FIRMVisualizationTools()

# Load data and create plots
df = viz.load_data("evaluation_results.json")

# Create comprehensive comparison plot
viz.create_technique_comparison_plot(df, "comparison.png")

# Create publication-ready figure
viz.create_publication_ready_plot(df, "publication_figure.png")

# Create interactive dashboard
viz.create_interactive_dashboard(df, "dashboard.html")
```

#### 3. **Experimental Design Planning**
```python
from experimental_design import ExperimentalDesignAnalyzer

# Initialize design analyzer  
design = ExperimentalDesignAnalyzer()

# Get recommendations for expected medium effect
recommendations = design.recommend_experimental_design(
    expected_effect_size=0.5,
    desired_power=0.8,
    alpha=0.05
)

# Create power analysis curves
design.create_power_analysis_curves("power_curves.png")

# Generate experimental report
design.generate_experimental_report(recommendations, "design_report.md")
```

#### 4. **Comprehensive Results Analysis**
```python
from results_analyzer import FIRMResultsAnalyzer

# Initialize analyzer
analyzer = FIRMResultsAnalyzer()

# Generate comprehensive report with insights
report = analyzer.create_comprehensive_analysis_report(
    "evaluation_results.json",
    "analysis_report.json"
)

# Access key findings
print("Research Insights:")
for insight in report['research_insights']:
    print(f"- {insight}")
```

## 📋 **Detailed Usage Guide**

### **Input Data Format**

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

### **Statistical Analysis Workflow**

#### **Step 1: Load and Analyze Data**
```python
analyzer = FIRMStatisticalAnalyzer()
results = analyzer.run_complete_analysis("results.json")
```

#### **Step 2: Examine Key Statistics**
```python
# Summary statistics
summary = results['summary_statistics']
print(f"Baseline mean: {summary['baseline']['CrowsPairs']['mean']:.3f}")

# Statistical significance
sig_tests = results['statistical_significance']
for dataset, tests in sig_tests.items():
    print(f"{dataset}: {len(tests['pairwise'])} comparisons")

# Effect sizes
effect_sizes = results['effect_sizes']
for dataset, comparisons in effect_sizes.items():
    for comp, effect in comparisons.items():
        if effect['interpretation'] in ['medium', 'large']:
            print(f"{dataset} {comp}: {effect['cohens_d']:.3f} ({effect['interpretation']})")
```

#### **Step 3: Interpret Results**
```python
# Confidence intervals
ci_results = results['confidence_intervals']
for technique in ['baseline', 'firm']:
    for dataset in ['CrowsPairs', 'WinoBias']:
        if technique in ci_results and dataset in ci_results[technique]:
            ci = ci_results[technique][dataset]
            print(f"{technique} {dataset}: {ci['mean']:.3f} [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")
```

### **Visualization Best Practices**

#### **1. Technique Comparison Plots**
```python
viz = FIRMVisualizationTools()
df = viz.load_data("results.json")

# Create comprehensive comparison with subplots
fig = viz.create_technique_comparison_plot(df, "comparison.png")

# Customize for specific needs
plt.suptitle("FIRM vs Competing Approaches: Bias Mitigation Effectiveness")
plt.tight_layout()
plt.savefig("custom_comparison.png", dpi=300, bbox_inches='tight')
```

#### **2. Statistical Significance Visualization**
```python
# Load statistical analysis results
with open("statistical_analysis.json", 'r') as f:
    analysis = json.load(f)

# Create significance plots
viz.create_statistical_significance_plot(analysis, "significance.png")
```

#### **3. Interactive Dashboards**
```python
# Create interactive dashboard for exploration
dashboard = viz.create_interactive_dashboard(df, "dashboard.html")

# Dashboard includes:
# - Performance overview
# - Dataset-wise comparisons  
# - Score distributions
# - Technique radar charts
```

### **Experimental Design Guidelines**

#### **1. Sample Size Calculation**
```python
design = ExperimentalDesignAnalyzer()

# For different effect sizes
for effect_name, effect_size in [('small', 0.2), ('medium', 0.5), ('large', 0.8)]:
    n_required = design.calculate_sample_size_ttest(effect_size, power=0.8)
    print(f"{effect_name.title()} effect (d={effect_size}): n={n_required} per group")

# Output:
# Small effect (d=0.2): n=393 per group
# Medium effect (d=0.5): n=64 per group  
# Large effect (d=0.8): n=26 per group
```

#### **2. Power Analysis**
```python
# Given sample constraints, what effect can we detect?
available_n = 50
mde = design.calculate_minimum_detectable_effect(available_n, power=0.8)
print(f"With n={available_n}, can detect effects ≥ {mde:.3f}")

# What power do we have for a specific effect?
power = design.calculate_power_analysis(available_n, effect_size=0.5)
print(f"Power for medium effect with n={available_n}: {power:.2f}")
```

#### **3. Multiple Comparisons**
```python
# Adjust p-values for multiple testing
p_values = [0.001, 0.023, 0.045, 0.089, 0.156]

# Bonferroni correction
bonf_results = design.multiple_comparisons_adjustment(p_values, method='bonferroni')
print(f"Significant before: {bonf_results['significant_original']}")
print(f"Significant after Bonferroni: {bonf_results['significant_adjusted']}")

# Holm correction (less conservative)
holm_results = design.multiple_comparisons_adjustment(p_values, method='holm')
print(f"Significant after Holm: {holm_results['significant_adjusted']}")
```

### **Advanced Results Analysis**

#### **1. Technique Effectiveness Analysis**
```python
analyzer = FIRMResultsAnalyzer()
df = analyzer.load_and_structure_data("results.json")

# Compute comprehensive effectiveness metrics
effectiveness = analyzer.compute_technique_effectiveness(df)

# Best techniques by category
best_techniques = analyzer.identify_best_techniques(df)

print("Best Overall:", best_techniques['overall_best']['technique'])
print("Most Consistent:", best_techniques['consistency_best']['technique'])

# Category-specific winners
for category, result in best_techniques['category_best'].items():
    print(f"Best for {category}: {result['technique']} (score: {result['score']:.3f})")
```

#### **2. Clustering and Pattern Analysis**
```python
# Identify patterns across techniques and datasets
clustering = analyzer.perform_clustering_analysis(df)

# Dataset clusters (similar bias patterns)
for dataset, cluster in clustering['dataset_clusters'].items():
    print(f"{dataset}: Cluster {cluster}")

# Cluster characteristics
for cluster_name, info in clustering['cluster_characteristics'].items():
    print(f"\n{cluster_name}:")
    print(f"  Datasets: {', '.join(info['datasets'])}")
    print(f"  Best technique: {info['best_technique']}")
    print(f"  Technique scores: {info['technique_scores']}")
```

#### **3. Improvement Pattern Analysis**
```python
# Analyze improvement over baseline
improvements = analyzer.analyze_improvement_patterns(df)

# Technique-wise improvements
for technique, data in improvements['technique_improvements'].items():
    success_rate = data['success_rate']
    mean_improvement = data['mean_improvement']
    print(f"{technique}:")
    print(f"  Success rate: {success_rate:.0%}")
    print(f"  Mean improvement: {mean_improvement:+.3f}")
    print(f"  Datasets improved: {data['positive_improvements']}/{data['total_comparisons']}")

# Dataset-wise analysis
for dataset, data in improvements['dataset_improvements'].items():
    best_tech = data['best_technique']
    best_improvement = data['best_improvement']
    print(f"{dataset}: {best_tech} (+{best_improvement:.3f})")
```

#### **4. Research Insights Generation**
```python
# Generate automated research insights
report = analyzer.create_comprehensive_analysis_report("results.json")

print("🎯 KEY RESEARCH INSIGHTS:")
for i, insight in enumerate(report['research_insights'], 1):
    print(f"{i:2d}. {insight}")

# Access specific findings
effectiveness = report['technique_effectiveness']
clustering = report['clustering_analysis'] 
improvements = report['improvement_patterns']
```

## 📊 **Output Files and Reports**

### **Generated Files**
- **`statistical_analysis.json`**: Complete statistical analysis results
- **`technique_comparison.png`**: Comprehensive technique comparison plot
- **`publication_figure.png`**: Publication-ready figure
- **`interactive_dashboard.html`**: Interactive analysis dashboard
- **`power_analysis_curves.png`**: Power analysis visualizations
- **`experimental_design_report.md`**: Methodology documentation
- **`comprehensive_analysis_report.json`**: Full results analysis with insights

### **Report Structure**
```json
{
  "data_summary": {...},
  "technique_effectiveness": {...},
  "best_techniques": {...},
  "clustering_analysis": {...},
  "improvement_patterns": {...},
  "research_insights": [...],
  "metadata": {...}
}
```

## 🔬 **Research Applications**

### **1. Bias Mitigation Evaluation**
- Compare effectiveness of different bias reduction techniques
- Identify which approaches work best for specific bias types
- Quantify improvement over baseline approaches
- Assess consistency across different evaluation datasets

### **2. Method Development**
- Guide development of new bias mitigation techniques
- Identify areas where current methods are insufficient
- Optimize hyperparameters based on performance patterns
- Validate new approaches with proper statistical rigor

### **3. Publication and Reporting**
- Generate publication-ready figures and tables
- Provide comprehensive statistical analysis for peer review
- Document experimental methodology and design decisions
- Create reproducible analysis workflows

### **4. Meta-Analysis**
- Combine results across multiple studies or model sizes
- Identify consistent patterns across different experimental conditions
- Guide future research directions based on empirical evidence
- Develop best practices for bias evaluation methodology

## 📈 **Statistical Methodology**

### **Significance Testing**
- **Parametric Tests**: t-tests for normally distributed data with equal variances
- **Non-parametric Tests**: Mann-Whitney U and Wilcoxon for non-normal data
- **Multiple Groups**: ANOVA or Kruskal-Wallis for comparing >2 techniques
- **Multiple Comparisons**: Holm-Bonferroni or FDR correction for family-wise error control

### **Effect Size Interpretation**
- **Cohen's d < 0.2**: Negligible effect
- **0.2 ≤ d < 0.5**: Small effect (still practically meaningful for bias reduction)
- **0.5 ≤ d < 0.8**: Medium effect (substantial bias reduction)
- **d ≥ 0.8**: Large effect (strong bias mitigation)

### **Power Analysis Guidelines**
- **Minimum Power**: 0.8 (80% chance of detecting true effects)
- **Preferred Power**: 0.9 (90% for important comparisons)
- **Sample Size**: Calculate based on smallest effect size of practical interest
- **Multiple Comparisons**: Adjust power calculations for planned comparisons

## 🎯 **Best Practices**

### **1. Data Collection**
- Use multiple random seeds for robust evaluation
- Include baseline comparisons for all techniques
- Evaluate on diverse bias types and datasets
- Document all experimental parameters

### **2. Statistical Analysis**
- Check assumptions before applying parametric tests
- Report effect sizes with confidence intervals
- Apply appropriate multiple comparisons corrections
- Use equivalence testing for non-significant results

### **3. Visualization**
- Include error bars or confidence intervals on all plots
- Use consistent color schemes across figures
- Provide clear axis labels and legends
- Create both overview and detailed plots

### **4. Interpretation**
- Focus on practical significance, not just statistical significance
- Consider consistency across datasets and bias types
- Acknowledge limitations and potential confounds
- Provide actionable recommendations for future research

## 🤝 **Contributing**

We welcome contributions to improve these data science tools:

1. **Statistical Methods**: Additional tests, corrections, or analysis approaches
2. **Visualizations**: New plot types or interactive features
3. **Experimental Design**: Advanced power analysis or design optimization
4. **Analysis Tools**: New metrics or insight generation methods

Please ensure all contributions include proper testing and documentation.

## 📜 **License**

This data science suite is released under the same license as the main FIRM repository. All tools are provided for research purposes with proper attribution.