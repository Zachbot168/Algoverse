# Unified Bias Mitigation Pipeline - Setup Guide

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# For GPU acceleration (optional)
pip install flash-attn nvidia-ml-py3
```

### 2. Validate Installation

```bash
# Test pipeline setup
python test_pipeline.py
```

### 3. Run Basic Bias Evaluation

```bash
# Baseline bias measurement
python run_full_pipeline.py --config configs/baseline.yaml --dataset_size 100

# Full bias mitigation pipeline
python run_full_pipeline.py --config configs/full.yaml --dataset_size 500
```

## 🔧 Detailed Setup

### System Requirements

- **Python**: 3.9+ 
- **GPU**: CUDA-compatible GPU recommended (8GB+ VRAM)
- **RAM**: 16GB+ recommended for larger models
- **Storage**: 10GB+ free space for models and results

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

The pipeline uses YAML configuration files in `configs/`:

- **`baseline.yaml`**: Raw model evaluation (no bias mitigation)
- **`full.yaml`**: Complete bias mitigation (training + inference)
- **`pinpoint.yaml`**: Training-time bias mitigation only
- **`steer.yaml`**: Inference-time bias steering only

### Data Setup

Ensure bias evaluation datasets are available:

```bash
# The pipeline expects datasets at:
../datasets/bbq/              # Bias Benchmark for QA
../datasets/winobias/         # Gender bias in coreference
../datasets/crows-pairs/      # Stereotype detection
../datasets/bold/             # Bias in open-ended generation
```

## 🎯 Understanding Bias Mitigation

### What This Pipeline Does

1. **Bias Detection**: Identifies model components that cause biased outputs
2. **Selective Debiasing**: Fine-tunes only bias-causing components
3. **Dynamic Steering**: Real-time bias correction during generation
4. **Comprehensive Evaluation**: Tests across multiple bias benchmarks

### Bias Types Addressed

- **Gender Bias**: "Women are naturally better at caregiving"
- **Racial Bias**: "Asian students are naturally good at math"  
- **Religious Bias**: "Religious people are more moral"
- **Socioeconomic Bias**: Class-based assumptions

### Pipeline Stages

```
1. Diagnostic Pass     → Identify bias-causing components
2. Component Registry  → Record which parts to target
3. Selective Training  → Fine-tune bias components only
4. Steering Vectors    → Compute real-time corrections
5. Unified Evaluation  → Test bias reduction effectiveness
6. Final Report        → Comprehensive bias analysis
```

## 🐛 Troubleshooting

### Common Issues

#### ImportError: No module named 'sklearn'
```bash
pip install scikit-learn
```

#### CUDA Out of Memory
- Reduce `batch_size` in config files
- Use smaller models (e.g., Llama-2-7b instead of 13b)
- Enable gradient checkpointing

#### JSON Decode Errors
- The pipeline now handles malformed JSON gracefully
- Check dataset files for formatting issues
- Use `test_pipeline.py` to validate setup

#### Pipeline Interrupted by User
- Partial results are saved to output directory
- Can resume from interruption point
- Check `pipeline_partial.json` for state

### Getting Help

1. **Validation**: Run `python test_pipeline.py` first
2. **Logs**: Check output directories for detailed logs
3. **Config**: Verify YAML files have required sections
4. **Dependencies**: Ensure all packages in requirements.txt are installed

## 📊 Expected Results

### Bias Reduction Targets
- **Gender Bias**: 60-80% reduction in stereotype scores
- **Racial Bias**: 50-70% reduction in ethnic stereotypes
- **Religious Bias**: 40-60% reduction in faith-based prejudice

### Performance Impact
- **Accuracy**: <5% degradation on standard benchmarks
- **Inference Speed**: <20% slower with dynamic steering
- **Training Time**: 2-4 hours for selective debiasing

## 🔬 Research Applications

This pipeline enables research into:

- **Bias Localization**: Where bias occurs in neural networks
- **Intervention Efficiency**: Minimal changes for maximum bias reduction
- **Bias Persistence**: How bias patterns evolve over time
- **Fairness Trade-offs**: Balance between bias reduction and performance

## 📝 Citation

When using this pipeline for research, please cite the relevant papers:
- Sycophancy-Interpretability framework
- Fairsteer bias mitigation approach
- Original bias evaluation datasets (BBQ, WinoBias, etc.)

## 🤝 Contributing

1. Test changes with `python test_pipeline.py`
2. Ensure bias mitigation effectiveness is maintained
3. Add tests for new bias categories or mitigation methods
4. Update documentation for new features

---

**Remember**: This pipeline is designed for **defensive bias mitigation research**. The goal is to make language models more fair and equitable, not to create biased systems.