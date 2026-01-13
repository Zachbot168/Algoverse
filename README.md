# FIRM: Fairness Interventions at Runtime and Model-training

A comprehensive research framework for bias evaluation and mitigation in Large Language Models (LLMs) implementing a novel 5-phase methodology combining causal analysis, targeted training, and runtime steering.

## Overview

FIRM implements a comprehensive approach to bias mitigation combining four distinct intervention strategies:

1. **Baseline Evaluation**: Comprehensive bias measurement across 13 datasets
2. **FairSteer Pipeline**: Dynamic bias steering using representation engineering
3. **Sycophancy Pipeline**: Truth vs. agreeableness bias mitigation using path patching
4. **FIRM Pipeline**: Complete 5-phase causal bias intervention framework

### FIRM 5-Phase Methodology

| Phase | Name | Description |
|-------|------|-------------|
| 1 | Bias Circuit Identification | Causal analysis to identify specific model components responsible for biased outputs |
| 2 | Causal Pinpoint Tuning | Selective fine-tuning targeting only bias-causing components |
| 3 | Layer-Aligned Steering Vectors | Computing steering vectors aligned with causal and training insights |
| 4 | Longitudinal Robustness Monitoring | Continuous bias drift detection and intervention persistence tracking |
| 5 | Multi-Layer Intervention Framework | Joint optimization across multiple model layers |

### Supported Models

| Model | Size | Architecture | Config File |
|-------|------|--------------|-------------|
| Qwen2.5 | 3B/1.5B | qwen2 | `qwen2.5-3b-instruct.yaml` |
| Llama 3.2 | 3B/1B | llama | `llama-3.2-3b-instruct.yaml` |
| Ministral | 3B | mistral | `ministral-3b-instruct.yaml` |
| Gemma 2 | 2B | gemma | `gemma-2-2b-it.yaml` |

### Integrated Bias Datasets

| Dataset | Bias Types | Samples |
|---------|------------|---------|
| CrowsPairs | Stereotypes, Gender, Racial, Religious | 1,508 |
| WinoBias | Gender, Occupational | 328 |
| BBQ | Demographic, Age, Religion, Nationality | 58,492 |
| SycophancyEval | Truth vs. Agreeableness | 51 |
| StereoSet | Gender, Profession, Race, Religion | 4,229 |
| WinoGender | Gender, Occupational | 120 |
| SEAT | Multiple social biases | 10 |
| TruthfulQA | Truthfulness, Sycophancy | 300 |
| BOLD | Demographic fairness | 43 |
| BiosBias | Occupational gender bias | 100 |
| MMLU | General knowledge bias | Various |
| HumanEval | Coding bias | Various |
| GSM8K | Mathematical reasoning | Various |

## Quick Start

### System Requirements

- Python 3.9 or higher
- CUDA-capable GPU with 16GB+ VRAM (24GB+ recommended for training)
- 32GB+ system RAM
- 50GB+ free disk space

### Installation

```bash
# Clone repository
git clone <repository-url>
cd Algoverse

# Create virtual environment
conda create -n algoverse python=3.9 -y
conda activate algoverse

# Run automated setup
python setup.py

# Download datasets
./pull_datasets.sh

# Authenticate with HuggingFace (required for gated models)
huggingface-cli login

# Verify installation
cd unified_pipeline
python test_installation.py
```

### Manual Installation

```bash
# Install PyTorch with CUDA support
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install all requirements
pip install -r requirements.txt

# Install spaCy language model
python -m spacy download en_core_web_sm
```

## Running the Pipeline

### Complete 4-Variant Evaluation (Recommended)

This runs all four model variants (baseline, FairSteer, sycophancy, FIRM) with statistical robustness:

```bash
cd unified_pipeline

# Quick evaluation
CUDA_VISIBLE_DEVICES=0 python run_integrated_pipeline.py \
    --model-config configs/models/qwen2.5-3b-instruct.yaml \
    --model-name "Qwen/Qwen2.5-3B-Instruct" \
    --suite comprehensive \
    --robust \
    --robustness-level quick

# Standard evaluation
CUDA_VISIBLE_DEVICES=0 python run_integrated_pipeline.py \
    --model-config configs/models/llama-3.2-3b-instruct.yaml \
    --model-name "meta-llama/Llama-3.2-3B-Instruct" \
    --suite comprehensive \
    --robust \
    --robustness-level standard

# Publication-quality evaluation
CUDA_VISIBLE_DEVICES=0 python run_integrated_pipeline.py \
    --model-config configs/models/gemma-2-2b-it.yaml \
    --model-name "google/gemma-2-2b-it" \
    --suite comprehensive \
    --robust \
    --robustness-level publication
```

### Robustness Levels

| Level | Training Seeds | Evaluation Seeds | Statistical Power | Estimated Time |
|-------|----------------|------------------|-------------------|----------------|
| quick | 2 | 2 | Basic (p<0.1) | 60-90 minutes |
| standard | 4 | 4 | Good (p<0.05) | 3-4 hours |
| publication | 6 | 6 | High (p<0.01) | 8-12 hours |

### Individual Pipeline Components

#### Baseline Evaluation Only

```bash
cd unified_pipeline
python run_unified_pipeline.py \
    --model-config configs/models/ministral-3b-instruct.yaml \
    --suite quick_evaluation \
    --output-dir baseline_results/
```

#### FIRM Pipeline Only

```bash
cd unified_pipeline
CUDA_VISIBLE_DEVICES=0 python firm_pipeline.py \
    --model-config configs/models/qwen2.5-3b-instruct.yaml \
    --model-name "Qwen/Qwen2.5-3B-Instruct" \
    --suite comprehensive \
    --output-dir firm_results/
```

#### FairSteer Debiasing

```bash
python fairsteer_debiasing.py \
    --config unified_pipeline/configs/models/gemma-2-2b-it.yaml \
    --output-dir steering_vectors/

cd unified_pipeline
python run_unified_pipeline.py \
    --model-config configs/models/gemma-2-2b-it.yaml \
    --model-variant fairsteer \
    --steering-vectors ../steering_vectors/
```

#### Sycophancy Mitigation

```bash
cd unified_pipeline/train
python sycophancy_pipeline.py \
    --model-config ../configs/models/qwen2.5-1.5b-instruct.yaml \
    --model-name "Qwen/Qwen2.5-1.5B-Instruct" \
    --output-dir sycophancy_results/
```

## Results and Outputs

All results are saved with timestamps:

```
unified_pipeline/
├── unified_pipeline_runs/          # Standard evaluation results
│   └── YYYYMMDD_HHMMSS/
│       ├── evaluation/baseline/    # Raw evaluation data
│       ├── evaluation/summary.csv  # Summary statistics
│       └── reports/                # Analysis reports
├── firm_pipeline_runs/             # FIRM pipeline results
│   └── firm_MODEL_YYYYMMDD_HHMMSS/
│       ├── phase_1_circuit_identification/
│       ├── phase_2_causal_training/
│       ├── phase_3_layer_aligned_steering/
│       ├── phase_4_longitudinal_monitoring/
│       ├── phase_5_multi_layer_intervention/
│       └── FIRM_COMPLETE_RESULTS.json
└── robust_evaluation_results/      # Multi-seed evaluation results
```

### Analyzing Results

```bash
cd data_science

# Generate statistical analysis
python statistical_analyzer.py \
    --results ../unified_pipeline/unified_pipeline_runs/latest/evaluation_results.json \
    --output-dir analysis_results/

# Create visualizations
python visualization_tools.py \
    --results ../unified_pipeline/unified_pipeline_runs/latest/evaluation_results.json \
    --output-dir plots/

# Generate research report
python results_analyzer.py \
    --results ../unified_pipeline/firm_pipeline_runs/latest/FIRM_COMPLETE_RESULTS.json \
    --output-format latex
```

## Project Structure

```
Algoverse/
├── unified_pipeline/           # Main framework
│   ├── configs/                # Model and dataset configurations
│   ├── datasets/               # Dataset loaders
│   ├── train/                  # Training components (Phase 2)
│   ├── steer/                  # Steering vectors (Phase 3)
│   ├── eval/                   # Evaluation framework (Phase 4-5)
│   ├── causal_analysis/        # Circuit identification (Phase 1)
│   └── utils/                  # Utility functions
├── datasets/                   # Dataset storage
├── models/                     # Model utilities
├── data_science/               # Analysis tools
├── fairsteer_debiasing.py      # Steering vector implementation
├── setup.py                    # Automated setup
├── pull_datasets.sh            # Dataset download script
├── requirements.txt            # Python dependencies
├── ARCHITECTURE.md             # Detailed architecture documentation
└── PIPELINE_DIAGRAM.md         # Visual pipeline explanation
```

## Configuration

### Model Configuration

Create custom model configs in `unified_pipeline/configs/models/`:

```yaml
model:
  name: "your-org/your-model"
  architecture: llama  # or qwen2, mistral, gemma
  device: auto
  torch_dtype: float16
  trust_remote_code: true

model_variant: baseline
num_layers: 32
num_heads: 32
hidden_size: 4096

max_length: 2048
temperature: 0.7
top_p: 0.9

interventions:
  pinpoint_tuning:
    component_selection:
      max_components: 32
      min_importance: 0.05
    lora:
      r: 8
      alpha: 16
      target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

### Evaluation Suite Configuration

Edit `unified_pipeline/configs/evaluation_suites/`:

```yaml
quick_evaluation:
  datasets: ["CrowsPairs", "WinoBias"]
  samples_per_dataset: 200

comprehensive:
  datasets: ["CrowsPairs", "WinoBias", "BBQ", "StereoSet", "TruthfulQA"]
  samples_per_dataset: 1000
```

## Troubleshooting

### CUDA Out of Memory

```bash
export BATCH_SIZE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Model Access Denied

```bash
huggingface-cli login
# Accept model license on HuggingFace website
```

### Dataset Not Found

```bash
./pull_datasets.sh
```

### Performance Optimization

```bash
export USE_MIXED_PRECISION=1
export GRADIENT_CHECKPOINTING=1
export TORCH_DYNAMO_DISABLE=1
export TOKENIZERS_PARALLELISM=false
```

## Citation

```bibtex
@misc{firm_framework_2024,
  title={FIRM: Fairness Interventions at Runtime and Model-training},
  author={[Authors]},
  year={2024},
  note={5-phase bias mitigation framework with causal analysis},
  url={<repository-url>}
}
```

## License

This project is under active research development. See LICENSE file for details.
