# Unified Bias Mitigation Pipeline

This directory contains the main FIRM (Fairness Interventions at Runtime and Model-training) framework implementation for comprehensive bias detection and mitigation in large language models.

## Overview

The unified pipeline integrates four complementary approaches:

1. **Baseline Evaluation**: Comprehensive bias measurement across 13 datasets
2. **FairSteer Pipeline**: Dynamic bias steering using representation engineering
3. **Sycophancy Pipeline**: Truth vs. agreeableness bias mitigation using path patching
4. **FIRM Pipeline**: Complete 5-phase causal bias intervention framework

## Directory Structure

```
unified_pipeline/
├── configs/                    # Configuration files
│   ├── models/                 # Model-specific configurations
│   ├── evaluation_suites/      # Evaluation suite definitions
│   ├── datasets.yaml           # Dataset configurations
│   └── *.yaml                  # Pipeline configurations
├── causal_analysis/            # Phase 1: Circuit Identification
│   ├── bias_circuit_tracer.py
│   └── real_circuit_identification.py
├── train/                      # Phase 2: Causal Training
│   ├── causal_pinpoint_tuning.py
│   ├── component_registry.py
│   ├── real_lora_training.py
│   └── sycophancy_pipeline.py
├── steer/                      # Phase 3: Steering Vectors
│   ├── compute_dsv.py
│   ├── layer_aligned_dsv.py
│   ├── model_agnostic_fairsteer.py
│   ├── multi_layer_steering.py
│   └── real_steering_vectors.py
├── eval/                       # Phase 4-5: Evaluation
│   ├── unified_evaluator.py
│   ├── real_bias_evaluator.py
│   ├── longitudinal_monitor.py
│   └── statistical_robustness_tester.py
├── datasets/                   # Dataset loaders
│   ├── bias_loaders.py
│   ├── sycophancy_loaders.py
│   └── unified_registry.py
├── utils/                      # Utility functions
├── firm_pipeline.py            # Main FIRM pipeline
├── run_integrated_pipeline.py  # 4-variant comparison runner
├── run_unified_pipeline.py     # Single-variant evaluation
└── test_installation.py        # Installation verification
```

## Supported Models

| Model | Size | Architecture | Config File |
|-------|------|--------------|-------------|
| Qwen2.5 | 3B | qwen2 | `configs/models/qwen2.5-3b-instruct.yaml` |
| Qwen2.5 | 1.5B | qwen2 | `configs/models/qwen2.5-1.5b-instruct.yaml` |
| Llama 3.2 | 3B | llama | `configs/models/llama-3.2-3b-instruct.yaml` |
| Llama 3.2 | 1B | llama | `configs/models/llama-3.2-1b-instruct.yaml` |
| Gemma 2 | 2B | gemma | `configs/models/gemma-2-2b-it.yaml` |
| Ministral | 3B | mistral | `configs/models/ministral-3b-instruct.yaml` |

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (8GB+ VRAM for basic, 24GB+ for full pipeline)
- HuggingFace account with access to gated models

### Installation Verification

```bash
python test_installation.py
```

### Running Evaluations

#### 4-Variant Robust Evaluation (Recommended)

```bash
CUDA_VISIBLE_DEVICES=0 python run_integrated_pipeline.py \
    --model-config configs/models/qwen2.5-3b-instruct.yaml \
    --model-name "Qwen/Qwen2.5-3B-Instruct" \
    --suite comprehensive \
    --robust \
    --robustness-level standard
```

#### FIRM Pipeline Only

```bash
CUDA_VISIBLE_DEVICES=0 python firm_pipeline.py \
    --model-config configs/models/qwen2.5-3b-instruct.yaml \
    --model-name "Qwen/Qwen2.5-3B-Instruct" \
    --suite comprehensive
```

#### Baseline Evaluation

```bash
python run_unified_pipeline.py \
    --model-config configs/models/gemma-2-2b-it.yaml \
    --suite quick_evaluation
```

## Robustness Levels

| Level | Seeds | Evaluations | Statistical Power | Time |
|-------|-------|-------------|-------------------|------|
| quick | 2x2 | 4 | Basic (p<0.1) | 60-90 min |
| standard | 4x4 | 16 | Good (p<0.05) | 3-4 hours |
| publication | 6x6 | 36 | High (p<0.01) | 8-12 hours |

## FIRM Pipeline Phases

### Phase 1: Bias Circuit Identification

Identifies specific model components responsible for biased outputs through causal analysis.

**Output**: `phase_1_circuit_identification/identified_circuits.json`

### Phase 2: Causal Pinpoint Tuning

Selective fine-tuning targeting only bias-causing components using LoRA adapters.

**Output**: `phase_2_causal_training/` (trained adapters and metadata)

### Phase 3: Layer-Aligned Steering Vectors

Computes steering vectors aligned with causal and training insights using 5 alignment strategies:
- causal_aligned
- training_aligned
- optimal_overlap
- baseline_middle
- downstream

**Output**: `phase_3_layer_aligned_steering/` (steering vectors and validation)

### Phase 4: Longitudinal Robustness Monitoring

Continuous bias drift detection and intervention persistence tracking.

**Output**: `phase_4_longitudinal_monitoring/` (monitoring reports)

### Phase 5: Multi-Layer Intervention Framework

Joint optimization across multiple model layers with downstream robustness analysis.

**Output**: `phase_5_multi_layer_intervention/` (final results)

## Output Structure

```
firm_pipeline_runs/firm_{model}_{timestamp}/
├── phase_1_circuit_identification/
│   └── identified_circuits.json
├── phase_2_causal_training/
│   ├── trained_model/
│   └── causal_training_metadata.json
├── phase_3_layer_aligned_steering/
│   ├── *_steering_vectors.pkl
│   └── layer_alignment_validation.json
├── phase_4_longitudinal_monitoring/
│   └── longitudinal_drift_analysis.json
├── phase_5_multi_layer_intervention/
│   └── multi_layer_intervention_results.json
├── FIRM_COMPLETE_RESULTS.json
└── FIRM_SUMMARY_REPORT.md
```

## Configuration

### Model Configuration

```yaml
model:
  name: "model-org/model-name"
  architecture: llama  # qwen2, mistral, gemma
  device: auto
  torch_dtype: float16
  trust_remote_code: true

model_variant: baseline
num_layers: 32
num_heads: 32
hidden_size: 4096

temperature: 0.7  # Required
max_length: 2048

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

## Troubleshooting

### Out of Memory

```bash
python utils/gpu_optimizer.py --config configs/models/MODEL.yaml
```

### Model Access Denied

```bash
huggingface-cli login
```

### Slow Performance

```bash
export GRADIENT_CHECKPOINTING=1
export USE_MIXED_PRECISION=1
```
