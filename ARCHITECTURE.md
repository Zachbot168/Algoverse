# FIRM Framework Architecture

## Overview

The FIRM (Fairness Interventions at Runtime and Model-training) framework implements a comprehensive 5-phase bias mitigation pipeline combining causal analysis, targeted training, and runtime steering.

## Core Components

### 📁 Directory Structure

```
unified_pipeline/
├── causal_analysis/           # Phase 1: Circuit Identification
│   ├── bias_circuit_tracer.py       # Legacy circuit identification
│   └── real_circuit_identification.py  # Real activation analysis
├── train/                     # Phase 2: Causal Training
│   ├── component_registry.py        # Component management
│   ├── real_lora_training.py       # Genuine LoRA training
│   └── sycophancy_pipeline.py      # Sycophancy mitigation
├── steer/                     # Phase 3: Steering Vectors
│   ├── das_wrapper.py              # Dynamic activation steering
│   ├── real_steering_vectors.py    # Real steering computation
│   └── multi_layer_steering.py     # Multi-layer interventions
├── eval/                      # Phase 4 & 5: Evaluation & Monitoring
│   ├── unified_evaluator.py        # Main evaluation engine
│   ├── real_bias_evaluator.py      # Real bias evaluation
│   ├── baseline_method_comparator.py  # Scientific validation
│   └── longitudinal_monitor.py     # Long-term monitoring
└── datasets/                  # Dataset Management
    ├── base_loader.py              # Abstract base classes
    ├── bias_loaders.py             # Dataset implementations
    └── data_validator.py           # Data integrity checks
```

## 5-Phase FIRM Methodology

### Phase 1: Bias Circuit Identification
**Goal**: Identify specific model components responsible for biased outputs

**Components**:
- `real_circuit_identification.py`: Genuine activation analysis
- `bias_circuit_tracer.py`: Legacy implementation (being phased out)

**Process**:
1. Collect activations for bias/neutral example pairs
2. Perform causal interventions on model components
3. Measure statistical significance of bias contributions
4. Generate circuit importance rankings

**Output**: List of bias-related attention heads and MLP layers

### Phase 2: Causal Pinpoint Tuning
**Goal**: Selectively fine-tune only bias-causing components

**Components**:
- `real_lora_training.py`: LoRA adapters for identified circuits
- `component_registry.py`: Component tracking and management

**Process**:
1. Load identified circuits from Phase 1
2. Apply LoRA adapters to target components only
3. Fine-tune on bias mitigation datasets
4. Validate intervention effectiveness

**Output**: Fine-tuned model with targeted bias reduction

### Phase 3: Layer-Aligned Steering Vectors
**Goal**: Compute steering vectors aligned with causal insights

**Components**:
- `real_steering_vectors.py`: Genuine steering vector computation
- `das_wrapper.py`: Dynamic activation steering system

**Process**:
1. Collect contrastive activations (biased vs neutral)
2. Compute mean difference vectors as steering directions
3. Validate steering effectiveness on held-out data
4. Optimize layer selection and steering strength

**Output**: Validated steering vectors for runtime bias mitigation

### Phase 4: Longitudinal Robustness Monitoring
**Goal**: Monitor intervention persistence and detect bias drift

**Components**:
- `longitudinal_monitor.py`: Long-term intervention tracking
- `intervention_persistence_tracker.py`: Persistence measurement

**Process**:
1. Periodically re-evaluate bias metrics
2. Track intervention degradation over time
3. Detect distribution shift and bias drift
4. Generate alerts for intervention failure

**Output**: Continuous monitoring reports and drift detection

### Phase 5: Multi-Layer Intervention Framework
**Goal**: Joint optimization across multiple intervention strategies

**Components**:
- `baseline_method_comparator.py`: Scientific validation
- `publication_results_generator.py`: Research-grade reporting

**Process**:
1. Compare FIRM against established baseline methods
2. Statistical significance testing across interventions
3. Multi-seed robustness evaluation
4. Generate publication-ready results

**Output**: Comprehensive scientific validation report

## Data Flow

```
Input Text
    ↓
[Phase 1] Circuit Identification
    ↓ (identified circuits)
[Phase 2] Causal Training  →  [Phase 3] Steering Vectors
    ↓ (fine-tuned model)       ↓ (steering vectors)
[Combined Model with Runtime Steering]
    ↓
[Phase 4] Longitudinal Monitoring
    ↓
[Phase 5] Scientific Validation
    ↓
Bias-Mitigated Output
```

## Key Design Principles

### 1. Real Data Only
- All fake data generation has been removed
- Statistical calculations use scipy/numpy
- Model predictions are genuine forward passes

### 2. Modular Architecture
- Each phase can be run independently
- Standardized interfaces between components
- Easy to swap implementations

### 3. Scientific Rigor
- Statistical significance testing throughout
- Multi-seed evaluation for robustness
- Reproducible experimental setup

### 4. Scalable Evaluation
- 13 integrated bias datasets
- Support for multiple model architectures
- Efficient batch processing

## Configuration System

### Model Configs (`configs/models/`)
```yaml
model:
  name: "model-org/model-name"
  architecture: llama  # or qwen2, mistral, gemma
  device: auto
  torch_dtype: float16

interventions:
  pinpoint_tuning:
    lora:
      r: 8
      alpha: 16
      target_modules: ["q_proj", "v_proj"]
```

### Dataset Configs (`configs/datasets.yaml`)
```yaml
datasets:
  CrowsPairs:
    path: "../datasets/crows-pairs"
    loader: "CrowsPairsLoader"
    bias_types: ["stereotype", "gender", "racial"]
```

### Evaluation Suites (`configs/evaluation_suites/`)
```yaml
comprehensive:
  datasets: ["CrowsPairs", "WinoBias", "BBQ"]
  samples_per_dataset: 1000
  metrics: ["bias_score", "accuracy", "fairness"]
```

## Integration Points

### External Libraries
- **Transformers**: Model loading and inference
- **PEFT**: LoRA adapter implementation
- **SciPy**: Statistical analysis and testing
- **scikit-learn**: Machine learning utilities
- **Pandas**: Data manipulation and analysis

### Model Support
- **Qwen2.5**: Native architecture support
- **Llama 3.2**: Full compatibility
- **Ministral**: Complete integration
- **Gemma 2**: Tested and validated

### Bias Datasets
- Standardized loader interface
- Automatic data validation
- Consistent evaluation metrics
- Comprehensive coverage (13 datasets)

## Performance Considerations

### Memory Optimization
- Gradient checkpointing for training
- Mixed precision inference
- Dynamic batch sizing
- CPU offloading for large models

### Computational Efficiency
- Selective component targeting (Phase 2)
- Cached activation analysis (Phase 1)
- Parallel dataset evaluation
- Optimized statistical computations

### Scalability
- Multi-GPU support for training
- Distributed evaluation pipelines
- Configurable resource allocation
- Progress tracking and resumption

## Quality Assurance

### Testing Framework
- Unit tests for each component
- Integration tests for full pipeline
- Regression tests for reproducibility
- Performance benchmarks

### Validation Pipeline
- Statistical significance requirements
- Cross-validation protocols
- Robustness testing (multiple seeds)
- Scientific peer review standards

### Monitoring and Logging
- Comprehensive execution logging
- Performance metrics tracking
- Error handling and recovery
- Progress visualization

This architecture ensures scientific rigor, computational efficiency, and practical applicability for bias mitigation research in large language models.