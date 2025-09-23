# FIRM: Fairness Interventions at Runtime and Model-training

A comprehensive research framework for bias evaluation and mitigation in Large Language Models (LLMs) featuring our **FIRM (Fairness Interventions at Runtime and Model-training)** 5-phase methodology.

> **✅ Status: Production Ready** - All fake data has been removed and replaced with real implementations

## 🎯 **Overview**

**FIRM** implements a novel **5-phase comprehensive approach** to bias mitigation combining four distinct intervention strategies:

1. **Baseline Evaluation**: Comprehensive bias measurement across 13 datasets
2. **FairSteer Pipeline**: Dynamic bias steering using representation engineering
3. **Sycophancy Pipeline**: Truth vs. agreeableness bias mitigation using path patching
4. **FIRM Pipeline**: Complete 5-phase causal bias intervention framework

### 🧠 **FIRM 5-Phase Methodology**

**Phase 1: Bias Circuit Identification** - Causal analysis to identify specific model components responsible for biased outputs  
**Phase 2: Causal Pinpoint Tuning** - Selective fine-tuning targeting only bias-causing components  
**Phase 3: Layer-Aligned Steering Vectors** - Computing steering vectors aligned with causal and training insights  
**Phase 4: Longitudinal Robustness Monitoring** - Continuous bias drift detection and intervention persistence tracking  
**Phase 5: Multi-Layer Intervention Framework** - Joint optimization across multiple model layers  

### 🚀 **Supported Models**

| Model | Size | Architecture | Status | Config File |
|-------|------|--------------|---------|-------------|
| **Qwen2.5** | 3B/1.5B | qwen2 | ✅ **Fully Working** | `qwen2.5-3b-instruct.yaml` |
| **Llama 3.2** | 3B/1B | llama | ✅ **Fully Working** | `llama-3.2-3b-instruct.yaml` |
| **Ministral** | 3B | mistral | ✅ **Fully Working** | `ministral-3b-instruct.yaml` |
| **Gemma 2** | 2B | gemma | ✅ **Fully Working** | `gemma-2-2b-it.yaml` |

### 📊 **13 Integrated Bias Datasets**

All datasets have implemented loaders, though source data must be downloaded separately:

| Dataset | Bias Types | Samples | Implementation |
|---------|------------|---------|----------------|
| **CrowsPairs** | Stereotypes, Gender, Racial, Religious | 1,508 | ✅ Loader Ready |
| **WinoBias** | Gender, Occupational | 328 | ✅ Loader Ready |
| **BBQ** | Demographic, Age, Religion, Nationality | 58,492 | ✅ Loader Ready |
| **SycophancyEval** | Truth vs. Agreeableness | 51 | ✅ Loader Ready |
| **StereoSet** | Gender, Profession, Race, Religion | 4,229 | ✅ Loader Ready |
| **WinoGender** | Gender, Occupational | 120 | ✅ Loader Ready |
| **SEAT** | Multiple social biases | 10 | ✅ Loader Ready |
| **TruthfulQA** | Truthfulness, Sycophancy | 300 | ✅ Loader Ready |
| **BOLD** | Demographic fairness | 43 | ✅ Loader Ready |
| **BiosBias** | Occupational gender bias | 100 | ✅ Loader Ready |
| **MMLU** | General knowledge bias | Various | ✅ Loader Ready |
| **HumanEval** | Coding bias | Various | ✅ Loader Ready |
| **GSM8K** | Mathematical reasoning | Various | ✅ Loader Ready |

## 📋 **Quick Start Installation**

### **Automated Setup (Recommended)**

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
./enhanced_pull_datasets.sh

# Authenticate with HuggingFace
huggingface-cli login

# Test installation
cd unified_pipeline
python test_installation.py
```

### **Manual Installation**

<details>
<summary>Click to expand manual installation steps</summary>

### **System Requirements**

- Python 3.9 or higher
- CUDA-capable GPU with 16GB+ VRAM (24GB+ recommended for training)
- 32GB+ system RAM
- 50GB+ free disk space
- Ubuntu 20.04+ or macOS 12+ (Windows with WSL2)

### **Step 1: Clone Repository**

```bash
git clone <repository-url>
cd Algoverse
```

### **Step 2: Create Virtual Environment**

```bash
# Using conda (recommended)
conda create -n algoverse python=3.9
conda activate algoverse

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **Step 3: Install Dependencies**

```bash
# Install PyTorch with CUDA support
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install all requirements
pip install -r requirements.txt

# Install spaCy language model
python -m spacy download en_core_web_sm
```

### **Step 4: Download Datasets**

```bash
# Run enhanced dataset download script
chmod +x enhanced_pull_datasets.sh
./enhanced_pull_datasets.sh
```

### **Step 5: Set Environment Variables**

```bash
# Automatically set by setup.py, or manually:
export TORCH_DYNAMO_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
```

</details>

## 🚀 **Running the Complete Pipeline**

### **Option 1: Complete 4-Variant Evaluation (Recommended)**

This runs all four model variants (baseline, FairSteer, sycophancy, FIRM) with statistical robustness:

```bash
cd unified_pipeline

# Quick evaluation (2-3 hours)
CUDA_VISIBLE_DEVICES=0 python run_integrated_pipeline.py \
    --model-config configs/models/qwen2.5-3b-instruct.yaml \
    --model-name "Qwen/Qwen2.5-3B-Instruct" \
    --suite comprehensive \
    --robust \
    --robustness-level quick

# Standard evaluation (4-6 hours)
CUDA_VISIBLE_DEVICES=0 python run_integrated_pipeline.py \
    --model-config configs/models/llama-3.2-3b-instruct.yaml \
    --model-name "meta-llama/Llama-3.2-3B-Instruct" \
    --suite comprehensive \
    --robust \
    --robustness-level standard

# Publication-quality evaluation (8-12 hours)
CUDA_VISIBLE_DEVICES=0 python run_integrated_pipeline.py \
    --model-config configs/models/gemma-2-2b-it.yaml \
    --model-name "google/gemma-2-2b-it" \
    --suite comprehensive \
    --robust \
    --robustness-level publication
```

### **Option 2: Individual Pipeline Components**

#### **2.1 Baseline Evaluation Only**

```bash
cd unified_pipeline
python run_unified_pipeline.py \
    --model-config configs/models/ministral-3b-instruct.yaml \
    --suite quick_evaluation \
    --output-dir baseline_results/
```

#### **2.2 FIRM Pipeline Only**

```bash
cd unified_pipeline
CUDA_VISIBLE_DEVICES=0 python firm_pipeline.py \
    --model-config configs/models/qwen2.5-3b-instruct.yaml \
    --model-name "Qwen/Qwen2.5-3B-Instruct" \
    --suite comprehensive \
    --output-dir firm_results/
```

#### **2.3 FairSteer Debiasing**

```bash
# First, create steering vectors
python fairsteer_debiasing.py \
    --config unified_pipeline/configs/models/gemma-2-2b-it.yaml \
    --output-dir steering_vectors/

# Then evaluate with steering
cd unified_pipeline
python run_unified_pipeline.py \
    --model-config configs/models/gemma-2-2b-it.yaml \
    --model-variant fairsteer \
    --steering-vectors ../steering_vectors/
```

#### **2.4 Sycophancy Mitigation**

```bash
cd unified_pipeline/train
python sycophancy_pipeline.py \
    --model-config ../configs/models/qwen2.5-1.5b-instruct.yaml \
    --model-name "Qwen/Qwen2.5-1.5B-Instruct" \
    --output-dir sycophancy_results/
```

### **Option 3: Custom Dataset Evaluation**

```bash
cd unified_pipeline
python eval/run_benchmark.py \
    --model-config configs/models/llama-3.2-1b-instruct.yaml \
    --datasets CrowsPairs WinoBias BBQ \
    --num-samples 500 \
    --output-dir custom_eval/
```

## 📊 **Understanding Results**

### **Results Location**

All results are saved with timestamps:

```
unified_pipeline/
├── unified_pipeline_runs/          # Standard evaluation results
│   └── YYYYMMDD_HHMMSS/           
│       ├── evaluation/baseline/    # Raw evaluation data
│       ├── evaluation/summary.csv  # Summary statistics
│       └── reports/               # Analysis reports
├── firm_pipeline_runs/            # FIRM pipeline results  
│   └── firm_MODEL_YYYYMMDD_HHMMSS/
│       ├── phase_1_circuit_identification/
│       ├── phase_2_causal_training/
│       ├── phase_3_layer_aligned_steering/
│       ├── phase_4_longitudinal_monitoring/
│       ├── phase_5_multi_layer_intervention/
│       └── FIRM_COMPLETE_RESULTS.json
└── robust_evaluation_results/     # Multi-seed evaluation results
```

### **Analyzing Results**

```bash
# Generate statistical analysis
cd data_science
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

## 🔧 **Configuration Guide**

### **Model Configuration Example**

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

# Required settings
max_length: 2048
temperature: 0.7
top_p: 0.9

# FIRM interventions
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

### **Evaluation Suite Configuration**

Edit `unified_pipeline/configs/evaluation_suites/`:

```yaml
quick_evaluation:
  datasets: ["CrowsPairs", "WinoBias"]
  samples_per_dataset: 200
  
comprehensive:
  datasets: ["CrowsPairs", "WinoBias", "BBQ", "StereoSet", "TruthfulQA"]
  samples_per_dataset: 1000
```

## 🚨 **Troubleshooting**

### **Common Issues and Solutions**

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   export BATCH_SIZE=1
   # Or use CPU offloading
   export CUDA_VISIBLE_DEVICES=""
   ```

2. **Dataset Not Found**
   ```bash
   # Check dataset path
   ls datasets/
   # Re-run download script
   ./pull_datasets.sh
   ```

3. **Model Access Denied**
   ```bash
   # Re-authenticate with HuggingFace
   huggingface-cli login
   # Accept model license on HF website
   ```

4. **Import Errors**
   ```bash
   # Ensure you're in the virtual environment
   which python
   # Reinstall requirements
   pip install -r requirements.txt --force-reinstall
   ```

5. **Empty Results**
   ```bash
   # Check logs for errors
   tail -n 100 unified_pipeline/unified_pipeline_runs/latest/run.log
   # Verify dataset files exist
   find datasets/ -name "*.json" -o -name "*.csv" | head
   ```

### **Performance Optimization**

```bash
# Use mixed precision
export USE_MIXED_PRECISION=1

# Enable gradient checkpointing
export GRADIENT_CHECKPOINTING=1

# Limit memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## 📁 **Project Structure**

```
Algoverse/
├── unified_pipeline/              # Main framework
│   ├── configs/                   # Model and dataset configs
│   ├── datasets/                  # Dataset loaders
│   ├── train/                     # Training components (Phase 2)
│   ├── steer/                     # Steering vectors (Phase 3)
│   ├── eval/                      # Evaluation framework (Phase 4-5)
│   ├── causal_analysis/           # Circuit identification (Phase 1)
│   ├── test_installation.py       # Installation verification
│   └── run_unified_pipeline.py    # Main entry point
├── datasets/                      # Dataset storage (run ./enhanced_pull_datasets.sh)
├── setup.py                       # Automated setup script
├── enhanced_pull_datasets.sh      # Enhanced dataset download
├── requirements.txt               # Python dependencies
├── ARCHITECTURE.md               # Detailed architecture docs
└── README.md                     # This file
```

## 🔬 **Advanced Usage**

### **Running with Custom Seeds**

```python
from unified_pipeline.robust_evaluation_framework import RobustEvaluationFramework

evaluator = RobustEvaluationFramework()
evaluator.set_custom_config(
    training_seeds=[42, 123, 456, 789, 999],
    evaluation_seeds=[100, 200, 300, 400, 500],
    dataset_sample_sizes={"CrowsPairs": 1000, "WinoBias": 500}
)

results = evaluator.run_robust_four_model_evaluation(
    base_config_path="configs/models/qwen2.5-3b-instruct.yaml",
    model_name="Qwen/Qwen2.5-3B-Instruct",
    suite="comprehensive",
    robustness_level="custom"
)
```

### **Batch Processing Multiple Models**

```bash
#!/bin/bash
# batch_evaluate.sh
models=("qwen2.5-3b-instruct" "llama-3.2-3b-instruct" "gemma-2-2b-it")

for model in "${models[@]}"; do
    echo "Evaluating $model..."
    python run_integrated_pipeline.py \
        --model-config configs/models/${model}.yaml \
        --suite comprehensive \
        --robust \
        --robustness-level quick
done
```

## 📚 **Citation**

If you use this framework in your research:

```bibtex
@misc{firm_framework_2024,
  title={FIRM: Fairness Interventions at Runtime and Model-training},
  author={[Authors]},
  year={2024},
  note={5-phase bias mitigation framework with causal analysis},
  url={<repository-url>}
}
```

## 🤝 **Contributing**

We welcome contributions! Areas of interest:
- Additional model architecture support
- New bias evaluation datasets
- Enhanced statistical analysis methods
- Performance optimizations

## ✅ **Recent Improvements**

### Fixed Issues (All fake data removed):
- ✅ **Baseline Method Comparator**: Real implementations replace time.sleep() simulation
- ✅ **Statistical Power Calculations**: Now uses scipy.stats instead of hardcoded values  
- ✅ **Steering Vector Generation**: Proper error handling instead of random fallbacks
- ✅ **Circuit Identification**: Real activation analysis replaces heuristics
- ✅ **Evaluation Framework**: Genuine model predictions throughout

### New Features:
- 🆕 **Automated Setup**: `python setup.py` for one-command installation
- 🆕 **Enhanced Dataset Download**: `./enhanced_pull_datasets.sh` with verification
- 🆕 **Installation Testing**: `python test_installation.py` to verify setup
- 🆕 **Architecture Documentation**: Detailed framework documentation in `ARCHITECTURE.md`
- 🆕 **Production Ready**: All components use real data and genuine computations

## 📜 **License**

This project is under active research development. See LICENSE file for details.