# Algoverse

A comprehensive research repository for bias evaluation and mitigation in Large Language Models (LLMs). 

## 🚀 Unified Bias Mitigation Pipeline

**NEW**: Complete end-to-end bias mitigation system that combines training-time and inference-time interventions to create production-ready, bias-reduced models.

### Key Features
- **Multi-bias detection**: Gender, racial, religious, sycophantic biases
- **Real evaluation data**: Uses actual bias datasets for component identification
- **Dual intervention**: Selective fine-tuning + runtime steering vectors
- **Production ready**: Deployable models with comprehensive evaluation
- **Scientific rigor**: Direct before/after comparison on identical test cases

### Quick Start
```bash
cd unified_pipeline
python run_full_pipeline.py --config configs/models/gemma-2-2b-it.yaml --dataset_size 500
```

**📖 Complete Guide**: See [`unified_pipeline/README_COMPLETE.md`](unified_pipeline/README_COMPLETE.md)

---

## 📊 Bias Evaluation Datasets | [Dataset Division](https://docs.google.com/document/d/1KniHEniG8daH5q5ou9aiCtROFqGAbuvgPsAHKcG_I6o/edit?tab=t.0)

This repository contains the following bias evaluation datasets:

- **CrowS-Pairs**: Crowdsourced Stereotype Pairs dataset
- **BiasBench**: Comprehensive bias evaluation framework (replaces StereoSet)
- **Winobias**: Winograd Bias dataset (all splits, from Hugging Face)
- **Winogender**: Winograd Gender dataset (manual download required)
- **Bias in Bios**: Bias in Bios dataset
- **BOLD**: Bias in Open-Ended Language Generation dataset
- **BBQ**: Bias Benchmark for Question Answering dataset
- **UNQOVER**: Similar QA to BBQ
- **CEB**: Open ended generation, similar to BOLD

## Sources

These datasets are pulled from their respective repositories and sources:

- CrowS-Pairs: https://github.com/nyu-mll/crows-pairs
- BiasBench: https://github.com/McGill-NLP/bias-bench
- Winobias: https://huggingface.co/datasets/wino_bias (all splits: type1_anti, type1_pro, type2_anti, type2_pro)
- Winogender: https://github.com/rudinger/winogender-schemas (see 'data' directory)
- Bias in Bios: https://github.com/microsoft/biosbias
- BOLD: https://github.com/amazon-research/bold
- BBQ: https://huggingface.co/datasets/walledai/BBQ?library=datasets
- UNQOVER: https://aclanthology.org/2020.findings-emnlp.311/
- CEB: https://arxiv.org/abs/2407.02408

---

## 🔧 Current Implementation Status

### ✅ Integrated into Unified Pipeline (4/12 datasets)
- **CrowS-Pairs**: 1,508 demographic bias pairs - ✅ Working
- **WinoBias**: 3,168 gender bias examples - ✅ Working  
- **Sycophancy Eval**: 10,997 agreement-seeking examples - ✅ Working
- **BBQ**: QA bias benchmark - ⚠️ Loader exists, needs testing

### ❌ Not Yet Integrated (8/12 datasets)

#### 🔥 High Priority (Ready for Integration)
- **StereoSet** (from BiasBench): Comprehensive stereotype evaluation
- **SEAT/WEAT** (from BiasBench): 40+ implicit association tests
- **TruthfulQA**: Truth vs sycophancy tradeoff evaluation
- **WinoGender**: Additional gender bias evaluation

#### 🟡 Medium Priority 
- **BOLD**: Open-ended generation bias evaluation
- **Bias in Bios**: Professional stereotype evaluation
- **MMLU**: Academic knowledge sycophancy (57 subjects)

#### 🟢 Lower Priority
- **HumanEval/GSM8K**: Task-specific sycophancy evaluation
- **UNQOVER/CEB**: Additional QA and generation bias tests

### 📈 Expansion Potential
**Current Coverage**: 33% (4/12 datasets)  
**Potential Coverage**: 100% with full integration  
**Impact**: 3x increase in bias evaluation comprehensiveness

---

## 🎯 What Still Needs to be Added

### Phase 1: Quick Wins (2-3 weeks)
1. **BBQ Integration**: Enable existing loader, add to configs
2. **TruthfulQA**: Add sycophancy vs truthfulness evaluation  
3. **WinoGender**: Similar structure to WinoBias integration

### Phase 2: Major Extensions (4-6 weeks)
1. **StereoSet Integration**: Comprehensive stereotype detection
2. **SEAT/WEAT Integration**: 40+ implicit bias association tests
3. **BOLD Integration**: Open-ended generation bias evaluation
4. **Bias in Bios Integration**: Professional bias evaluation

### Phase 3: Advanced Features (6-8 weeks)
1. **MMLU Sycophancy**: Academic domain coverage (57 subjects)
2. **Out-of-distribution Tests**: Political, philosophical sycophancy
3. **Advanced Analytics**: Cross-dataset bias correlation analysis
4. **Multi-model Framework**: Comparative bias analysis

### Implementation Priorities
1. 🔥 **StereoSet + SEAT/WEAT**: Biggest impact on bias evaluation coverage
2. 🔥 **TruthfulQA**: Critical for sycophancy vs truth tradeoff analysis
3. 🟡 **BOLD**: Important for generative bias evaluation
4. 🟡 **Professional bias datasets**: Important for real-world applications

---

## 🚀 Getting Started

### For Bias Evaluation Research
```bash
# Download all datasets
./pull_datasets.sh

# Explore individual model evaluation
cd ZacharyModels  # or YangModels, AlexModels
python evaluate_model.py
```

### For Bias Mitigation (Recommended)
```bash
# Run complete bias mitigation pipeline
cd unified_pipeline
python run_full_pipeline.py --config configs/models/gemma-2-2b-it.yaml --dataset_size 500

# Results: Deployable bias-reduced model + comprehensive evaluation
```

**📖 Full Documentation**: [`unified_pipeline/README_COMPLETE.md`](unified_pipeline/README_COMPLETE.md)

