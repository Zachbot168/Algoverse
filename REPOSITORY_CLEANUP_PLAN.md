# Repository Cleanup and Documentation Plan

## Files to Remove (Outdated/Test Artifacts)

### Individual Developer Folders
- AlexModels/
- YangModels/
- ZacharyModels/

### Test Files and Debug Scripts
- test_*.py (60+ debug scripts from development)
- debug_*.py (pipeline debugging artifacts)
- simple_test.py, quick_dataset_test.py
- validate_fixes.py

### Temporary Documentation
- *_ANALYSIS.md
- *_SUMMARY.md
- *_GUIDE.md
- *_REPORT.md
- DATASET_DOWNLOAD_GUIDE.md (consolidate into main README)
- EVALUATION_PIPELINE_FIXES.md
- FAIRSTEER_OPTIMIZATION_REPORT.md

### Old Pipeline Runs and Results
- pipeline_runs/ (old format)
- gpt2_quick_results/
- benchmark_results/
- unified_pipeline/pipeline_runs/ (old runs)
- unified_pipeline/comparative_evaluation/results/ (timestamped comparison files)
- unified_pipeline/robust_evaluation_results/ (many timestamped runs)

### Redundant Configuration Files
- unified_pipeline/configs/models/*_20250*.yaml (timestamped configs)
- unified_pipeline/configs/test_baseline.yaml

## Files to Keep and Document

### Core Components
- unified_pipeline/ - Main bias evaluation framework
- sycophancy-interpretability/ - Sycophancy mitigation
- datasets/ - All bias evaluation datasets
- models/ - Model storage
- fairsteer_debiasing.py - Steering vectors
- fairsteer_gemma2b.pkl - Trained steering data

### Configuration
- unified_pipeline/configs/models/gemma-2-2b-it.yaml (main config)
- unified_pipeline/configs/datasets.yaml
- requirements.txt

### Scripts
- pull_datasets.sh - Dataset download automation
- unified_pipeline/run_integrated_pipeline.py - Main execution script

## Documentation Plan

### Main README.md Structure
1. Technical Overview
2. Quick Start Guide
3. Dataset Download Instructions
4. Complete Setup Guide
5. Sycophancy Repository Integration
6. Evaluation Interpretation Guide
7. Advanced Usage

### Implementation Details to Document
- Exact dataset download procedures
- Model path configurations
- Evaluation suite options
- Results interpretation
- Known limitations