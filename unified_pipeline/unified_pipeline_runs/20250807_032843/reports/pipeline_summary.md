# Unified Bias Mitigation Pipeline Report

**Execution ID:** 20250807_032843  
**Model:** gpt2  
**Start Time:** 2025-08-07T03:28:43.674817  
**End Time:** 2025-08-07T03:30:35.556978  

## Dataset Coverage

- **Total Datasets:** 13
- **Working Datasets:** 4
- **High Priority Pending:** 0
- **Medium Priority Pending:** 3
- **Low Priority Pending:** 2

### Working Datasets
- ✅ CrowsPairs
- ✅ WinoBias
- ✅ SycophancyEval
- ✅ BBQ

### High Priority Pending
- 🔥 StereoSet
- 🔥 SEAT
- 🔥 TruthfulQA
- 🔥 WinoGender

## Baseline Evaluation Results

- **Datasets Evaluated:** 4
- **Total Time:** 104.3s

### Bias Type Performance

- **Stereotype:** 0.846 (1 datasets)
- **Gender:** 0.393 (3 datasets)
- **Racial:** 0.589 (2 datasets)
- **Religious:** 0.589 (2 datasets)
- **Sycophancy:** 0.700 (1 datasets)
- **Demographic:** 0.332 (1 datasets)

## Output Files

- **Reports:** `unified_pipeline_runs/20250807_032843/reports/`
- **Evaluation:** `unified_pipeline_runs/20250807_032843/evaluation/`
- **Diagnostics:** `unified_pipeline_runs/20250807_032843/diagnostics/`
