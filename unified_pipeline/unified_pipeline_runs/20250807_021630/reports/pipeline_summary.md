# Unified Bias Mitigation Pipeline Report

**Execution ID:** 20250807_021630  
**Model:** gpt2  
**Start Time:** 2025-08-07T02:16:30.476276  
**End Time:** 2025-08-07T02:26:21.385410  

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

- **Datasets Evaluated:** 13
- **Total Time:** 583.0s

### Bias Type Performance

- **Stereotype:** 0.671 (2 datasets)
- **Gender:** 0.209 (8 datasets)
- **Racial:** 0.335 (5 datasets)
- **Religious:** 0.419 (4 datasets)
- **Demographic:** 0.111 (3 datasets)
- **Profession:** 0.000 (2 datasets)
- **Sycophancy:** 0.200 (5 datasets)

## Output Files

- **Reports:** `unified_pipeline_runs/20250807_021630/reports/`
- **Evaluation:** `unified_pipeline_runs/20250807_021630/evaluation/`
- **Diagnostics:** `unified_pipeline_runs/20250807_021630/diagnostics/`
