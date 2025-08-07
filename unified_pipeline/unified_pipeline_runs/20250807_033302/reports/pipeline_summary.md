# Unified Bias Mitigation Pipeline Report

**Execution ID:** 20250807_033302  
**Model:** gpt2  
**Start Time:** 2025-08-07T03:33:02.900739  
**End Time:** 2025-08-07T03:43:05.620454  

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
- **Total Time:** 594.6s

### Bias Type Performance

- **Stereotype:** 0.671 (2 datasets)
- **Gender:** 0.343 (8 datasets)
- **Racial:** 0.377 (5 datasets)
- **Religious:** 0.469 (4 datasets)
- **Demographic:** 0.181 (3 datasets)
- **Profession:** 0.005 (2 datasets)
- **Sycophancy:** 0.514 (5 datasets)

## Output Files

- **Reports:** `unified_pipeline_runs/20250807_033302/reports/`
- **Evaluation:** `unified_pipeline_runs/20250807_033302/evaluation/`
- **Diagnostics:** `unified_pipeline_runs/20250807_033302/diagnostics/`
