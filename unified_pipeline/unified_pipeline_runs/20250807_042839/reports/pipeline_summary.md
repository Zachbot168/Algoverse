# Unified Bias Mitigation Pipeline Report

**Execution ID:** 20250807_042839  
**Model:** gpt2  
**Start Time:** 2025-08-07T04:28:39.873451  
**End Time:** 2025-08-07T04:32:18.427445  

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

- **Datasets Evaluated:** 8
- **Total Time:** 210.9s

### Bias Type Performance

- **Stereotype:** 0.671 (2 datasets)
- **Gender:** 0.376 (8 datasets)
- **Racial:** 0.497 (5 datasets)
- **Religious:** 0.619 (4 datasets)
- **Demographic:** 0.381 (3 datasets)
- **Profession:** 0.040 (2 datasets)

## Output Files

- **Reports:** `unified_pipeline_runs/20250807_042839/reports/`
- **Evaluation:** `unified_pipeline_runs/20250807_042839/evaluation/`
- **Diagnostics:** `unified_pipeline_runs/20250807_042839/diagnostics/`
