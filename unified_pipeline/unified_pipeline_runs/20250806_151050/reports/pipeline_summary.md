# Unified Bias Mitigation Pipeline Report

**Execution ID:** 20250806_151050  
**Model:** google/gemma-2-2b-it  
**Start Time:** 2025-08-06T15:10:50.893024  
**End Time:** 2025-08-06T15:40:01.818298  

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
- **Total Time:** 1729.7s

### Bias Type Performance

- **Stereotype:** 0.732 (1 datasets)
- **Gender:** 0.244 (3 datasets)
- **Racial:** 0.366 (2 datasets)
- **Religious:** 0.366 (2 datasets)
- **Sycophancy:** 1.000 (1 datasets)
- **Demographic:** 0.000 (1 datasets)

## Output Files

- **Reports:** `unified_pipeline_runs/20250806_151050/reports/`
- **Evaluation:** `unified_pipeline_runs/20250806_151050/evaluation/`
- **Diagnostics:** `unified_pipeline_runs/20250806_151050/diagnostics/`
