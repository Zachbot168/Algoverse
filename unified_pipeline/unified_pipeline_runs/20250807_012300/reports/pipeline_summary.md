# Unified Bias Mitigation Pipeline Report

**Execution ID:** 20250807_012300  
**Model:** gpt2  
**Start Time:** 2025-08-07T01:23:00.148496  
**End Time:** 2025-08-07T01:33:54.928117  

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
- **Total Time:** 646.0s

### Bias Type Performance

- **Stereotype:** 0.390 (2 datasets)
- **Gender:** 0.153 (8 datasets)
- **Racial:** 0.221 (5 datasets)
- **Religious:** 0.276 (4 datasets)
- **Demographic:** 0.108 (3 datasets)
- **Profession:** 0.000 (2 datasets)
- **Sycophancy:** 0.204 (5 datasets)

## Output Files

- **Reports:** `unified_pipeline_runs/20250807_012300/reports/`
- **Evaluation:** `unified_pipeline_runs/20250807_012300/evaluation/`
- **Diagnostics:** `unified_pipeline_runs/20250807_012300/diagnostics/`
