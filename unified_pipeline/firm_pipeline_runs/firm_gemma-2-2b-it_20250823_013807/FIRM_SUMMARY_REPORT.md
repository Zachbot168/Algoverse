# FIRM Pipeline Results Report

## Model Information
- **Model**: google/gemma-2-2b-it
- **Bias Types Analyzed**: gender, race, religion
- **Pipeline Version**: FIRM v1.0

## Phase Results Summary

### Phase 1: Bias Circuit Identification
- ✅ Circuits Identified: 30
- ✅ Bias Types Covered: 3

### Phase 2: Causal Pinpoint Tuning
- ✅ Components Trained: 27
- ✅ Validation: PASSED

### Phase 3: Layer-Aligned Steering Vectors
- ✅ Layer Alignment Hypothesis: SUPPORTED
- ✅ Strategies Tested: 5

### Phase 4: Longitudinal Robustness
- ✅ Intervention Persistence: CONFIRMED
- ✅ Bias Drift: NONE

### Phase 5: Multi-Layer Intervention
- ✅ Downstream Robustness: VALIDATED
- ✅ Unrelated Layer Isolation: CONFIRMED

## Overall FIRM Validation

✅ **FIRM PIPELINE SUCCESSFUL**

All core FIRM requirements have been implemented and tested:
- ✅ Causal circuit identification using path patching
- ✅ Layer alignment between training and inference
- ✅ Longitudinal robustness monitoring  
- ✅ Multi-layer intervention robustness

## Output Files
All detailed results and intermediate outputs are saved in the pipeline output directory.
