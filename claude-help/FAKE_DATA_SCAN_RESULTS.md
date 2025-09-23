# Fake Data Scan Results

**Scan Date**: Sat Sep 20 23:33:17 CDT 2025

🚨 FAKE DATA DETECTION REPORT
==================================================

Total issues found: 9

HIGH ISSUES (4):
------------------------------
File: unified_pipeline/causal_analysis/bias_circuit_tracer.py
Line: 128
Category: hardcoded_values
Pattern: importance.*=.*0\.[89]\d*
Code: head_importance = importance_score * 0.8  # Placeholder: replace with real head analysis

File: unified_pipeline/causal_analysis/bias_circuit_tracer.py
Line: 205
Category: hardcoded_values
Pattern: importance.*=.*0\.[89]\d*
Code: importance_score = 0.8 - (abs(layer_idx - 18) * 0.1)  # Peak around layer 18

File: unified_pipeline/train/component_registry.py
Line: 136
Category: hardcoded_values
Pattern: accuracy.*=.*0\.[567]\d*
Code: accuracy_threshold = config.get('bad_accuracy_threshold', 0.65)

File: unified_pipeline/steer/das_wrapper.py
Line: 422
Category: hardcoded_values
Pattern: accuracy.*=.*0\.[567]\d*
Code: classifier.accuracy_ = results.get('accuracy', 0.5)

MEDIUM ISSUES (1):
------------------------------
File: unified_pipeline/eval/unified_evaluator.py
Line: 579
Category: suspicious_fallbacks
Pattern: return.*np\.zeros\(\d+\)
Code: return np.zeros(64)

LOW ISSUES (4):
------------------------------
File: unified_pipeline/causal_analysis/bias_circuit_tracer.py
Line: 128
Category: mock_indicators
Pattern: #.*placeholder
Code: head_importance = importance_score * 0.8  # Placeholder: replace with real head analysis

File: unified_pipeline/eval/run_benchmark.py
Line: 545
Category: mock_indicators
Pattern: #.*placeholder
Code: response = "A"  # Placeholder

File: unified_pipeline/eval/run_benchmark.py
Line: 608
Category: mock_indicators
Pattern: #.*placeholder
Code: response = ""  # Placeholder

File: unified_pipeline/eval/firm_vs_fairsteer_comparator.py
Line: 273
Category: mock_indicators
Pattern: #.*placeholder
Code: "gender_bias": 0.45,  # Placeholder scores


ISSUES BY FILE:
--------------------
 3 issues: unified_pipeline/causal_analysis/bias_circuit_tracer.py
 2 issues: unified_pipeline/eval/run_benchmark.py
 1 issues: unified_pipeline/train/component_registry.py
 1 issues: unified_pipeline/eval/unified_evaluator.py
 1 issues: unified_pipeline/eval/firm_vs_fairsteer_comparator.py
 1 issues: unified_pipeline/steer/das_wrapper.py

## Dataset Availability

- crows-pairs: ❌
- winobias: ❌
- winogender: ❌
- bbq: ❌
- stereoset: ❌
- bold: ❌
- truthfulqa: ❌
