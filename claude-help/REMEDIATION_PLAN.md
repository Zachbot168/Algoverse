# 🛠️ ALGOVERSE REMEDIATION PLAN

**Objective**: Transform Algoverse from a demo/prototype with fake data into a legitimate research framework with real bias evaluation capabilities.

---

## 📋 **PHASE 1: IMMEDIATE CLEANUP (Week 1-2)**

### **1.1 Remove All Fake Data Generation**

#### **Critical Files to Fix:**

**File**: `unified_pipeline/causal_analysis/bias_circuit_tracer.py`
- **Remove**: All `np.random` calls in result generation
- **Replace with**: Actual activation patching using hooks
- **Need from you**: 
  - Confirm which models you want to support (Qwen, Llama, Gemma, Ministral)
  - Preferred batch sizes for activation analysis

**File**: `unified_pipeline/train/component_registry.py`
- **Remove**: `np.random.random((12, 8))` fake importance matrices
- **Replace with**: Real attention head importance from gradient analysis
- **Need from you**: 
  - Which importance metrics to use (gradient norm, attention entropy, activation magnitude)

**File**: `unified_pipeline/eval/longitudinal_monitor.py` 
- **Remove**: `np.random.randint(-2, 3)` drift simulation
- **Replace with**: Real bias metric tracking over time
- **Need from you**:
  - How many monitoring points you want (3, 5, 10?)
  - Acceptable drift thresholds

**File**: `unified_pipeline/steer/das_wrapper.py`
- **Remove**: `0.5 + np.random.random() * 0.3` fake accuracy
- **Replace with**: Actual classifier training and evaluation
- **Need from you**:
  - Preferred classifier architecture (linear, MLP, transformer probe)
  - Training hyperparameters

### **1.2 Dataset Acquisition Priority**

#### **CRITICAL DATASETS (Must Have)**
1. **CrowS-Pairs** - Stereotype evaluation
2. **WinoBias** - Gender bias in coreference
3. **BBQ** - Demographic bias in QA
4. **StereoSet** - Stereotype completion

#### **HIGH PRIORITY DATASETS**
5. **SEAT/WEAT** - Implicit association tests
6. **TruthfulQA** - Truthfulness vs sycophancy
7. **BOLD** - Demographic fairness in generation

**Need from you**:
- Which datasets are highest priority for your research goals?
- Any licensing concerns or access restrictions?
- Preferred dataset versions/splits?

---

## 📊 **PHASE 2: REAL DATA INTEGRATION (Week 3-4)**

### **2.1 Dataset Download and Integration**

#### **Step 1: Automated Dataset Fetching**
```bash
# Enhanced pull_datasets.sh that actually works
#!/bin/bash
cd datasets/

# CrowS-Pairs
git clone https://github.com/nyu-mll/crows-pairs.git
cd crows-pairs && python download.py && cd ..

# WinoBias  
git clone https://github.com/uclanlp/corefBias.git winobias
cd winobias && python process_data.py && cd ..

# BBQ
mkdir bbq && cd bbq
wget https://github.com/nyu-mll/bbq/raw/main/data/BBQ.csv
cd ..

# StereoSet
git clone https://github.com/moinnadeem/StereoSet.git stereoset
cd stereoset && python download.py && cd ..
```

**Need from you**:
- Should I create this enhanced download script?
- Any specific dataset preprocessing requirements?
- Preferred data formats (JSON, CSV, HuggingFace datasets)?

#### **Step 2: Data Validation Pipeline**
```python
# New file: unified_pipeline/datasets/data_validator.py
class DatasetValidator:
    def validate_dataset_integrity(self, dataset_name):
        # Check file existence
        # Validate data format
        # Verify sample counts
        # Test data loading
```

**Need from you**:
- What validation checks are most important?
- Should we fail hard or warn on missing datasets?

### **2.2 Real Evaluation Implementation**

#### **Replace Fake Evaluators with Real Ones**

**File**: `unified_pipeline/eval/bias_evaluators.py` (NEW)
```python
class RealCrowsPairsEvaluator:
    def evaluate(self, model, tokenizer, dataset):
        # REAL: Load actual CrowS-Pairs data
        # REAL: Get model probabilities for stereotypical vs anti-stereotypical
        # REAL: Compute bias score from actual model outputs
        
class RealWinoBiasEvaluator:
    def evaluate(self, model, tokenizer, dataset):
        # REAL: Load WinoBias test cases
        # REAL: Get model predictions for pronoun resolution
        # REAL: Compute accuracy difference between pro/anti-stereotypical
```

**Need from you**:
- Which bias metrics are most important for your research?
- Should we implement multiple metrics per dataset?
- Preferred statistical significance tests?

---

## 🧠 **PHASE 3: FIRM PIPELINE RECONSTRUCTION (Week 5-8)**

### **3.1 Real Circuit Identification**

#### **Replace Fake Circuit Tracer**
```python
class RealBiasCircuitTracer:
    def identify_bias_circuits(self, model, bias_examples):
        # REAL: Hook into model attention layers
        # REAL: Run contrastive examples (biased vs unbiased)
        # REAL: Measure activation differences
        # REAL: Compute statistical significance of differences
        # REAL: Return circuits with p-values < 0.05
```

**Implementation Strategy**:
1. Use PyTorch hooks to capture attention weights
2. Run paired examples (stereotypical vs counter-stereotypical)
3. Compute statistical differences in attention patterns
4. Identify heads with significant bias-related activation

**Need from you**:
- How many example pairs per bias type? (100, 500, 1000?)
- Statistical significance threshold? (p < 0.05, p < 0.01?)
- Which activation patterns to analyze? (attention weights, value vectors, output projections?)

### **3.2 Real Causal Training**

#### **Replace Mock Pinpoint Tuning**
```python
class RealCausalPinpointTuner:
    def train_on_bias_circuits(self, model, identified_circuits, training_data):
        # REAL: Select top-k circuits by statistical significance
        # REAL: Apply LoRA to only those specific components
        # REAL: Train on bias mitigation examples
        # REAL: Validate intervention effectiveness
```

**Need from you**:
- How many circuits to target? (Top 10, 20, 50?)
- LoRA configuration preferences? (rank, alpha, dropout)
- Training data sources? (Custom examples, existing datasets, generated pairs?)
- Validation metrics for intervention success?

### **3.3 Real Steering Vector Computation**

#### **Replace Fake DSV Computer**
```python
class RealDSVComputer:
    def compute_steering_vectors(self, model, contrastive_pairs):
        # REAL: Get activations for biased vs unbiased examples
        # REAL: Compute mean difference vectors
        # REAL: Test steering effectiveness on held-out examples
        # REAL: Optimize layer and strength parameters
```

**Need from you**:
- How many contrastive pairs per bias type?
- Which layers to test for steering? (All, subset, learned selection?)
- Steering strength optimization method? (Grid search, Bayesian optimization?)

---

## 🔬 **PHASE 4: VALIDATION & ROBUSTNESS (Week 9-10)**

### **4.1 Real Multi-Seed Evaluation**

#### **Implement Genuine Robustness Testing**
```python
class RealRobustnessFramework:
    def run_multi_seed_evaluation(self, models, seeds, datasets):
        # REAL: Train with different random seeds
        # REAL: Evaluate on identical test sets
        # REAL: Compute confidence intervals
        # REAL: Test statistical significance
```

**Need from you**:
- How many seeds for robust evaluation? (5, 10, 20?)
- Which statistical tests? (t-test, Wilcoxon, Bootstrap?)
- Confidence interval levels? (95%, 99%?)

### **4.2 Longitudinal Monitoring**

#### **Real Intervention Persistence Tracking**
```python
class RealLongitudinalMonitor:
    def track_intervention_persistence(self, model, time_points):
        # REAL: Re-evaluate bias metrics at different time points
        # REAL: Track actual drift in model behavior
        # REAL: Measure intervention degradation
```

**Need from you**:
- How to simulate "time" in evaluation? (Different data splits, perturbed inputs?)
- What constitutes intervention failure? (X% increase in bias?)

---

## 📈 **PHASE 5: SCIENTIFIC VALIDATION (Week 11-12)**

### **5.1 Baseline Comparisons**

#### **Implement Real Baseline Methods**
```python
class BaselineComparison:
    def compare_against_baselines(self):
        # REAL: Implement CDA (Counterfactual Data Augmentation)
        # REAL: Implement INLP (Iterative Nullspace Projection)
        # REAL: Implement standard fine-tuning
        # REAL: Compare all methods on same datasets
```

**Need from you**:
- Which baseline methods to implement?
- How detailed should the comparison be?

### **5.2 Publication-Ready Results**

#### **Generate Legitimate Research Outputs**
```python
class ResearchResultsGenerator:
    def generate_publication_results(self):
        # REAL: Run complete evaluation pipeline
        # REAL: Generate statistical significance tests
        # REAL: Create publication-quality plots
        # REAL: Write results summary with proper caveats
```

---

## 💾 **WHAT I NEED FROM YOU**

### **Immediate Decisions (This Week)**

#### **1. Research Priorities**
- **Primary goal**: What's the main research question? (FIRM effectiveness, bias detection, intervention methods?)
- **Target models**: Which models are most important? (Focus on 1-2 initially?)
- **Key datasets**: Which 3-5 datasets are critical for your research?

#### **2. Computational Resources**
- **GPU availability**: What hardware do you have? (V100, A100, RTX 4090?)
- **Memory constraints**: Batch size limitations?
- **Time budget**: How long can experiments run? (Hours, days?)

#### **3. Implementation Preferences**
- **Framework choice**: Pure PyTorch or HuggingFace Transformers ecosystem?
- **Experiment tracking**: WandB, TensorBoard, or simple logging?
- **Result storage**: JSON files, databases, or HuggingFace datasets?

### **Technical Specifications Needed**

#### **1. Circuit Identification**
```
NEED YOUR INPUT:
- Statistical significance threshold: p < ? (0.05, 0.01, 0.001)
- Minimum effect size for circuit inclusion: Cohen's d > ? (0.2, 0.5, 0.8)
- Number of contrastive pairs per bias type: ? (100, 500, 1000)
- Attention analysis method: (gradient-based, activation-based, integrated gradients)
```

#### **2. Training Configuration**
```
NEED YOUR INPUT:
- LoRA rank: ? (8, 16, 32, 64)
- Learning rate: ? (1e-5, 5e-5, 1e-4)
- Training epochs: ? (1, 3, 5)
- Batch size: ? (2, 4, 8, 16)
- Gradient accumulation steps: ? (2, 4, 8)
```

#### **3. Evaluation Setup**
```
NEED YOUR INPUT:
- Robustness seeds: ? (5, 10, 20)
- Test set size per dataset: ? (100, 500, 1000, full)
- Statistical confidence level: ? (95%, 99%)
- Minimum detectable effect size: ? (0.1, 0.2, 0.3)
```

### **Development Timeline Options**

#### **Option A: Full Rebuild (12 weeks)**
- Complete rewrite of all fake components
- Implement all 13 datasets
- Full FIRM pipeline with real analysis
- Comprehensive validation and baselines

#### **Option B: Core Focus (6 weeks)**
- Fix 3-4 most critical datasets
- Implement real circuit identification
- Basic intervention validation
- Essential statistical testing

#### **Option C: Proof of Concept (3 weeks)**
- Single dataset (CrowS-Pairs) working end-to-end
- Basic real circuit identification
- Simple intervention testing
- Demonstrate feasibility

**Which timeline fits your needs?**

---

## 🎯 **SUCCESS CRITERIA**

### **Phase 1 Success**
- [ ] Zero `np.random` calls in result generation
- [ ] All datasets downloadable and loadable
- [ ] Basic evaluation runs without fake data

### **Phase 2 Success**
- [ ] Real model predictions on real datasets
- [ ] Computed bias metrics (not hardcoded)
- [ ] Statistical significance testing working

### **Phase 3 Success**
- [ ] Circuit identification with p-values
- [ ] Training that measurably changes model behavior
- [ ] Steering vectors with validated effectiveness

### **Phase 4 Success**
- [ ] Multi-seed robustness with confidence intervals
- [ ] Reproducible results across runs
- [ ] No random variation in identical conditions

### **Phase 5 Success**
- [ ] Publication-ready results
- [ ] Comprehensive evaluation report
- [ ] Code that passes scientific review

---

## 📞 **NEXT STEPS**

### **Immediate Actions (Today)**
1. **Review this plan** - Does the scope/timeline work for you?
2. **Prioritize components** - Which parts are most important?
3. **Specify requirements** - Answer the "NEED YOUR INPUT" questions
4. **Resource assessment** - Confirm available computational resources

### **This Week**
1. **Dataset prioritization** - Choose 3-5 critical datasets
2. **Implementation strategy** - Choose Option A, B, or C timeline
3. **Start Phase 1 cleanup** - Remove fake data from priority files

### **Next Week**
1. **Begin real implementation** - Start with highest-priority components
2. **Continuous validation** - Test each component as we build
3. **Documentation updates** - Keep remediation progress documented

---

**Let me know your decisions on the key questions above, and we'll start the systematic transformation from demo code to legitimate research framework!**