## High level overview

The code applies the FairSteer method to Google's Gemma-2-2b-it model with the Winobias dataset, performing debiasing through:

1. **Biased Activation Detection (BAD)** - Identifies bias-sensitive model layers
2. **Debiasing Steering Vector (DSV) Computation** - Calculates intervention vectors
3. **Inference-Time Intervention** - Applies minimal interventions during generation to not steer unbiased text

Explanation by function: 
` _load_gemma_model` - loads model (requires huggingface token in a separate .py)

## BAD Computation
`_get_optimal_layer_range` - obtains optimal layer based on fairsteer (13-15)

`construct_dbad_dataset` - uses BBQ and MMLU to construct a bias activation detection dataset. Basically combining BBQ and MMLU data. MMLU data prevents overfitting while training. This uses `_load_bbq_data` which has 2 fallbacks: `_load_bbq_from_bias_bench` and `_load_bbq_from_huggingface`

## DSV Computation
`construct_ddsv_dataset` creates contrastive pairs, which is a pair of the biased and unbiased prompt in each category. 

`compute_steering_vectors` uses the contrastive pairs to calculate the magnitude of the steering vectors. 

## Inference-Time Intervention
`detect_bias` provides the bias probability for each prompt, where `debias_generation` steers the model whenever the `bias_probability` is above the threshold of 0.5

`_generate_with_intervention()` Applies steering vector intervention during text generation. It registers a forward hook on the optimal layer, modifies hidden states by adding the steering vector (scaled by intervention strength), and then generates text

`intervention_strength`: Controls how strongly the debiasing is applied, also uses temperature-controlled sampling for diverse outputs

`_get_layer_module()` is a helper to access specific transformer layers 

`_generate_with_hooks()` provides alternative intervention method using more robust hook handling  

`_generate_with_intervention` handles multiple output formats (tuples, direct tensors), more comprehensive hidden state modification
 
`_generate_normal()` baseline generation without any intervention  

`train_bias_classifiers()` trains layer-wise logistic regression models for bias detection by splitting DBAD dataset into train/validation sets.

Extracts activations for all prompts and trains classifiers on each layer's activations

`validate_linear_separability()` validates paper's claim about linear separability of bias features. Uses 3 fold cross validation. 

`evaluate_capability_preservation()` tests if intervention preserves model's general knowledge. Uses word overlap similarity between original/debiased outputs and also checks factual correctness on non-bias prompts. 80% is the threshold

`_calculate_response_similarity()` computes simple similarity metric between generations  

`similarity = (shared_words) / (total_unique_words)`

Workflow functions:

`main_fairsteer_demo()`
`evaluate_fairsteer_on_winobias()`



