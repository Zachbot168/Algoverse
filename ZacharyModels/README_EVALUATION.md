# Model Evaluation Guide

## Running Evaluations

The evaluation harness provides two ways to run model evaluations:

1. Single model evaluation:
```bash
python evaluate_model.py <model_name> <model_type> [--resume] [--overwrite] [--hf-token TOKEN] [--include-gated]
```

2. Batch evaluation of all models:
```bash
python run_all.py [--resume] [--overwrite] [--hf-token TOKEN] [--include-gated]
```

## Command-line Arguments

- `--resume` (default: True): Skip models that have already been evaluated
- `--overwrite`: Force re-evaluation even if results exist
- `--hf-token TOKEN`: HuggingFace token for accessing gated models
- `--include-gated`: Attempt to evaluate gated models (default: False)

## Environment Variables

- `HUGGINGFACE_TOKEN`: Alternative way to provide HuggingFace token
  ```bash
  export HUGGINGFACE_TOKEN="your_token_here"
  ```

## Directory Structure

```
ZacharyModels/
├── results/                    # Evaluation results
│   ├── model_name_results.json    # Raw evaluation results
│   └── model_name_analysis.json   # Analysis metrics
├── evaluate_model.py          # Single model evaluation script
└── run_all.py                # Batch evaluation script
```

## Features

### Resume Capability
- By default, the system tracks evaluation progress and skips models that have already been evaluated
- Progress is stored in `.evaluation_state.json`
- Use `--overwrite` to force re-evaluation

### Gated Model Support
- Handles license-gated models (e.g., Gemma, Llama, Qwen)
- Automatically resolves model aliases to canonical names
- Provides clear messages about license requirements
- Can skip gated models unless --include-gated is specified

### Token Handling
- Accepts HuggingFace token via environment or CLI flag
- Retries failed requests with token when available
- Gracefully handles access control failures
- Clear instructions for enabling gated models

### Model Validation
- Automatically validates models exist on HuggingFace Hub
- Gracefully skips unavailable models with warning
- Provides detailed error messages for failed evaluations

### Template Fallbacks
- Attempts to use model-specific templates first
- Falls back to general LLM templates if needed
- Ultimately defaults to basic template as last resort

### Logging & Progress
- Detailed logging to both console and `evaluation.log`
- Progress tracking with completed/skipped/failed counts
- Summary report at end of batch evaluation

## Examples

1. Resume a previous batch evaluation:
```bash
python run_all.py
```

2. Re-evaluate all models:
```bash
python run_all.py --overwrite
```

3. Evaluate a single model:
```bash
python evaluate_model.py bert-base-uncased mlm
```

4. Re-evaluate a specific model:
```bash
python evaluate_model.py gpt2 llm --overwrite
```

5. Evaluate gated model:
```bash
python evaluate_model.py gemma-2-2b-it gemma --include-gated --hf-token "YOUR_TOKEN"
```

6. Batch evaluate with gated models:
```bash
export HUGGINGFACE_TOKEN="YOUR_TOKEN"
python run_all.py --include-gated
```

## Handling Gated Models

Some models (like Gemma, Llama, and Qwen) require license acceptance and authentication. To use these:

1. Visit the model page on HuggingFace Hub (e.g., https://huggingface.co/google/gemma-2b-it)
2. Accept the model license by clicking "Agree"
3. Create an access token at https://huggingface.co/settings/tokens
4. Either:
   - Set the `HUGGINGFACE_TOKEN` environment variable, or
   - Pass the token via `--hf-token`
5. Run with `--include-gated` to evaluate these models

The system will automatically:
- Resolve model aliases (e.g., "gemma-2-2b-it" → "google/gemma-2b-it")
- Try public access first
- Retry with token if needed
- Skip gated models if token is missing

## Error Handling

The evaluation harness provides detailed feedback:
- Model not found: Suggests checking the model name
- License required: Provides instructions for accepting license
- Access denied: Indicates token is needed
- Template missing: Falls back to simpler templates
