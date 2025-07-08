# CrowS-Pairs Evaluation Infrastructure

A modular framework for evaluating masked language models (MLMs) and large language models (LLMs) using the CrowS-Pairs dataset, a benchmark for measuring social bias via stereotypical vs. anti-stereotypical sentence pairs.

## Structure

```
ZacharyModels/
├── configs/
│   ├── models.yaml    # Model configurations
│   └── prompts.yaml   # Prompt templates
├── src/
│   ├── data/          # Dataset handling
│   ├── models/        # Model implementations
│   ├── prompts/       # Prompt management
│   └── evaluation/    # Scoring infrastructure
```

## Components

### Data Handling

- `BaseDataset`: Abstract base class for datasets
- `CrowsPairsDataset`: CrowS-Pairs specific implementation
- Features:
  - Efficient data loading
  - Caching support
  - Preprocessing pipeline
  - Bias type filtering

### Model Support

Supports various model architectures through a unified interface:

#### MLM Models
- BERT (Base/Large)
- RoBERTa (Base)

#### Autoregressive Models
- GPT-2 (Small/Medium)
- Qwen (0.6B/1.5B/3B)
- Gemma (2B/3B)
- LLaMA (1B/3B)

### Prompt System

Flexible prompt management with:
- Template library
- Dynamic formatting
- Few-shot support
- Model-specific templates
- Custom template creation

### Evaluation Framework

Comprehensive evaluation tools:
- Pair-wise scoring
- Batch processing
- Result caching
- Bias metrics computation
- Analysis tools

## Usage

### Basic Example

```python
from src.data import CrowsPairsDataset
from src.models import ModelFactory
from src.prompts import PromptWrapper
from src.evaluation import BiasScorer

# Load dataset
dataset = CrowsPairsDataset(
    data_dir="path/to/crows-pairs",
    cache_dir="path/to/cache"
)
dataset.load()

# Create model
model = ModelFactory.create("bert-base-uncased")
model.load()

# Setup prompt wrapper
wrapper = PromptWrapper(
    model_type="mlm",
    template_name="bias_comparison"
)

# Initialize scorer
scorer = BiasScorer(
    model=model,
    prompt_wrapper=wrapper
)

# Run evaluation
results = scorer.score_dataset(dataset)
```

### Advanced Usage

#### Custom Prompt Template

```python
from src.prompts import PromptTemplate

template = PromptTemplate(
    name="custom",
    template="Analyze bias in: {text}",
    description="Custom bias analysis prompt",
    variables=["text"]
)

wrapper = PromptWrapper(
    model_type="llm",
    custom_template=template.template
)
```

#### Bias Analysis

```python
from src.evaluation import BiasMetrics

# Analyze results
analysis = BiasMetrics.analyze_results(results)

# Save analysis
BiasMetrics.save_analysis(
    analysis,
    "output/analysis.json",
    model_name="bert-base-uncased"
)
```

## Model Configurations

Edit `configs/models.yaml` to:
- Add new models
- Modify model parameters
- Set default configurations

## Prompt Templates

Edit `configs/prompts.yaml` to:
- Define new templates
- Add few-shot examples
- Customize prompt modifiers
- Set default prompt settings

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- NumPy
- Pandas
- PyYAML

## Development

To contribute:
1. Fork the repository
2. Create a feature branch
3. Make changes
4. Submit pull request

## Notes

- The infrastructure is designed for research purposes
- Supports easy extension to new models and datasets
- Provides tools for analyzing and comparing model biases
- Can be integrated into larger evaluation pipelines
