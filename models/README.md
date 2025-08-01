# Model Management System

This directory provides a comprehensive system for downloading and managing open-source models for bias evaluation and testing.

## 🚨 **IMPORTANT: Models are NOT committed to repository**

All downloaded models are automatically excluded from git commits via `.gitignore` rules. This prevents accidentally committing large model files (GBs) to the repository.

## 📋 **Available Models**

### BERT Family
- `bert-base-uncased` - 110M parameters, 0.4 GB
- `bert-large-uncased` - 340M parameters, 1.3 GB  
- `roberta-base` - 125M parameters, 0.5 GB

### Gemma Family (🔒 Requires Authentication)
- `gemma-2-2b-it` - 2B parameters, 4.9 GB
- `gemma-2-9b-it` - 9B parameters, 18.5 GB

### Llama Family (🔒 Requires Authentication)  
- `llama-3.2-1b` - 1B parameters, 2.5 GB
- `llama-3.2-3b` - 3B parameters, 6.2 GB

### GPT-2 Family
- `gpt2` - 124M parameters, 0.5 GB
- `gpt2-medium` - 355M parameters, 1.4 GB  
- `gpt2-large` - 774M parameters, 3.1 GB

## 🚀 **Quick Start**

### 1. Install Dependencies
```bash
pip install transformers huggingface_hub torch
```

### 2. List Available Models
```bash
python models/download_models.py --list
```

### 3. Download Models

**Download specific models:**
```bash
python models/download_models.py --models bert-base-uncased gpt2
```

**Download by collection:**
```bash
# Small models (good for testing)
python models/download_models.py --collection small_models

# Models that don't require authentication
python models/download_models.py --collection no_auth_required

# All BERT family models
python models/download_models.py --collection bert_family
```

**Download all models:**
```bash
python models/download_models.py --all
```

### 4. Authentication for Restricted Models

Some models (Gemma, Llama) require Hugging Face authentication:

```bash
# Option 1: Provide token directly
python models/download_models.py --token YOUR_HF_TOKEN --models gemma-2-2b-it

# Option 2: Set environment variable
export HF_TOKEN=YOUR_HF_TOKEN
python models/download_models.py --models gemma-2-2b-it

# Option 3: Interactive login (will prompt)
python models/download_models.py --models gemma-2-2b-it
```

## 📁 **Model Collections**

Pre-defined collections for easy batch downloading:

- `small_models` - Models under 1GB (bert-base-uncased, roberta-base, llama-3.2-1b, gpt2)
- `medium_models` - Models 1-5GB (bert-large-uncased, gemma-2-2b-it, llama-3.2-3b, gpt2-medium)  
- `large_models` - Models over 5GB (gemma-2-9b-it, gpt2-large)
- `no_auth_required` - Models that don't need authentication
- `auth_required` - Models requiring Hugging Face authentication
- `bias_testing` - Models suitable for bias evaluation
- `fairsteer_compatible` - Models compatible with FairSteer methodology
- `sycophancy_compatible` - Models compatible with sycophancy testing

## 🔧 **Management Commands**

### Check Download Status
```bash
python models/download_models.py --status
```

### Clean Up Failed Downloads
```bash  
python models/download_models.py --cleanup
```

### Force Re-download
```bash
python models/download_models.py --models gpt2 --force
```

## 📂 **Directory Structure**

After downloading, your models directory will look like:
```
models/
├── README.md                 # This file
├── download_models.py        # Download script
├── model_registry.json       # Model definitions
├── requirements.txt          # Python dependencies
├── cache/                    # HuggingFace cache
├── bert-base-uncased/        # Downloaded model
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer.json
├── gpt2/                     # Downloaded model
│   ├── config.json  
│   ├── pytorch_model.bin
│   └── tokenizer.json
└── ...
```

## 🔒 **Authentication Setup**

### Getting Hugging Face Token
1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" permissions
3. Accept model-specific licenses (Gemma, Llama) on their model pages

### Model License Requirements
- **Gemma models**: Accept Google Gemma license on model page
- **Llama models**: Accept Meta Llama license on model page  
- **BERT/GPT-2/RoBERTa**: No additional licenses required

## 💾 **Storage Requirements**

- **Minimum free space**: 20 GB
- **Recommended free space**: 50 GB
- **All models total**: ~40 GB

The system automatically checks available disk space before downloading.

## 🔗 **Integration with Bias Testing**

Downloaded models can be used directly with:
- **Unified Pipeline**: `unified_pipeline/run_full_pipeline.py`
- **FairSteer**: `fairsteer_debiasing.py`  
- **ZacharyModels**: `ZacharyModels/evaluate_model.py`
- **AlexModels**: Qwen bias testing scripts
- **YangModels**: BBQ, BOLD, WinoBias evaluation

## ⚠️ **Important Notes**

1. **Never commit models**: Models are automatically git-ignored
2. **Check licenses**: Some models require accepting terms
3. **Monitor disk space**: Models can be several GBs each
4. **Use collections**: Easier than downloading individually  
5. **Authentication**: Set up HF token for restricted models

## 🛠 **Troubleshooting**

### Common Issues

**"Model requires authentication"**
- Solution: Provide HF token and accept model license

**"Insufficient storage space"**  
- Solution: Free up disk space or download fewer models

**"Download failed"**
- Solution: Run `--cleanup` then retry, check internet connection

**"Model verification failed"**
- Files downloaded but loading failed - usually still usable

### Getting Help
```bash
python models/download_models.py --help
```

## 🔄 **Example Workflows**

### Quick Testing Setup
```bash
# Download small, no-auth models for quick testing
python models/download_models.py --collection small_models
```

### Full Research Setup  
```bash
# Authenticate first
export HF_TOKEN=your_token_here

# Download all bias-testing models
python models/download_models.py --collection bias_testing
```

### FairSteer Research
```bash  
# Download FairSteer-compatible models
python models/download_models.py --collection fairsteer_compatible --token YOUR_TOKEN
```

This system ensures you have easy access to all the models needed for comprehensive bias evaluation while keeping your repository clean and manageable.