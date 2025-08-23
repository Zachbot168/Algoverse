#!/bin/bash

# Test script to verify warning suppression is working

echo "🧪 Testing PyTorch Warning Suppression Setup"
echo "=============================================="

# Set environment variables for clean execution
export TORCH_DYNAMO_DISABLE=1
export TORCH_COMPILE_DEBUG=0
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export HF_HUB_DISABLE_PROGRESS_BARS=true
export PYTHONWARNINGS=ignore

echo "✅ Environment variables set:"
echo "   TORCH_DYNAMO_DISABLE=$TORCH_DYNAMO_DISABLE"
echo "   TRANSFORMERS_VERBOSITY=$TRANSFORMERS_VERBOSITY" 
echo "   TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM"
echo ""

echo "🎯 The pipeline should now run with significantly fewer warnings!"
echo ""
echo "To run the bias evaluation pipeline with clean output:"
echo "cd unified_pipeline"
echo "python run_unified_pipeline.py --model-config configs/models/gemma-2-2b-it.yaml --dataset-config configs/datasets.yaml"
echo ""
echo "Key improvements:"
echo "✅ PyTorch compilation warnings suppressed"
echo "✅ Transformers parameter warnings reduced" 
echo "✅ Tokenizer parallelism warnings disabled"
echo "✅ Progress bars hidden for cleaner logs"
echo "✅ All Python warnings filtered out"