#!/bin/bash
# Download All Models Script
# This script downloads all models needed for bias testing in Algoverse
#
# Usage:
#   ./download_all_models.sh              # Download all models (interactive auth)
#   ./download_all_models.sh --token TOKEN # Download with provided HF token
#   ./download_all_models.sh --no-auth     # Download only non-auth models
#   ./download_all_models.sh --quick       # Download small models for quick testing

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MODELS_DIR="$SCRIPT_DIR/models"

echo -e "${BLUE}🚀 Algoverse Model Download Script${NC}"
echo -e "${BLUE}====================================${NC}"

# Check if we're in the right directory
if [ ! -f "$MODELS_DIR/download_models.py" ]; then
    echo -e "${RED}❌ Error: Cannot find models/download_models.py${NC}"
    echo -e "${RED}   Make sure you're running this from the Algoverse root directory${NC}"
    exit 1
fi

# Check if Python dependencies are installed
echo -e "${YELLOW}🔍 Checking dependencies...${NC}"
if ! python -c "import transformers, huggingface_hub, torch" 2>/dev/null; then
    echo -e "${YELLOW}📦 Installing required dependencies...${NC}"
    pip install -r "$MODELS_DIR/requirements.txt"
else
    echo -e "${GREEN}✅ Dependencies already installed${NC}"
fi

# Parse command line arguments
HF_TOKEN=""
MODE="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --token)
            HF_TOKEN="$2"
            shift 2
            ;;
        --no-auth)
            MODE="no-auth"
            shift
            ;;
        --quick)
            MODE="quick"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --token TOKEN    Use provided HuggingFace token"
            echo "  --no-auth        Download only models that don't require authentication"
            echo "  --quick          Download only small models for quick testing"
            echo "  --help           Show this help message"
            echo ""
            echo "Download modes:"
            echo "  Default: Download all models (requires authentication for some)"
            echo "  --no-auth: Download BERT, GPT-2, RoBERTa models (~7 GB)"
            echo "  --quick: Download small models for testing (~3 GB)"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Function to run download command
run_download() {
    local cmd="python $MODELS_DIR/download_models.py $1"
    if [ -n "$HF_TOKEN" ]; then
        cmd="$cmd --token $HF_TOKEN"
    fi
    
    echo -e "${BLUE}📥 Running: $cmd${NC}"
    eval $cmd
}

# Check current status
echo -e "\n${YELLOW}📊 Current download status:${NC}"
python "$MODELS_DIR/download_models.py" --status 2>/dev/null || echo "Status check failed - continuing anyway"

# Download based on mode
case $MODE in
    "all")
        echo -e "\n${GREEN}🌍 Downloading ALL models for comprehensive testing${NC}"
        echo -e "${YELLOW}⚠️  This will download ~40 GB of models${NC}"
        echo -e "${YELLOW}⚠️  Some models require HuggingFace authentication${NC}"
        
        if [ -z "$HF_TOKEN" ]; then
            echo -e "\n${BLUE}🔐 Authentication will be handled interactively${NC}"
        fi
        
        read -p "Continue? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}Download cancelled${NC}"
            exit 0
        fi
        
        # Download all models
        run_download "--all"
        ;;
        
    "no-auth")
        echo -e "\n${GREEN}🔓 Downloading models that don't require authentication${NC}"
        echo -e "${BLUE}📦 This includes: BERT, RoBERTa, GPT-2 models (~7 GB)${NC}"
        
        run_download "--collection no_auth_required"
        ;;
        
    "quick")
        echo -e "\n${GREEN}⚡ Downloading small models for quick testing${NC}"
        echo -e "${BLUE}📦 This includes: bert-base-uncased, roberta-base, llama-3.2-1b, gpt2 (~3 GB)${NC}"
        
        run_download "--collection small_models"
        ;;
esac

# Show final status
echo -e "\n${GREEN}✅ Download process completed!${NC}"
echo -e "\n${YELLOW}📊 Final status:${NC}"
python "$MODELS_DIR/download_models.py" --status

echo -e "\n${BLUE}🎯 Next steps:${NC}"
echo -e "${BLUE}• Models are ready for use with the unified pipeline${NC}"
echo -e "${BLUE}• Run: python unified_pipeline/run_full_pipeline.py --config configs/full.yaml${NC}"
echo -e "${BLUE}• Or use individual model evaluation scripts${NC}"

echo -e "\n${GREEN}🎉 Ready for bias testing!${NC}"