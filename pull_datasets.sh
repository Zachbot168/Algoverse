#!/bin/bash

# Dataset Download Script for Algoverse FIRM Framework
# Downloads bias evaluation datasets for research

set -e  # Exit on any error

echo "Algoverse Dataset Download"
echo "=========================="

# Create datasets directory
mkdir -p datasets
cd datasets

echo ""
echo "Downloading bias evaluation datasets..."

# 1. CrowS-Pairs - Stereotype Evaluation
echo ""
echo "[1/9] Downloading CrowS-Pairs dataset..."
if [ ! -d "crows-pairs" ]; then
    git clone https://github.com/nyu-mll/crows-pairs.git
    echo "  CrowS-Pairs downloaded"
else
    echo "  CrowS-Pairs already exists"
fi

# 2. WinoBias - Gender bias in coreference
echo ""
echo "[2/9] Downloading WinoBias dataset..."
if [ ! -d "winobias" ]; then
    git clone https://github.com/uclanlp/corefBias.git winobias
    echo "  WinoBias downloaded"
else
    echo "  WinoBias already exists"
fi

# 3. WinoGender - Gender coreference resolution
echo ""
echo "[3/9] Downloading WinoGender dataset..."
if [ ! -d "winogender" ]; then
    git clone https://github.com/rudinger/winogender-schemas.git winogender
    echo "  WinoGender downloaded"
else
    echo "  WinoGender already exists"
fi

# 4. BBQ - Bias Benchmark for QA
echo ""
echo "[4/9] Downloading BBQ dataset..."
mkdir -p bbq
cd bbq
if [ ! -f "BBQ.csv" ]; then
    wget -q https://github.com/nyu-mll/bbq/raw/main/data/BBQ.csv || echo "  Warning: BBQ download failed - may need manual download"
    if [ -f "BBQ.csv" ]; then
        echo "  BBQ downloaded"
    else
        echo "  BBQ download failed"
    fi
else
    echo "  BBQ already exists"
fi
cd ..

# 5. StereoSet - Stereotype evaluation
echo ""
echo "[5/9] Downloading StereoSet dataset..."
if [ ! -d "stereoset" ]; then
    git clone https://github.com/moinnadeem/StereoSet.git stereoset
    echo "  StereoSet downloaded"
else
    echo "  StereoSet already exists"
fi

# 6. BOLD - Bias in Open-ended Language Generation
echo ""
echo "[6/9] Downloading BOLD dataset..."
mkdir -p bold
cd bold
if [ ! -f "prompts.json" ]; then
    wget -q https://github.com/amazon-science/bold/raw/main/prompts/wikipedia_prompts_toxic.json -O prompts.json || \
    wget -q https://raw.githubusercontent.com/amazon-science/bold/main/prompts/prompts.json -O prompts.json || \
    echo "  Warning: BOLD download failed - may need manual download"

    if [ -f "prompts.json" ]; then
        echo "  BOLD downloaded"
    else
        echo "  BOLD download failed"
    fi
else
    echo "  BOLD already exists"
fi
cd ..

# 7. TruthfulQA - Truthfulness evaluation
echo ""
echo "[7/9] Downloading TruthfulQA dataset..."
mkdir -p truthfulqa
cd truthfulqa
if [ ! -f "TruthfulQA.csv" ]; then
    wget -q https://github.com/sylinrl/TruthfulQA/raw/main/TruthfulQA.csv || echo "  Warning: TruthfulQA download failed"
    if [ -f "TruthfulQA.csv" ]; then
        echo "  TruthfulQA downloaded"
    else
        echo "  TruthfulQA download failed"
    fi
else
    echo "  TruthfulQA already exists"
fi
cd ..

# 8. Bias in Bios - Occupational bias
echo ""
echo "[8/9] Downloading BiosBias dataset..."
if [ ! -d "biosbias" ]; then
    mkdir -p biosbias
    echo "# BiosBias dataset requires special access" > biosbias/README.md
    echo "  Warning: BiosBias requires manual download from original source"
else
    echo "  BiosBias directory exists"
fi

# 9. Create SEAT/WEAT test data
echo ""
echo "[9/9] Creating SEAT test structure..."
if [ ! -d "seat" ]; then
    mkdir -p seat
    echo "# SEAT tests - implement based on Caliskan et al. 2017" > seat/README.md
    echo "  Warning: SEAT tests need manual implementation"
else
    echo "  SEAT directory exists"
fi

# Return to root directory
cd ..

echo ""
echo "Dataset Download Summary"
echo "========================"
echo "Checking downloaded datasets..."

# Verification function
check_dataset() {
    local name=$1
    local path=$2
    local file_check=$3

    if [ -d "$path" ] && [ -n "$file_check" ] && [ -f "$path/$file_check" ]; then
        echo "  [OK] $name: Ready"
        return 0
    elif [ -d "$path" ]; then
        echo "  [--] $name: Directory exists but may be incomplete"
        return 1
    else
        echo "  [XX] $name: Missing"
        return 1
    fi
}

# Check each dataset
ready_count=0
total_count=8

check_dataset "CrowS-Pairs" "datasets/crows-pairs" "data/crows_pairs_anonymized.csv" && ((ready_count++))
check_dataset "WinoBias" "datasets/winobias" "wino/data/anti_stereotyped_type1.txt.dev" && ((ready_count++))
check_dataset "WinoGender" "datasets/winogender" "data/templates.tsv" && ((ready_count++))
check_dataset "BBQ" "datasets/bbq" "BBQ.csv" && ((ready_count++))
check_dataset "StereoSet" "datasets/stereoset" "data/bias-bench/stereoset/test.json" && ((ready_count++))
check_dataset "BOLD" "datasets/bold" "prompts.json" && ((ready_count++))
check_dataset "TruthfulQA" "datasets/truthfulqa" "TruthfulQA.csv" && ((ready_count++))
check_dataset "BiosBias" "datasets/biosbias" "" && ((ready_count++))

echo ""
echo "Summary: $ready_count/$total_count datasets ready"

if [ $ready_count -eq $total_count ]; then
    echo "All datasets successfully downloaded!"
    echo ""
    echo "Next steps:"
    echo "1. cd unified_pipeline"
    echo "2. python test_installation.py"
    echo "3. python run_unified_pipeline.py --model-config configs/models/gemma-2-2b-it.yaml --suite quick_evaluation"
elif [ $ready_count -gt $((total_count / 2)) ]; then
    echo "Most datasets ready - you can start testing!"
    echo ""
    echo "To test with available datasets:"
    echo "cd unified_pipeline && python run_unified_pipeline.py --model-config configs/models/gemma-2-2b-it.yaml --suite quick_evaluation"
else
    echo "Limited datasets available. You may need to:"
    echo "1. Check internet connection"
    echo "2. Manually download missing datasets"
    echo "3. Verify repository URLs are still valid"
fi

echo ""
echo "For manual downloads, check:"
echo "- BBQ: https://github.com/nyu-mll/bbq"
echo "- BOLD: https://github.com/amazon-science/bold"
echo "- TruthfulQA: https://github.com/sylinrl/TruthfulQA"
echo "- BiosBias: https://github.com/microsoft/biosbias"
