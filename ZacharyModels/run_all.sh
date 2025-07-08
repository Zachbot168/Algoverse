#!/bin/bash

# Get HuggingFace token from environment if available
if [ -n "$HUGGINGFACE_TOKEN" ]; then
    TOKEN_ARG="--hf-token $HUGGINGFACE_TOKEN"
else
    TOKEN_ARG=""
fi

# Parse command line arguments
INCLUDE_GATED=""
OVERWRITE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --include-gated)
            INCLUDE_GATED="--include-gated"
            shift
            ;;
        --overwrite)
            OVERWRITE="--overwrite"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--include-gated] [--overwrite]"
            exit 1
            ;;
    esac
done

# Execute Python script with all arguments
python ZacharyModels/run_all.py $TOKEN_ARG $INCLUDE_GATED $OVERWRITE
