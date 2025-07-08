import sys
import os
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, NamedTuple

from huggingface_hub import HfApi
from ZacharyModels.src.models.registry import (
    ModelStatus,
    validate_model,
    resolve_model_name
)
from ZacharyModels.src.data import CrowsPairsDataset
from ZacharyModels.src.models import ModelFactory
from ZacharyModels.src.evaluation import create_scorer, analyze_results
from ZacharyModels.src.prompts.wrapper import PromptWrapper
from ZacharyModels.src.evaluation.metrics import BiasMetrics

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_result_paths(model_name: str) -> tuple[Path, Path]:
    """Get paths for results and analysis files."""
    results = Path(f"ZacharyModels/results/{model_name}_results.json")
    analysis = Path(f"ZacharyModels/results/{model_name}_analysis.json")
    return results, analysis

def should_evaluate(model_name: str, resume: bool, overwrite: bool) -> bool:
    """Check if model should be evaluated based on flags and existing results."""
    if overwrite:
        return True
    
    if resume:
        results_path, analysis_path = get_result_paths(model_name)
        return not (results_path.exists() and analysis_path.exists())
        
    return True

def save_results(
    results: Dict[str, Any],
    analysis: Dict[str, Any],
    model_name: str
) -> None:
    """Save evaluation results and analysis."""
    results_path, analysis_path = get_result_paths(model_name)
    
    # Create results directory if needed
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save analysis
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Evaluate model bias')
    parser.add_argument('model_name', help='Name of the model to evaluate')
    parser.add_argument('model_type', help='Type of model (mlm, llm, etc.)')
    parser.add_argument('--resume', action='store_true', default=True,
                       help='Skip if results exist (default: True)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing results')
    parser.add_argument('--hf-token', 
                       help='HuggingFace token for accessing gated models')
    parser.add_argument('--include-gated', action='store_true',
                       help='Attempt to evaluate gated models')
    args = parser.parse_args()
    
    model_name = args.model_name
    model_type = args.model_type
    
    try:
        # Get token from args or environment
        hf_token = args.hf_token or os.getenv("HUGGINGFACE_TOKEN")
        
        # Get canonical model name
        canonical_name = resolve_model_name(model_name)
        if canonical_name != model_name:
            logger.info(f"Resolved model name: {model_name} → {canonical_name}")
            
        # Validate model early to fail fast
        validation = validate_model(
            canonical_name,  # Use canonical name for validation 
            token=hf_token,
            retry_with_token=args.include_gated
        )
        
        if validation.status != ModelStatus.VALID:
            if validation.status == ModelStatus.LICENSE_REQUIRED:
                if not args.include_gated:
                    logger.error(
                        f"Model {canonical_name} requires license acceptance and --include-gated flag. "
                        "Visit https://huggingface.co/{canonical_name} to accept the license."
                    )
                    return 1
                elif not hf_token:
                    logger.error(
                        f"Model {canonical_name} requires authentication. Please provide a token "
                        "via --hf-token or HUGGINGFACE_TOKEN environment variable."
                    )
                    return 1
            elif validation.status == ModelStatus.ACCESS_DENIED:
                logger.error(f"Access denied: {validation.message}")
                return 1
            else:
                logger.error(f"Model validation failed: {validation.message}")
                return 1
        
        # Check if evaluation needed
        if not should_evaluate(model_name, args.resume, args.overwrite):
            logger.info(f"Skipping {model_name} (already evaluated)")
            return 0
            
        # Load dataset
        logger.info(f"Loading dataset")
        dataset = CrowsPairsDataset(
            data_dir="datasets/crows-pairs",
            cache_dir="ZacharyModels/data/cache"
        )
        dataset.load()
        
        # Initialize model with token if needed
        logger.info(f"Initializing {canonical_name}")
        model = ModelFactory.create(canonical_name)
        # Pass token for gated models
        if validation.status == ModelStatus.VALID and args.include_gated:
            model.load(token=hf_token)
        else:
            model.load()
        
        # Create scorer with error handling for templates
        logger.info("Setting up evaluation")
        try:
            wrapper = None if model_type == "mlm" else PromptWrapper(
                model_type=model_type,
                template_name="bias_analysis"
            )
            scorer = create_scorer(model=model, prompt_wrapper=wrapper)
        except ValueError as e:
            logger.warning(f"Template error: {str(e)}. Using default template.")
            # Fall back to basic template
            wrapper = None if model_type == "mlm" else PromptWrapper(
                model_type="llm",  # Fall back to general LLM templates
                template_name="basic"
            )
            scorer = create_scorer(model=model, prompt_wrapper=wrapper)
            
        # Run evaluation
        logger.info(f"Evaluating {model_name}")
        results = scorer.score_dataset(dataset)
        analysis = analyze_results(results)
        
        # Save results
        logger.info(f"Saving results for {model_name}")
        save_results(results, analysis, model_name)
        
        return 0
            
    except Exception as e:
        logger.error(f"Evaluation failed for {model_name}: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
