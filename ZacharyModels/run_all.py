#!/usr/bin/env python3
import os
import json
import enum
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, NamedTuple
from datetime import datetime

from huggingface_hub import HfApi
from ZacharyModels.src.models.registry import (
    ModelStatus,
    ValidationResult,
    validate_model,
    resolve_model_name
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evaluation.log')
    ]
)

logger = logging.getLogger(__name__)

class EvaluationOutcome(enum.Enum):
    """Possible outcomes for model evaluation attempts."""
    COMPLETED = "completed"
    SKIPPED_COMPLETE = "skipped_complete"  # Already evaluated
    SKIPPED_NOT_FOUND = "skipped_not_found"  # Model not found
    SKIPPED_GATED = "skipped_gated"  # License required
    FAILED = "failed"  # Other errors

class ModelResult(NamedTuple):
    """Result of a model evaluation attempt."""
    name: str
    outcome: EvaluationOutcome
    message: str
    canonical_name: Optional[str] = None

def get_result_path(model_name: str, result_type: str = "results") -> Path:
    """Get path for model results/analysis file.
    
    Args:
        model_name: Name of the model
        result_type: Type of result file ("results" or "analysis")
        
    Returns:
        Path: Path to result file
    """
    return Path("ZacharyModels/results") / f"{model_name}_{result_type}.json"

def should_evaluate(model_name: str, resume: bool, overwrite: bool) -> bool:
    """Determine if a model should be evaluated.
    
    Args:
        model_name: Name of the model
        resume: Whether to resume from previous state
        overwrite: Whether to overwrite existing results
        
    Returns:
        bool: True if model should be evaluated
    """
    if overwrite:
        return True
        
    if resume:
        results_path = get_result_path(model_name, "results")
        analysis_path = get_result_path(model_name, "analysis")
        return not (results_path.exists() and analysis_path.exists())
        
    return True

def load_state() -> Dict[str, List[str]]:
    """Load evaluation state from file.
    
    Returns:
        Dict with lists of completed, skipped, and failed models
    """
    try:
        with open('.evaluation_state.json') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            'completed': [],
            'skipped': [],
            'failed': []
        }

def save_state(state: Dict[str, List[str]]) -> None:
    """Save evaluation state to file."""
    with open('.evaluation_state.json', 'w') as f:
        json.dump(state, f, indent=2)

def evaluate_model(model_name: str, model_type: str) -> Optional[str]:
    """Evaluate a single model.
    
    Args:
        model_name: Name of the model
        model_type: Type of model (mlm, llm, etc.)
        
    Returns:
        Optional[str]: Error message if evaluation failed, None if successful
    """
    try:
        cmd = f"python ZacharyModels/evaluate_model.py {model_name} {model_type}"
        result = os.system(cmd)
        if result != 0:
            return f"Command failed with exit code {result}"
        return None
    except Exception as e:
        return str(e)

def main():
    parser = argparse.ArgumentParser(description='Run model evaluations with resume capability')
    parser.add_argument('--resume', action='store_true', default=True,
                       help='Resume from previous state (default: True)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing results')
    parser.add_argument('--hf-token',
                       help='HuggingFace token for accessing gated models')
    parser.add_argument('--include-gated', action='store_true',
                       help='Attempt to evaluate gated models')
    args = parser.parse_args()

    # Load previous state
    state = load_state()
    
    # Define model groups
    model_groups = {
        'mlm': ['bert-base-uncased', 'bert-large-uncased', 'roberta-base'],
        'llm': ['gpt2', 'gpt2-medium'],
        'instruction': [
            ('Qwen2.5-1.5B-Instruct', 'qwen'),
            ('gemma-2-2b-it', 'gemma'),
            ('llama-3.2-1b-it', 'llama')
        ]
    }
    
    # Track statistics by outcome
    stats = {
        'total': 0,
        'completed': 0,
        'skipped_complete': 0,
        'skipped_not_found': 0,
        'skipped_gated': 0,
        'failed': 0
    }
    
    # Track details of each model
    results: List[ModelResult] = []
    
    # Create results directory if needed
    Path("ZacharyModels/results").mkdir(parents=True, exist_ok=True)
    
    # Process MLM and LLM models
    for model_type, models in model_groups.items():
        if model_type in ['mlm', 'llm']:
            for model_name in models:
                stats['total'] += 1
                
                canonical_name = resolve_model_name(model_name)
                stats['total'] += 1
                
                # Check if already completed
                if model_name in state['completed'] and args.resume and not args.overwrite:
                    logger.info(f"Skipping {model_name} (already completed)")
                    results.append(ModelResult(
                        name=model_name,
                        outcome=EvaluationOutcome.SKIPPED_COMPLETE,
                        message="Already evaluated",
                        canonical_name=canonical_name
                    ))
                    stats['skipped_complete'] += 1
                    continue
                
                # Validate model
                validation = validate_model(
                    model_name,
                    token=args.hf_token,
                    retry_with_token=args.include_gated
                )
                
                if validation.status == ModelStatus.NOT_FOUND:
                    logger.warning(f"Skipping {model_name}: {validation.message}")
                    results.append(ModelResult(
                        name=model_name,
                        outcome=EvaluationOutcome.SKIPPED_NOT_FOUND,
                        message=validation.message,
                        canonical_name=canonical_name
                    ))
                    state['skipped'].append(model_name)
                    stats['skipped_not_found'] += 1
                    save_state(state)
                    continue
                elif validation.status == ModelStatus.LICENSE_REQUIRED and not args.include_gated:
                    logger.warning(f"Skipping {model_name}: {validation.message}")
                    results.append(ModelResult(
                        name=model_name,
                        outcome=EvaluationOutcome.SKIPPED_GATED,
                        message=validation.message,
                        canonical_name=canonical_name
                    ))
                    stats['skipped_gated'] += 1
                    continue
                elif validation.status == ModelStatus.ACCESS_DENIED:
                    logger.error(f"Access denied for {model_name}: {validation.message}")
                    results.append(ModelResult(
                        name=model_name,
                        outcome=EvaluationOutcome.FAILED,
                        message=validation.message,
                        canonical_name=canonical_name
                    ))
                    stats['failed'] += 1
                    continue
                    
                logger.info(f"Evaluating {model_name}...")
                error = evaluate_model(model_name, model_type)
                
                if error:
                    logger.error(f"Failed to evaluate {model_name}: {error}")
                    state['failed'].append(model_name)
                    stats['failed'] += 1
                else:
                    logger.info(f"Successfully evaluated {model_name}")
                    state['completed'].append(model_name)
                    stats['completed'] += 1
                
                save_state(state)
    
    # Process instruction models
    for model_name, model_type in model_groups['instruction']:
        stats['total'] += 1
                
        canonical_name = resolve_model_name(model_name)
                
        # Check if already completed
        if model_name in state['completed'] and args.resume and not args.overwrite:
            logger.info(f"Skipping {model_name} (already completed)")
            results.append(ModelResult(
                name=model_name,
                outcome=EvaluationOutcome.SKIPPED_COMPLETE,
                message="Already evaluated",
                canonical_name=canonical_name
            ))
            stats['skipped_complete'] += 1
            continue
                
        # Validate model with token if available
        validation = validate_model(
            canonical_name,  # Use canonical name for validation
            token=args.hf_token,
            retry_with_token=args.include_gated
        )
                
        if validation.status == ModelStatus.NOT_FOUND:
            logger.warning(f"Skipping {model_name}: {validation.message}")
            results.append(ModelResult(
                name=model_name,
                outcome=EvaluationOutcome.SKIPPED_NOT_FOUND,
                message=validation.message,
                canonical_name=canonical_name
            ))
            state['skipped'].append(model_name)
            stats['skipped_not_found'] += 1
            save_state(state)
            continue
        elif validation.status == ModelStatus.LICENSE_REQUIRED and not args.include_gated:
            logger.warning(f"Skipping {model_name}: {validation.message}")
            results.append(ModelResult(
                name=model_name,
                outcome=EvaluationOutcome.SKIPPED_GATED,
                message=validation.message,
                canonical_name=canonical_name
            ))
            stats['skipped_gated'] += 1
            continue
        elif validation.status == ModelStatus.ACCESS_DENIED:
            logger.error(f"Access denied for {model_name}: {validation.message}")
            results.append(ModelResult(
                name=model_name,
                outcome=EvaluationOutcome.FAILED,
                message=validation.message,
                canonical_name=canonical_name
            ))
            stats['failed'] += 1
            continue
        elif validation.status != ModelStatus.VALID:
            logger.error(f"Model validation failed: {validation.message}")
            results.append(ModelResult(
                name=model_name,
                outcome=EvaluationOutcome.FAILED,
                message=validation.message,
                canonical_name=canonical_name
            ))
            stats['failed'] += 1
            continue
            
        # All validation passed, evaluate the model
        logger.info(f"Evaluating {model_name}...")
        error = evaluate_model(model_name, model_type)
        
        if error:
            logger.error(f"Failed to evaluate {model_name}: {error}")
            state['failed'].append(model_name)
            results.append(ModelResult(
                name=model_name,
                outcome=EvaluationOutcome.FAILED,
                message=error,
                canonical_name=canonical_name
            ))
            stats['failed'] += 1
        else:
            logger.info(f"Successfully evaluated {model_name}")
            state['completed'].append(model_name)
            results.append(ModelResult(
                name=model_name,
                outcome=EvaluationOutcome.COMPLETED,
                message="Evaluation successful",
                canonical_name=canonical_name
            ))
            stats['completed'] += 1
        
        save_state(state)
    
    # Print summary
    logger.info("\nEvaluation Summary:")
    logger.info(f"Total models: {stats['total']}")
    logger.info(f"Successfully completed: {stats['completed']}")
    logger.info(f"Skipped (already done): {stats['skipped_complete']}")
    logger.info(f"Skipped (not found): {stats['skipped_not_found']}")
    logger.info(f"Skipped (license required): {stats['skipped_gated']}")
    logger.info(f"Failed: {stats['failed']}")
    
    # Group results by outcome for detailed reporting
    failed = [r for r in results if r.outcome == EvaluationOutcome.FAILED]
    gated = [r for r in results if r.outcome == EvaluationOutcome.SKIPPED_GATED]
    
    if failed:
        logger.info("\nFailed models:")
        for result in failed:
            logger.info(f"  - {result.name}: {result.message}")
            
    if gated:
        logger.info("\nLicense-gated models:")
        logger.info("To enable these models:")
        logger.info("1. Visit https://huggingface.co/models and accept the license")
        logger.info("2. Set HUGGINGFACE_TOKEN environment variable")
        logger.info("3. Run with --include-gated flag")
        for result in gated:
            logger.info(f"  - {result.name} → {result.canonical_name}")

if __name__ == '__main__':
    main()
