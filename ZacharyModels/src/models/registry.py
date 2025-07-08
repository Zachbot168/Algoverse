"""Model registry and validation utilities."""
import os
import enum
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

from huggingface_hub import HfApi, ModelInfo
from huggingface_hub.utils import HfHubHTTPError
import logging

logger = logging.getLogger(__name__)

class ModelStatus(enum.Enum):
    """Status codes for model validation."""
    VALID = "valid"
    NOT_FOUND = "not_found"
    LICENSE_REQUIRED = "license_required"
    ACCESS_DENIED = "access_denied"

@dataclass
class ValidationResult:
    """Result of model validation."""
    status: ModelStatus
    message: str
    model_info: Optional[ModelInfo] = None

# Mapping of shorthand names to full HuggingFace repo IDs
MODEL_ALIASES: Dict[str, str] = {
    "gemma-2-2b-it": "google/gemma-2b-it",
    "llama-3.2-1b-it": "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct"
}

# Known gated models requiring license acceptance
GATED_MODELS: List[str] = [
    "google/gemma-2b-it",
    "google/gemma-7b-it",
    "meta-llama/Llama-2-7b-hf",
    "Qwen/Qwen2.5-1.5B-Instruct",
]

def resolve_model_name(name: str) -> str:
    """Get canonical model name from possible alias.
    
    Args:
        name: Model name or alias
        
    Returns:
        str: Canonical model name
    """
    return MODEL_ALIASES.get(name, name)

def is_gated_model(name: str) -> bool:
    """Check if a model requires license acceptance.
    
    Args:
        name: Model name (canonical form)
        
    Returns:
        bool: True if model is gated
    """
    canonical_name = resolve_model_name(name)
    return any(
        canonical_name.startswith(gated) 
        for gated in GATED_MODELS
    )

def validate_model(
    name: str,
    token: Optional[str] = None,
    retry_with_token: bool = True
) -> ValidationResult:
    """Validate model existence and accessibility.
    
    Args:
        name: Model name/ID
        token: Optional HuggingFace token
        retry_with_token: Whether to retry with token on 401/403
        
    Returns:
        ValidationResult: Validation status and details
    """
    canonical_name = resolve_model_name(name)
    api = HfApi(token=token)
    
    def _try_validate(using_token: bool = False) -> ValidationResult:
        try:
            info = api.model_info(canonical_name)
            return ValidationResult(
                status=ModelStatus.VALID,
                message=f"Model {canonical_name} is valid and accessible",
                model_info=info
            )
        except HfHubHTTPError as e:
            if e.response.status_code in (401, 403):
                if is_gated_model(canonical_name):
                    return ValidationResult(
                        status=ModelStatus.LICENSE_REQUIRED,
                        message=(
                            f"Model {canonical_name} requires license acceptance. "
                            "Visit https://huggingface.co/{canonical_name} "
                            "and click 'Agree' to access."
                        )
                    )
                if not using_token:
                    return ValidationResult(
                        status=ModelStatus.ACCESS_DENIED,
                        message=(
                            f"Model {canonical_name} requires authentication. "
                            "Please provide a HuggingFace token."
                        )
                    )
            return ValidationResult(
                status=ModelStatus.NOT_FOUND,
                message=f"Model {canonical_name} not found: {str(e)}"
            )
        except Exception as e:
            return ValidationResult(
                status=ModelStatus.NOT_FOUND,
                message=f"Error validating {canonical_name}: {str(e)}"
            )
    
    # First try without token
    if token is None:
        result = _try_validate()
        if result.status in (ModelStatus.ACCESS_DENIED, ModelStatus.LICENSE_REQUIRED):
            # Try with token from environment if available
            env_token = os.getenv("HUGGINGFACE_TOKEN")
            if env_token and retry_with_token:
                logger.info(f"Retrying {canonical_name} with token from environment")
                api = HfApi(token=env_token)
                return _try_validate(using_token=True)
        return result
    
    # Try directly with provided token
    return _try_validate(using_token=True)

def get_model_type(name: str) -> str:
    """Determine model type from name.
    
    Args:
        name: Model name or alias
        
    Returns:
        str: Model type identifier
    """
    canonical_name = resolve_model_name(name).lower()
    
    if "bert" in canonical_name or "roberta" in canonical_name:
        return "mlm"
    elif "gpt" in canonical_name:
        return "llm"
    elif "qwen" in canonical_name:
        return "qwen"
    elif "gemma" in canonical_name:
        return "gemma"
    elif "llama" in canonical_name:
        return "llama"
    else:
        return "llm"  # Default to LLM for unknown models
