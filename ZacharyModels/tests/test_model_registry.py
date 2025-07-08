"""Tests for model registry and validation."""
import os
import pytest
from unittest.mock import MagicMock, patch

from huggingface_hub import ModelInfo
from huggingface_hub.utils import HfHubHTTPError

from ZacharyModels.src.models.registry import (
    ModelStatus,
    ValidationResult,
    validate_model,
    resolve_model_name,
    is_gated_model
)

def test_model_name_resolution():
    """Test model alias resolution."""
    assert resolve_model_name("bert-base-uncased") == "bert-base-uncased"
    assert resolve_model_name("gemma-2-2b-it") == "google/gemma-2b-it"
    assert resolve_model_name("Qwen2.5-1.5B-Instruct") == "Qwen/Qwen2.5-1.5B-Instruct"
    assert resolve_model_name("llama-3.2-1b-it") == "meta-llama/Llama-3.2-1B-Instruct"

def test_gated_model_detection():
    """Test gated model detection."""
    assert is_gated_model("google/gemma-2b-it")
    assert is_gated_model("gemma-2-2b-it")  # Should work with alias
    assert is_gated_model("Qwen/Qwen2.5-1.5B-Instruct")
    assert not is_gated_model("bert-base-uncased")
    assert not is_gated_model("gpt2")

@pytest.fixture
def mock_model_info():
    """Mock ModelInfo response."""
    return ModelInfo(
        modelId="test/model",
        sha="abc123",
        lastModified="2025-07-07",
        tags=[],
        pipeline_tags=[],
        siblings=[]
    )

@pytest.fixture
def mock_401_error():
    """Mock 401 unauthorized error."""
    response = MagicMock()
    response.status_code = 401
    return HfHubHTTPError("Unauthorized", response)

@pytest.fixture
def mock_404_error():
    """Mock 404 not found error."""
    response = MagicMock()
    response.status_code = 404
    return HfHubHTTPError("Not Found", response)

def test_validate_public_model(mock_model_info):
    """Test validation of public model."""
    with patch("huggingface_hub.HfApi.model_info", return_value=mock_model_info):
        result = validate_model("bert-base-uncased")
        assert result.status == ModelStatus.VALID
        assert result.model_info is not None

def test_validate_not_found_model(mock_404_error):
    """Test validation of non-existent model."""
    with patch("huggingface_hub.HfApi.model_info", side_effect=mock_404_error):
        result = validate_model("nonexistent-model")
        assert result.status == ModelStatus.NOT_FOUND

def test_validate_gated_model_without_token(mock_401_error):
    """Test validation of gated model without token."""
    with patch("huggingface_hub.HfApi.model_info", side_effect=mock_401_error):
        result = validate_model("google/gemma-2b-it")
        assert result.status == ModelStatus.LICENSE_REQUIRED
        assert "requires license acceptance" in result.message

def test_validate_gated_model_with_token(mock_model_info, mock_401_error):
    """Test validation of gated model with token retry."""
    def mock_info(*args, **kwargs):
        if kwargs.get("token") == "valid_token":
            return mock_model_info
        raise mock_401_error

    with patch("huggingface_hub.HfApi.model_info", side_effect=mock_info):
        # Should fail without token
        result = validate_model("google/gemma-2b-it")
        assert result.status == ModelStatus.LICENSE_REQUIRED

        # Should succeed with token
        result = validate_model("google/gemma-2b-it", token="valid_token")
        assert result.status == ModelStatus.VALID
        assert result.model_info is not None

def test_validate_with_environment_token(mock_model_info, mock_401_error):
    """Test validation using token from environment."""
    def mock_info(*args, **kwargs):
        if kwargs.get("token") == "env_token":
            return mock_model_info
        raise mock_401_error

    with patch("huggingface_hub.HfApi.model_info", side_effect=mock_info):
        with patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "env_token"}):
            result = validate_model(
                "google/gemma-2b-it",
                retry_with_token=True
            )
            assert result.status == ModelStatus.VALID
            assert result.model_info is not None
