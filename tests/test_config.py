"""Exhaustive unit tests for reco.config."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch

from reco.config import Config


class TestConfigDefaults:
    """Tests for Config default values."""
    
    def test_default_llm_provider(self):
        """Test default LLM provider is OpenRouter."""
        config = Config()
        assert config.llm_provider == "openrouter"
    
    def test_default_embedding_model(self):
        """Test default embedding model."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.embedding_model == "all-MiniLM-L6-v2"
    
    def test_default_min_cluster_size(self):
        """Test default min cluster size."""
        config = Config()
        assert config.min_cluster_size == 2
    
    def test_default_storage_backend(self):
        """Test default storage backend is memory."""
        config = Config()
        assert config.storage_backend == "memory"
    
    def test_default_storage_path(self):
        """Test default storage path is project-relative."""
        config = Config()
        assert config.storage_path == Path(".reco") / "data.db"


class TestConfigEnvironmentVariables:
    """Tests for Config environment variable loading."""
    
    def test_openrouter_api_key_from_env(self):
        """Test loading OpenRouter API key from environment."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-123"}, clear=True):
            config = Config()
            assert config.openrouter_api_key == "test-key-123"
    
    def test_openrouter_api_key_empty_when_not_set(self):
        """Test API key is empty when not set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENROUTER_API_KEY", None)
            config = Config()
            assert config.openrouter_api_key == ""
    
    def test_llm_model_from_env(self):
        """Test loading LLM model from environment."""
        with patch.dict(os.environ, {"RECO_LLM_MODEL": "openai/gpt-4o"}):
            config = Config()
            assert config.llm_model == "openai/gpt-4o"
    
    def test_llm_model_default_when_not_set(self):
        """Test LLM model default when env not set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("RECO_LLM_MODEL", None)
            config = Config()
            # Default is xiaomi/mimo-v2-flash:free
            assert "xiaomi" in config.llm_model.lower() or "mimo" in config.llm_model.lower()
    
    def test_embedding_model_from_env(self):
        """Test loading embedding model from environment."""
        with patch.dict(os.environ, {"RECO_EMBEDDING_MODEL": "all-mpnet-base-v2"}):
            config = Config()
            assert config.embedding_model == "all-mpnet-base-v2"


class TestConfigLoad:
    """Tests for Config.load() class method."""
    
    def test_load_returns_config(self):
        """Test that load returns a Config instance."""
        config = Config.load()
        assert isinstance(config, Config)
    
    def test_load_picks_up_env_vars(self):
        """Test that load picks up environment variables."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "loaded-key"}, clear=True):
            config = Config.load()
            assert config.openrouter_api_key == "loaded-key"


class TestConfigValidation:
    """Tests for Config.validate() method."""
    
    def test_validate_no_errors_with_api_key(self):
        """Test validation passes when API key is set."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "valid-key"}, clear=True):
            config = Config.load()
            errors = config.validate()
            
            # Should have no errors about missing key
            key_errors = [e for e in errors if "OPENROUTER_API_KEY" in e]
            assert len(key_errors) == 0
    
    def test_validate_error_when_api_key_missing(self):
        """Test validation fails when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENROUTER_API_KEY", None)
            config = Config()
            config.openrouter_api_key = ""  # Ensure it's empty
            
            errors = config.validate()
            
            assert len(errors) > 0
            assert any("OPENROUTER_API_KEY" in e for e in errors)
    
    def test_validate_returns_list(self):
        """Test that validate always returns a list."""
        config = Config()
        errors = config.validate()
        assert isinstance(errors, list)
    
    def test_validate_error_message_includes_fix_hint(self):
        """Test that validation error includes hint for fixing."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENROUTER_API_KEY", None)
            config = Config()
            config.openrouter_api_key = ""
            
            errors = config.validate()
            
            # Should include a hint about how to fix
            assert any("export" in e.lower() for e in errors)


class TestConfigCustomValues:
    """Tests for Config with custom values."""
    
    def test_config_with_all_custom_values(self):
        """Test creating config with all custom values."""
        config = Config(
            llm_provider="openrouter",
            openrouter_api_key="custom-openrouter",
            llm_model="openai/gpt-4o",
            embedding_model="custom-embeddings",
            min_cluster_size=5,
            storage_backend="sqlite",
            storage_path=Path("/custom/path/data.db"),
        )
        
        assert config.llm_provider == "openrouter"
        assert config.openrouter_api_key == "custom-openrouter"
        assert config.llm_model == "openai/gpt-4o"
        assert config.embedding_model == "custom-embeddings"
        assert config.min_cluster_size == 5
        assert config.storage_backend == "sqlite"
        assert config.storage_path == Path("/custom/path/data.db")
    
    def test_config_partial_override(self):
        """Test config with partial custom values."""
        config = Config(min_cluster_size=10)
        
        assert config.min_cluster_size == 10
        assert config.llm_provider == "openrouter"  # Default preserved
