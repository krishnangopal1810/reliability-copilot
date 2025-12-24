"""Exhaustive unit tests for reco.providers.llm.openrouter."""

import pytest
from unittest.mock import patch, MagicMock

from reco.providers.llm.openrouter import OpenRouterProvider


class TestOpenRouterProviderInit:
    """Tests for OpenRouterProvider initialization."""
    
    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        with patch("reco.providers.llm.openrouter.OpenAI") as mock_openai:
            provider = OpenRouterProvider(api_key="test-key")
            assert provider.api_key == "test-key"
    
    def test_init_from_env(self):
        """Test initialization from OPENROUTER_API_KEY environment variable."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "env-key"}, clear=True):
            with patch("reco.providers.llm.openrouter.OpenAI"):
                provider = OpenRouterProvider()
                assert provider.api_key == "env-key"
    
    def test_init_raises_without_key(self):
        """Test that initialization raises without API key."""
        with patch.dict("os.environ", {}, clear=True):
            import os
            os.environ.pop("OPENROUTER_API_KEY", None)
            
            with pytest.raises(ValueError) as exc_info:
                OpenRouterProvider(api_key=None)
            
            assert "API key required" in str(exc_info.value)
    
    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        with patch("reco.providers.llm.openrouter.OpenAI"):
            provider = OpenRouterProvider(api_key="key", model="openai/gpt-4o")
            assert provider.model == "openai/gpt-4o"
    
    def test_init_default_model(self):
        """Test default model is set."""
        with patch("reco.providers.llm.openrouter.OpenAI"):
            provider = OpenRouterProvider(api_key="key")
            # Default is xiaomi/mimo-v2-flash:free
            assert "xiaomi" in provider.model.lower() or "mimo" in provider.model.lower()
    
    def test_init_uses_openrouter_base_url(self):
        """Test that OpenRouter base URL is used."""
        with patch("reco.providers.llm.openrouter.OpenAI") as mock_openai:
            OpenRouterProvider(api_key="key")
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"


class TestOpenRouterProviderComplete:
    """Tests for OpenRouterProvider.complete method."""
    
    @pytest.fixture
    def mock_provider(self):
        """Create a provider with mocked client."""
        with patch("reco.providers.llm.openrouter.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            
            provider = OpenRouterProvider(api_key="test-key")
            
            # Setup default response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Mocked response"
            mock_client.chat.completions.create.return_value = mock_response
            
            yield provider, mock_client
    
    def test_complete_returns_string(self, mock_provider):
        """Test that complete returns a string."""
        provider, mock_client = mock_provider
        
        result = provider.complete("Test prompt")
        
        assert isinstance(result, str)
    
    def test_complete_calls_api(self, mock_provider):
        """Test that complete calls the OpenAI API."""
        provider, mock_client = mock_provider
        
        provider.complete("Test prompt")
        
        mock_client.chat.completions.create.assert_called_once()
    
    def test_complete_passes_prompt(self, mock_provider):
        """Test that prompt is passed to API."""
        provider, mock_client = mock_provider
        
        provider.complete("My specific prompt")
        
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert any("My specific prompt" in m["content"] for m in messages)
    
    def test_complete_uses_max_tokens(self, mock_provider):
        """Test that max_tokens is passed to API."""
        provider, mock_client = mock_provider
        
        provider.complete("Prompt", max_tokens=500)
        
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 500


class TestOpenRouterProviderCompleteStructured:
    """Tests for OpenRouterProvider.complete_structured method."""
    
    @pytest.fixture
    def mock_provider(self):
        """Create a provider with mocked client."""
        with patch("reco.providers.llm.openrouter.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            
            provider = OpenRouterProvider(api_key="test-key")
            
            # Setup response with key-value format
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "KEY: value\nANOTHER: data"
            mock_client.chat.completions.create.return_value = mock_response
            
            yield provider, mock_client
    
    def test_complete_structured_returns_dict(self, mock_provider):
        """Test that complete_structured returns a dict."""
        provider, _ = mock_provider
        
        result = provider.complete_structured("Prompt", {"key": "string"})
        
        assert isinstance(result, dict)
    
    def test_complete_structured_parses_key_value(self, mock_provider):
        """Test that key-value pairs are parsed."""
        provider, mock_client = mock_provider
        
        # Setup response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "NAME: John\nAGE: 30"
        mock_client.chat.completions.create.return_value = mock_response
        
        result = provider.complete_structured("Prompt", {})
        
        assert "name" in result
        assert result["name"] == "John"
