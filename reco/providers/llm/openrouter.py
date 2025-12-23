"""OpenRouter LLM provider implementation using OpenAI-compatible API."""

import os
from typing import Any

from openai import OpenAI


class OpenRouterProvider:
    """OpenRouter provider using OpenAI-compatible API."""
    
    def __init__(
        self, 
        api_key: str | None = None, 
        model: str = "xiaomi/mimo-v2-flash:free"
    ):
        """Initialize OpenRouter provider.
        
        Args:
            api_key: OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.
            model: Model identifier (e.g., 'anthropic/claude-3.5-sonnet', 
                   'openai/gpt-4o', 'google/gemini-2.0-flash-exp')
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var, or pass api_key."
            )
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        self.model = model
    
    def complete(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate a completion for the given prompt."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    
    def complete_structured(self, prompt: str, schema: dict) -> dict:
        """Generate a structured response matching the schema.
        
        For Phase 0, we parse text responses.
        Phase 1+ could use JSON mode for guaranteed structured output.
        """
        response = self.complete(prompt, max_tokens=1000)
        
        # Simple key-value parsing
        result: dict[str, Any] = {}
        for line in response.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                result[key.strip().lower().replace(" ", "_")] = value.strip()
        
        return result
