"""Claude (Anthropic) LLM provider implementation."""

import os
from typing import Any

import anthropic


class ClaudeProvider:
    """Anthropic Claude provider implementation."""
    
    def __init__(
        self, 
        api_key: str | None = None, 
        model: str = "claude-sonnet-4-20250514"
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key."
            )
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
    
    def complete(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate a completion for the given prompt."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def complete_structured(self, prompt: str, schema: dict) -> dict:
        """Generate a structured response matching the schema.
        
        Uses Claude's tool_use feature for structured output.
        """
        # For Phase 0, we'll parse text responses
        # Phase 1+ could use tool_use for guaranteed structured output
        response = self.complete(prompt, max_tokens=1000)
        
        # Simple key-value parsing for now
        result: dict[str, Any] = {}
        for line in response.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                result[key.strip().lower().replace(" ", "_")] = value.strip()
        
        return result
