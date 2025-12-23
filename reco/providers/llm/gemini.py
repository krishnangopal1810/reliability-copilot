"""Google Gemini LLM provider implementation."""

import os
from typing import Any

import google.generativeai as genai


class GeminiProvider:
    """Google Gemini provider implementation."""
    
    def __init__(
        self, 
        api_key: str | None = None, 
        model: str = "gemini-2.0-flash"
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY or GOOGLE_API_KEY env var, or pass api_key."
            )
        genai.configure(api_key=self.api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(model)
    
    def complete(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate a completion for the given prompt."""
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
            )
        )
        return response.text
    
    def complete_structured(self, prompt: str, schema: dict) -> dict:
        """Generate a structured response matching the schema.
        
        For Phase 0, we parse text responses.
        Phase 1+ could use Gemini's JSON mode for guaranteed structured output.
        """
        response = self.complete(prompt, max_tokens=1000)
        
        # Simple key-value parsing
        result: dict[str, Any] = {}
        for line in response.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                result[key.strip().lower().replace(" ", "_")] = value.strip()
        
        return result
