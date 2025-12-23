"""Base LLM provider protocol."""

from abc import abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers. Swap implementations without changing core logic."""
    
    @abstractmethod
    def complete(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate a completion for the given prompt."""
        ...
    
    @abstractmethod
    def complete_structured(self, prompt: str, schema: dict) -> dict:
        """Generate a structured response matching the schema."""
        ...
