"""Configuration management."""

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class Config:
    """Global configuration, loaded from env vars.
    
    Phase 0: Environment variables only.
    Phase 1+: Add config file support.
    """
    
    # LLM settings
    llm_provider: str = "openrouter"
    openrouter_api_key: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("RECO_LLM_MODEL", "xiaomi/mimo-v2-flash:free")
    )
    
    # Embedding settings
    embedding_model: str = field(
        default_factory=lambda: os.getenv("RECO_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    
    # Clustering settings  
    min_cluster_size: int = 2
    
    # Storage settings (Phase 1+)
    storage_backend: str = "memory"  # memory, sqlite, postgres
    storage_path: Path = field(
        default_factory=lambda: Path.home() / ".reco" / "data.db"
    )
    
    @classmethod
    def load(cls) -> "Config":
        """Load config from environment.
        
        Phase 1+: Also load from ~/.reco/config.toml if exists.
        """
        return cls()
    
    def validate(self) -> list[str]:
        """Validate configuration, return list of errors."""
        errors = []
        
        if self.llm_provider == "openrouter" and not self.openrouter_api_key:
            errors.append(
                "OPENROUTER_API_KEY not set. Run: export OPENROUTER_API_KEY=your-key"
            )
        
        return errors
