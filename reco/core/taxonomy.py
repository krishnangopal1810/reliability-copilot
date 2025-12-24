"""Failure taxonomy for consistent classification.

Provides a two-level taxonomy:
1. Universal categories (built-in) - common LLM failure modes
2. Domain categories (user config) - domain-specific extensions
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Category:
    """A failure category in the taxonomy."""
    name: str
    description: str
    is_universal: bool = True


# Universal taxonomy - applies to all LLM systems
UNIVERSAL_TAXONOMY: dict[str, str] = {
    "Hallucination": "Fabricates information not present in context or training",
    "Factual Error": "States incorrect facts that could be verified",
    "Format Violation": "Ignores required output format (JSON, markdown, etc.)",
    "Instruction Ignored": "Fails to follow explicit user instructions",
    "Incomplete Response": "Response is cut off or missing required parts",
    "Safety Violation": "Produces harmful, biased, or inappropriate content",
    "Reasoning Breakdown": "Logical errors in multi-step reasoning chains",
    "Tool Execution Error": "Incorrectly uses tools, APIs, or function calls",
    "Context Overflow": "Loses or confuses information from long context",
    "Refusal Error": "Incorrectly refuses a safe request",
}


def load_taxonomy(config_path: Optional[Path] = None) -> list[Category]:
    """Load combined universal + domain taxonomy.
    
    Args:
        config_path: Path to domain taxonomy YAML. Defaults to .reco/taxonomy.yaml
        
    Returns:
        List of Category objects (universal first, then domain)
    """
    categories = []
    
    # Add universal categories
    for name, description in UNIVERSAL_TAXONOMY.items():
        categories.append(Category(
            name=name,
            description=description,
            is_universal=True,
        ))
    
    # Load domain categories from config
    if config_path is None:
        config_path = Path(".reco") / "taxonomy.yaml"
    
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            domain_categories = config.get("domain_categories", [])
            for cat in domain_categories:
                categories.append(Category(
                    name=cat.get("name", "Unknown"),
                    description=cat.get("description", ""),
                    is_universal=False,
                ))
        except (yaml.YAMLError, IOError):
            pass  # Silently ignore invalid config
    
    return categories


def format_taxonomy_for_prompt(categories: list[Category]) -> str:
    """Format taxonomy for LLM classification prompt.
    
    Returns:
        Formatted string with numbered categories
    """
    lines = []
    for i, cat in enumerate(categories, 1):
        lines.append(f"{i}. {cat.name}: {cat.description}")
    return "\n".join(lines)


def get_category_names(categories: list[Category]) -> list[str]:
    """Get just the category names."""
    return [cat.name for cat in categories]
