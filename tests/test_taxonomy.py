"""Tests for the failure taxonomy system."""

import pytest
import tempfile
from pathlib import Path

from reco.core.taxonomy import (
    UNIVERSAL_TAXONOMY,
    Category,
    load_taxonomy,
    format_taxonomy_for_prompt,
    get_category_names,
)


class TestUniversalTaxonomy:
    """Tests for the built-in universal taxonomy."""
    
    def test_universal_taxonomy_has_categories(self):
        """Test that universal taxonomy is populated."""
        assert len(UNIVERSAL_TAXONOMY) > 0
    
    def test_universal_taxonomy_has_descriptions(self):
        """Test that all categories have descriptions."""
        for name, description in UNIVERSAL_TAXONOMY.items():
            assert name, "Category name should not be empty"
            assert description, f"Category '{name}' should have a description"
    
    def test_universal_taxonomy_expected_categories(self):
        """Test that key categories exist."""
        expected = ["Hallucination", "Factual Error", "Format Violation"]
        for category in expected:
            assert category in UNIVERSAL_TAXONOMY


class TestLoadTaxonomy:
    """Tests for taxonomy loading."""
    
    def test_load_taxonomy_returns_categories(self):
        """Test that load_taxonomy returns Category objects."""
        categories = load_taxonomy()
        
        assert len(categories) > 0
        assert all(isinstance(c, Category) for c in categories)
    
    def test_load_taxonomy_includes_universal(self):
        """Test that universal categories are included."""
        categories = load_taxonomy()
        names = [c.name for c in categories]
        
        assert "Hallucination" in names
        assert "Format Violation" in names
    
    def test_load_taxonomy_marks_universal(self):
        """Test that universal categories are marked as such."""
        categories = load_taxonomy()
        
        universal = [c for c in categories if c.is_universal]
        assert len(universal) == len(UNIVERSAL_TAXONOMY)
    
    def test_load_taxonomy_with_domain_config(self):
        """Test loading domain-specific categories from config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "taxonomy.yaml"
            config_path.write_text("""
domain_categories:
  - name: Custom Category
    description: A custom domain-specific category
  - name: Another Custom
    description: Another one
""")
            
            categories = load_taxonomy(config_path)
            names = [c.name for c in categories]
            
            assert "Custom Category" in names
            assert "Another Custom" in names
            
            # Check they're marked as non-universal
            custom = [c for c in categories if c.name == "Custom Category"][0]
            assert custom.is_universal is False
    
    def test_load_taxonomy_missing_config(self):
        """Test that missing config doesn't crash."""
        categories = load_taxonomy(Path("/nonexistent/path/config.yaml"))
        
        # Should still have universal categories
        assert len(categories) == len(UNIVERSAL_TAXONOMY)


class TestFormatTaxonomy:
    """Tests for taxonomy formatting."""
    
    def test_format_taxonomy_for_prompt(self):
        """Test that formatting produces numbered list."""
        categories = [
            Category("Test1", "Description 1"),
            Category("Test2", "Description 2"),
        ]
        
        result = format_taxonomy_for_prompt(categories)
        
        assert "1. Test1: Description 1" in result
        assert "2. Test2: Description 2" in result
    
    def test_get_category_names(self):
        """Test extracting just names."""
        categories = [
            Category("Cat1", "Desc1"),
            Category("Cat2", "Desc2"),
        ]
        
        names = get_category_names(categories)
        
        assert names == ["Cat1", "Cat2"]
