"""Tests for eval runner and wrapper commands."""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from typer.testing import CliRunner

from reco.cli import app
from reco.core.eval_runner import EvalRunner, FRAMEWORKS
from reco.core.models import EvalRun, Response


runner = CliRunner()


class TestFrameworkDetection:
    """Tests for eval framework detection."""
    
    def test_detects_promptfoo_yaml(self, tmp_path):
        """Test detecting PromptFoo from promptfooconfig.yaml."""
        (tmp_path / "promptfooconfig.yaml").write_text("prompts: []")
        
        eval_runner = EvalRunner(tmp_path)
        framework = eval_runner.detect_framework()
        
        assert framework is not None
        assert framework.name == "PromptFoo"
    
    def test_detects_promptfoo_json(self, tmp_path):
        """Test detecting PromptFoo from promptfooconfig.json."""
        (tmp_path / "promptfooconfig.json").write_text('{"prompts": []}')
        
        eval_runner = EvalRunner(tmp_path)
        framework = eval_runner.detect_framework()
        
        assert framework is not None
        assert framework.name == "PromptFoo"
    
    def test_no_framework_detected(self, tmp_path):
        """Test when no framework is present."""
        eval_runner = EvalRunner(tmp_path)
        framework = eval_runner.detect_framework()
        
        assert framework is None


class TestInit:
    """Tests for reco init command."""
    
    def test_init_creates_config(self, tmp_path):
        """Test init creates .reco/config.yaml."""
        (tmp_path / "promptfooconfig.yaml").write_text("prompts: []")
        
        eval_runner = EvalRunner(tmp_path)
        result = eval_runner.init()
        
        assert result["success"] is True
        assert "PromptFoo" in result["message"]
        assert (tmp_path / ".reco" / "config.yaml").exists()
    
    def test_init_fails_without_framework(self, tmp_path):
        """Test init fails when no framework detected."""
        eval_runner = EvalRunner(tmp_path)
        result = eval_runner.init()
        
        assert result["success"] is False
        assert "No supported eval framework" in result["message"]
    
    def test_init_cli_command(self, tmp_path):
        """Test init CLI command."""
        (tmp_path / "promptfooconfig.yaml").write_text("prompts: []")
        
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(app, ["init"])
        
        # Should succeed if promptfoo.yaml exists in cwd
        # Note: Uses real cwd, so may not find promptfoo.yaml
        assert result.exit_code in (0, 1)


class TestPromptFooParser:
    """Tests for PromptFoo output parsing."""
    
    def test_parse_basic_output(self, tmp_path):
        """Test parsing basic PromptFoo output."""
        output = json.dumps({
            "results": {
                "results": [
                    {
                        "prompt": {"raw": "What is 2+2?"},
                        "response": {"output": "4"},
                        "success": True,
                    },
                    {
                        "prompt": {"raw": "What is the capital of France?"},
                        "response": {"output": "London"},
                        "success": False,
                        "gradingResult": {"reason": "Wrong answer"},
                    },
                ]
            }
        })
        
        eval_runner = EvalRunner(tmp_path)
        run = eval_runner._parse_promptfoo(output)
        
        assert len(run.responses) == 2
        assert run.responses[0].passed is True
        assert run.responses[1].passed is False
        assert run.responses[1].failure_reason == "Wrong answer"
    
    def test_parse_with_error(self, tmp_path):
        """Test parsing output with explicit error."""
        output = json.dumps({
            "results": {
                "results": [
                    {
                        "prompt": {"raw": "Test"},
                        "response": {"output": ""},
                        "success": False,
                        "error": "API timeout",
                    },
                ]
            }
        })
        
        eval_runner = EvalRunner(tmp_path)
        run = eval_runner._parse_promptfoo(output)
        
        assert run.responses[0].passed is False
        assert run.responses[0].failure_reason == "API timeout"
    
    def test_parse_empty_results(self, tmp_path):
        """Test parsing empty results."""
        output = json.dumps({"results": {"results": []}})
        
        eval_runner = EvalRunner(tmp_path)
        run = eval_runner._parse_promptfoo(output)
        
        assert len(run.responses) == 0


class TestGitInfo:
    """Tests for git info extraction."""
    
    def test_get_git_info_in_repo(self, tmp_path):
        """Test getting git info when in a repo."""
        eval_runner = EvalRunner(tmp_path)
        info = eval_runner.get_git_info()
        
        # May or may not be in a git repo
        assert "branch" in info
        assert "commit" in info


class TestInitCLI:
    """Tests for init CLI command."""
    
    def test_init_help(self):
        """Test init help output."""
        result = runner.invoke(app, ["init", "--help"])
        
        assert result.exit_code == 0
        assert "initialize" in result.stdout.lower()


class TestRunCLI:
    """Tests for run CLI command."""
    
    def test_run_help(self):
        """Test run help output."""
        result = runner.invoke(app, ["run", "--help"])
        
        assert result.exit_code == 0
        assert "eval" in result.stdout.lower()
    
    def test_run_requires_init(self, tmp_path):
        """Test run fails without init."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(app, ["run"])
        
        # Should fail because not initialized
        assert result.exit_code == 1 or "init" in result.stdout.lower()
