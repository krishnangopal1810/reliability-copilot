"""Eval framework runner and detection.

Wraps eval frameworks like PromptFoo to provide seamless integration.
"""

import json
import subprocess
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Literal
from datetime import datetime, timezone

import yaml

from .models import EvalRun, Response


@dataclass
class FrameworkConfig:
    """Configuration for detected eval framework."""
    name: str
    config_file: str
    run_command: list[str]
    output_parser: str  # Name of parser function to use


# Supported frameworks and their detection
FRAMEWORKS = {
    "promptfoo": FrameworkConfig(
        name="PromptFoo",
        config_file="promptfooconfig.yaml",
        run_command=["npx", "promptfoo", "eval", "--output", "-", "--no-progress-bar"],
        output_parser="promptfoo",
    ),
    "promptfoo_json": FrameworkConfig(
        name="PromptFoo",
        config_file="promptfooconfig.json",
        run_command=["npx", "promptfoo", "eval", "--output", "-", "--no-progress-bar"],
        output_parser="promptfoo",
    ),
}


class EvalRunner:
    """Detects and runs eval frameworks, converts output to reco format."""
    
    def __init__(self, working_dir: Optional[Path] = None):
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.config_path = self.working_dir / ".reco" / "config.yaml"
    
    def detect_framework(self) -> Optional[FrameworkConfig]:
        """Detect which eval framework is being used in the project.
        
        Returns:
            FrameworkConfig if detected, None otherwise.
        """
        for framework_id, config in FRAMEWORKS.items():
            if (self.working_dir / config.config_file).exists():
                return config
        return None
    
    def init(self) -> dict:
        """Initialize reco in the current directory.
        
        Creates .reco/config.yaml with detected framework settings.
        
        Returns:
            Dict with initialization status and message.
        """
        framework = self.detect_framework()
        
        if not framework:
            return {
                "success": False,
                "message": "No supported eval framework detected. Supported: PromptFoo",
                "framework": None,
            }
        
        # Create .reco directory
        reco_dir = self.working_dir / ".reco"
        reco_dir.mkdir(exist_ok=True)
        
        # Create config
        config = {
            "framework": framework.name.lower(),
            "config_file": framework.config_file,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)
        
        # Add .reco to .gitignore if it exists and not already there
        gitignore = self.working_dir / ".gitignore"
        if gitignore.exists():
            content = gitignore.read_text()
            if ".reco/" not in content and ".reco" not in content:
                with open(gitignore, "a") as f:
                    f.write("\n# Reco local data\n.reco/\n")
        
        return {
            "success": True,
            "message": f"Detected: {framework.name} ({framework.config_file})",
            "framework": framework.name,
        }
    
    def load_config(self) -> Optional[dict]:
        """Load .reco/config.yaml if it exists."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return None
    
    def run_eval(self) -> tuple[EvalRun, str]:
        """Run the configured eval framework and capture output.
        
        Returns:
            Tuple of (EvalRun, raw_output_string)
            
        Raises:
            RuntimeError: If no framework configured or execution fails.
        """
        config = self.load_config()
        if not config:
            raise RuntimeError("Not initialized. Run 'reco init' first.")
        
        framework = self.detect_framework()
        if not framework:
            raise RuntimeError(f"Framework config file not found: {config.get('config_file')}")
        
        # Create temp file for output
        temp_output = self.working_dir / ".reco" / "eval_output.json"
        
        # Build command with temp file output
        run_command = framework.run_command.copy()
        # Replace "-" with temp file path if present
        if "-" in run_command:
            run_command = [temp_output if x == "-" else x for x in run_command]
        else:
            run_command.extend(["--output", str(temp_output)])
        
        # Check if npx is available before running
        if run_command[0] == "npx":
            npx_check = subprocess.run(
                ["which", "npx"],
                capture_output=True,
                text=True,
            )
            if npx_check.returncode != 0:
                raise RuntimeError(
                    "npx not found. Node.js is required to run PromptFoo.\n\n"
                    "Install Node.js:\n"
                    "  macOS:   brew install node\n"
                    "  Ubuntu:  sudo apt install nodejs npm\n"
                    "  Windows: https://nodejs.org/\n\n"
                    "Then try again."
                )
        
        # Run the eval command
        try:
            result = subprocess.run(
                run_command,
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Eval command timed out after 10 minutes")
        except FileNotFoundError:
            raise RuntimeError(
                f"Command not found: {run_command[0]}.\n\n"
                "Make sure the eval framework is installed and available in PATH."
            )
        
        # PromptFoo returns non-zero exit code for test failures, but still writes output
        # So we check if output file exists rather than return code
        if not temp_output.exists():
            error_msg = result.stderr or result.stdout or "Unknown error"
            
            # Check for common errors
            if "401" in error_msg or "Unauthorized" in error_msg:
                raise RuntimeError(
                    "API authentication failed.\n\n"
                    "Make sure your API key is set:\n"
                    "  export OPENROUTER_API_KEY=your_key\n\n"
                    "Then try again."
                )
            
            raise RuntimeError(f"Eval failed:\n{error_msg}")
        
        # Read output
        output = temp_output.read_text()
        
        # Parse output
        eval_run = self._parse_output(output, framework.output_parser)
        return eval_run, output
    
    def _parse_output(self, output: str, parser_name: str) -> EvalRun:
        """Parse framework output into EvalRun."""
        if parser_name == "promptfoo":
            return self._parse_promptfoo(output)
        else:
            raise ValueError(f"Unknown parser: {parser_name}")
    
    def _parse_promptfoo(self, output: str) -> EvalRun:
        """Parse PromptFoo JSON output.
        
        PromptFoo output structure:
        {
            "results": {
                "results": [
                    {
                        "prompt": {...},
                        "response": {...},
                        "success": true/false,
                        "error": "...",
                        ...
                    }
                ]
            }
        }
        """
        try:
            data = json.loads(output)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse PromptFoo output as JSON: {e}")
        
        responses = []
        
        # Handle nested results structure
        results = data.get("results", {})
        if isinstance(results, dict):
            result_list = results.get("results", [])
        else:
            result_list = results
        
        for i, result in enumerate(result_list):
            # Extract prompt/input
            prompt = result.get("prompt", {})
            if isinstance(prompt, dict):
                input_text = prompt.get("raw", prompt.get("label", str(prompt)))
            else:
                input_text = str(prompt)
            
            # Extract response/output
            response = result.get("response", {})
            if isinstance(response, dict):
                output_text = response.get("output", str(response))
            else:
                output_text = str(response)
            
            # Determine pass/fail
            success = result.get("success", True)
            error = result.get("error")
            
            # Build failure reason
            failure_reason = None
            if not success:
                if error:
                    failure_reason = str(error)
                else:
                    # Check assertion results
                    grade_result = result.get("gradingResult", {})
                    if isinstance(grade_result, dict):
                        failure_reason = grade_result.get("reason", "Assertion failed")
            
            responses.append(Response(
                id=result.get("id", f"test_{i+1}"),
                input=input_text[:1000],  # Truncate long inputs
                output=output_text[:2000],  # Truncate long outputs
                expected=result.get("expected"),
                passed=success,
                failure_reason=failure_reason,
            ))
        
        return EvalRun(
            name=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            responses=responses,
            metadata={
                "framework": "promptfoo",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
    
    def get_git_info(self) -> dict:
        """Get current git branch and commit info."""
        info = {"branch": None, "commit": None}
        
        try:
            # Get branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.working_dir,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                info["branch"] = result.stdout.strip()
            
            # Get commit
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=self.working_dir,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                info["commit"] = result.stdout.strip()
        except FileNotFoundError:
            pass  # Git not available
        
        return info
