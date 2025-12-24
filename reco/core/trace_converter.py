"""Trace format converters for framework integration.

Supports importing traces from:
- LangChain (JSONL callback format)
- OpenAI Assistants (Run Steps API)
"""

import json
from pathlib import Path
from typing import Optional

from .models import AgentTrace, AgentStep


class TraceConverter:
    """Converts external trace formats to reco format."""
    
    @staticmethod
    def from_langchain(filepath: str) -> AgentTrace:
        """Convert LangChain JSONL trace to AgentTrace.
        
        LangChain callbacks export events like:
        {"event": "on_chain_start", "run_id": "...", "name": "AgentExecutor", ...}
        {"event": "on_tool_start", "run_id": "...", "name": "search", "input": {...}}
        {"event": "on_tool_end", "output": "..."}
        
        Args:
            filepath: Path to JSONL file with LangChain callback events
            
        Returns:
            AgentTrace in reco format
        """
        path = Path(filepath)
        events = []
        
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        
        steps = []
        step_num = 0
        current_tool = None
        current_input = None
        goal = "LangChain Agent Execution"
        outcome = "success"
        
        for event in events:
            event_type = event.get("event", "")
            
            # Extract goal from chain start
            if event_type == "on_chain_start":
                name = event.get("name", "")
                if "Agent" in name:
                    inputs = event.get("inputs", {})
                    if isinstance(inputs, dict):
                        goal = inputs.get("input", goal)
            
            # Tool start
            elif event_type == "on_tool_start":
                current_tool = event.get("name", "unknown_tool")
                current_input = event.get("input", {})
                if isinstance(current_input, str):
                    current_input = {"query": current_input}
            
            # Tool end
            elif event_type == "on_tool_end" and current_tool:
                step_num += 1
                output = event.get("output", "")
                error = event.get("error")
                success = error is None
                
                if not success:
                    outcome = "failed"
                
                steps.append(AgentStep(
                    step=step_num,
                    action=current_tool,
                    input=current_input or {},
                    output=output,
                    success=success,
                    error=error,
                ))
                current_tool = None
                current_input = None
            
            # Tool error
            elif event_type == "on_tool_error" and current_tool:
                step_num += 1
                outcome = "failed"
                
                steps.append(AgentStep(
                    step=step_num,
                    action=current_tool,
                    input=current_input or {},
                    output=None,
                    success=False,
                    error=str(event.get("error", "Unknown error")),
                ))
                current_tool = None
                current_input = None
        
        return AgentTrace(
            id=path.stem,
            goal=goal,
            outcome=outcome,
            steps=steps,
            metadata={"source": "langchain", "original_file": str(path)},
        )
    
    @staticmethod
    def from_openai(filepath: str) -> AgentTrace:
        """Convert OpenAI Assistants run steps to AgentTrace.
        
        OpenAI format:
        {
            "data": [
                {
                    "id": "step_abc",
                    "status": "completed",
                    "step_details": {
                        "type": "tool_calls",
                        "tool_calls": [{"type": "function", "function": {...}}]
                    }
                }
            ]
        }
        
        Args:
            filepath: Path to JSON file with OpenAI run steps
            
        Returns:
            AgentTrace in reco format
        """
        path = Path(filepath)
        
        with open(path) as f:
            data = json.load(f)
        
        # Handle both direct list and wrapped format
        if isinstance(data, dict):
            run_steps = data.get("data", data.get("steps", []))
            run_id = data.get("run_id", path.stem)
            thread_id = data.get("thread_id", "")
        else:
            run_steps = data
            run_id = path.stem
            thread_id = ""
        
        steps = []
        outcome = "success"
        
        for i, step in enumerate(run_steps, 1):
            status = step.get("status", "completed")
            step_details = step.get("step_details", {})
            step_type = step_details.get("type", "")
            
            if step_type == "tool_calls":
                for tool_call in step_details.get("tool_calls", []):
                    if tool_call.get("type") == "function":
                        func = tool_call.get("function", {})
                        
                        # Parse arguments
                        args_str = func.get("arguments", "{}")
                        try:
                            args = json.loads(args_str) if isinstance(args_str, str) else args_str
                        except json.JSONDecodeError:
                            args = {"raw": args_str}
                        
                        # Parse output
                        output_str = func.get("output", tool_call.get("output", ""))
                        try:
                            output = json.loads(output_str) if isinstance(output_str, str) else output_str
                        except (json.JSONDecodeError, TypeError):
                            output = output_str
                        
                        success = status in ("completed", "in_progress")
                        error = None
                        
                        if status == "failed":
                            success = False
                            outcome = "failed"
                            last_error = step.get("last_error", {})
                            error = last_error.get("message", "Unknown error")
                        
                        steps.append(AgentStep(
                            step=len(steps) + 1,
                            action=func.get("name", "unknown_function"),
                            input=args,
                            output=output,
                            success=success,
                            error=error,
                        ))
            
            elif step_type == "message_creation":
                # Agent generating a response
                message = step_details.get("message_creation", {})
                steps.append(AgentStep(
                    step=len(steps) + 1,
                    action="generate_response",
                    input={},
                    output={"message_id": message.get("message_id", "")},
                    success=status == "completed",
                ))
        
        # Determine final outcome
        if run_steps:
            last_status = run_steps[-1].get("status", "")
            if last_status == "failed":
                outcome = "failed"
            elif last_status == "cancelled":
                outcome = "failed"
        
        return AgentTrace(
            id=run_id,
            goal=f"OpenAI Assistant Run ({thread_id})" if thread_id else "OpenAI Assistant Run",
            outcome=outcome,
            steps=steps,
            metadata={"source": "openai", "original_file": str(path)},
        )
    
    @classmethod
    def convert(cls, filepath: str, format: str) -> AgentTrace:
        """Convert trace from specified format.
        
        Args:
            filepath: Path to trace file
            format: One of 'langchain', 'openai', 'auto'
            
        Returns:
            AgentTrace in reco format
        """
        if format == "auto":
            format = cls._detect_format(filepath)
        
        if format == "langchain":
            return cls.from_langchain(filepath)
        elif format == "openai":
            return cls.from_openai(filepath)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    @staticmethod
    def _detect_format(filepath: str) -> str:
        """Auto-detect trace format from file contents."""
        path = Path(filepath)
        
        with open(path) as f:
            first_line = f.readline()
        
        try:
            data = json.loads(first_line)
            
            # LangChain uses "event" key
            if "event" in data:
                return "langchain"
            
            # OpenAI uses "data" or "step_details"
            if "data" in data or "step_details" in data:
                return "openai"
        except json.JSONDecodeError:
            pass
        
        # Default to trying OpenAI (full JSON file)
        return "openai"
