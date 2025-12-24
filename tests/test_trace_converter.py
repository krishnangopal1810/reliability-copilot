"""Tests for trace format converters."""

import pytest
import json
from pathlib import Path

from typer.testing import CliRunner

from reco.cli import app
from reco.core.trace_converter import TraceConverter


runner = CliRunner()


class TestLangChainConverter:
    """Tests for LangChain JSONL conversion."""
    
    def test_converts_tool_events(self, tmp_path):
        """Test converting LangChain tool events."""
        trace = tmp_path / "trace.jsonl"
        trace.write_text("""{"event": "on_chain_start", "name": "AgentExecutor", "inputs": {"input": "Test goal"}}
{"event": "on_tool_start", "name": "search", "input": {"query": "test"}}
{"event": "on_tool_end", "output": "search results"}
{"event": "on_tool_start", "name": "process", "input": {"data": "test"}}
{"event": "on_tool_end", "output": "processed"}
""")
        
        result = TraceConverter.from_langchain(str(trace))
        
        assert len(result.steps) == 2
        assert result.steps[0].action == "search"
        assert result.steps[1].action == "process"
        assert result.outcome == "success"
    
    def test_handles_tool_errors(self, tmp_path):
        """Test handling tool errors in LangChain traces."""
        trace = tmp_path / "trace.jsonl"
        trace.write_text("""{"event": "on_tool_start", "name": "api_call", "input": {}}
{"event": "on_tool_error", "error": "Connection refused"}
""")
        
        result = TraceConverter.from_langchain(str(trace))
        
        assert len(result.steps) == 1
        assert result.steps[0].success is False
        assert result.steps[0].error == "Connection refused"
        assert result.outcome == "failed"
    
    def test_extracts_goal_from_chain_start(self, tmp_path):
        """Test extracting goal from chain start event."""
        trace = tmp_path / "trace.jsonl"
        trace.write_text("""{"event": "on_chain_start", "name": "AgentExecutor", "inputs": {"input": "Book a flight"}}
{"event": "on_tool_start", "name": "search", "input": {}}
{"event": "on_tool_end", "output": "done"}
""")
        
        result = TraceConverter.from_langchain(str(trace))
        
        assert result.goal == "Book a flight"
    
    def test_empty_file(self, tmp_path):
        """Test handling empty trace file."""
        trace = tmp_path / "empty.jsonl"
        trace.write_text("")
        
        result = TraceConverter.from_langchain(str(trace))
        
        assert len(result.steps) == 0
        assert result.outcome == "success"  # No failures
    
    def test_string_input_wrapped_in_dict(self, tmp_path):
        """Test that string inputs are wrapped properly."""
        trace = tmp_path / "trace.jsonl"
        trace.write_text("""{"event": "on_tool_start", "name": "search", "input": "raw query string"}
{"event": "on_tool_end", "output": "results"}
""")
        
        result = TraceConverter.from_langchain(str(trace))
        
        assert result.steps[0].input == {"query": "raw query string"}
    
    def test_tool_end_without_start_ignored(self, tmp_path):
        """Test that tool_end without matching start is ignored."""
        trace = tmp_path / "trace.jsonl"
        trace.write_text("""{"event": "on_tool_end", "output": "orphan output"}
{"event": "on_tool_start", "name": "real_tool", "input": {}}
{"event": "on_tool_end", "output": "proper output"}
""")
        
        result = TraceConverter.from_langchain(str(trace))
        
        # Only the proper tool should be recorded
        assert len(result.steps) == 1
        assert result.steps[0].action == "real_tool"
    
    def test_metadata_source_is_langchain(self, tmp_path):
        """Test that metadata includes source."""
        trace = tmp_path / "trace.jsonl"
        trace.write_text("""{"event": "on_tool_start", "name": "test", "input": {}}
{"event": "on_tool_end", "output": "done"}
""")
        
        result = TraceConverter.from_langchain(str(trace))
        
        assert result.metadata["source"] == "langchain"
        assert "original_file" in result.metadata
    
    def test_tool_end_with_error_field(self, tmp_path):
        """Test on_tool_end event that includes error field (rare but possible)."""
        trace = tmp_path / "trace.jsonl"
        trace.write_text("""{"event": "on_tool_start", "name": "api", "input": {}}
{"event": "on_tool_end", "output": "partial", "error": "timeout after output"}
""")
        
        result = TraceConverter.from_langchain(str(trace))
        
        assert result.steps[0].success is False
        assert result.outcome == "failed"
class TestOpenAIConverter:
    """Tests for OpenAI Run Steps conversion."""
    
    def test_converts_tool_calls(self, tmp_path):
        """Test converting OpenAI tool calls."""
        trace = tmp_path / "trace.json"
        trace.write_text(json.dumps({
            "data": [
                {
                    "id": "step_1",
                    "status": "completed",
                    "step_details": {
                        "type": "tool_calls",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "NYC"}',
                                    "output": '{"temp": 70}'
                                }
                            }
                        ]
                    }
                }
            ]
        }))
        
        result = TraceConverter.from_openai(str(trace))
        
        assert len(result.steps) == 1
        assert result.steps[0].action == "get_weather"
        assert result.steps[0].input == {"city": "NYC"}
        assert result.steps[0].output == {"temp": 70}
    
    def test_handles_failed_steps(self, tmp_path):
        """Test handling failed OpenAI steps."""
        trace = tmp_path / "trace.json"
        trace.write_text(json.dumps({
            "data": [
                {
                    "id": "step_1",
                    "status": "failed",
                    "step_details": {
                        "type": "tool_calls",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": "failing_tool", "arguments": "{}"}
                            }
                        ]
                    },
                    "last_error": {"message": "API error"}
                }
            ]
        }))
        
        result = TraceConverter.from_openai(str(trace))
        
        assert result.steps[0].success is False
        assert result.steps[0].error == "API error"
        assert result.outcome == "failed"
    
    def test_handles_message_creation(self, tmp_path):
        """Test handling message creation steps."""
        trace = tmp_path / "trace.json"
        trace.write_text(json.dumps({
            "data": [
                {
                    "id": "step_1",
                    "status": "completed",
                    "step_details": {
                        "type": "message_creation",
                        "message_creation": {"message_id": "msg_123"}
                    }
                }
            ]
        }))
        
        result = TraceConverter.from_openai(str(trace))
        
        assert len(result.steps) == 1
        assert result.steps[0].action == "generate_response"
        assert result.steps[0].output == {"message_id": "msg_123"}
    
    def test_handles_cancelled_status(self, tmp_path):
        """Test handling cancelled run steps."""
        trace = tmp_path / "trace.json"
        trace.write_text(json.dumps({
            "data": [
                {
                    "id": "step_1",
                    "status": "cancelled",
                    "step_details": {
                        "type": "message_creation",
                        "message_creation": {}
                    }
                }
            ]
        }))
        
        result = TraceConverter.from_openai(str(trace))
        
        assert result.outcome == "failed"
    
    def test_handles_malformed_json_arguments(self, tmp_path):
        """Test handling invalid JSON in arguments."""
        trace = tmp_path / "trace.json"
        trace.write_text(json.dumps({
            "data": [
                {
                    "id": "step_1",
                    "status": "completed",
                    "step_details": {
                        "type": "tool_calls",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "test",
                                    "arguments": "not valid json {"
                                }
                            }
                        ]
                    }
                }
            ]
        }))
        
        result = TraceConverter.from_openai(str(trace))
        
        # Should fall back to raw
        assert result.steps[0].input == {"raw": "not valid json {"}
    
    def test_empty_run_steps(self, tmp_path):
        """Test handling empty run steps."""
        trace = tmp_path / "trace.json"
        trace.write_text(json.dumps({"data": []}))
        
        result = TraceConverter.from_openai(str(trace))
        
        assert len(result.steps) == 0
        assert result.outcome == "success"
    
    def test_direct_list_format(self, tmp_path):
        """Test handling direct list format (not wrapped in 'data')."""
        trace = tmp_path / "trace.json"
        trace.write_text(json.dumps([
            {
                "id": "step_1",
                "status": "completed",
                "step_details": {
                    "type": "message_creation",
                    "message_creation": {"message_id": "msg_1"}
                }
            }
        ]))
        
        result = TraceConverter.from_openai(str(trace))
        
        assert len(result.steps) == 1
    
    def test_metadata_includes_thread_id(self, tmp_path):
        """Test that thread_id appears in goal if present."""
        trace = tmp_path / "trace.json"
        trace.write_text(json.dumps({
            "thread_id": "thread_xyz",
            "data": []
        }))
        
        result = TraceConverter.from_openai(str(trace))
        
        assert "thread_xyz" in result.goal


class TestAutoDetect:
    """Tests for format auto-detection."""
    
    def test_detects_langchain(self, tmp_path):
        """Test auto-detecting LangChain format."""
        trace = tmp_path / "trace.jsonl"
        trace.write_text('{"event": "on_tool_start", "name": "test"}')
        
        format = TraceConverter._detect_format(str(trace))
        
        assert format == "langchain"
    
    def test_detects_openai(self, tmp_path):
        """Test auto-detecting OpenAI format."""
        trace = tmp_path / "trace.json"
        trace.write_text('{"data": []}')
        
        format = TraceConverter._detect_format(str(trace))
        
        assert format == "openai"
    
    def test_fallback_to_openai(self, tmp_path):
        """Test fallback to OpenAI when format unknown."""
        trace = tmp_path / "trace.json"
        trace.write_text('{"unknown": "format"}')
        
        format = TraceConverter._detect_format(str(trace))
        
        assert format == "openai"  # Default fallback
    
    def test_handles_invalid_json_first_line(self, tmp_path):
        """Test handling invalid JSON in first line."""
        trace = tmp_path / "trace.txt"
        trace.write_text('not json at all')
        
        format = TraceConverter._detect_format(str(trace))
        
        assert format == "openai"  # Fallback


class TestConvertMethod:
    """Tests for the main convert method."""
    
    def test_convert_langchain(self, tmp_path):
        """Test convert() with langchain format."""
        trace = tmp_path / "trace.jsonl"
        trace.write_text('{"event": "on_tool_start", "name": "test", "input": {}}\n{"event": "on_tool_end", "output": "done"}')
        
        result = TraceConverter.convert(str(trace), "langchain")
        
        assert result.metadata["source"] == "langchain"
    
    def test_convert_openai(self, tmp_path):
        """Test convert() with openai format."""
        trace = tmp_path / "trace.json"
        trace.write_text(json.dumps({"data": []}))
        
        result = TraceConverter.convert(str(trace), "openai")
        
        assert result.metadata["source"] == "openai"
    
    def test_convert_auto(self, tmp_path):
        """Test convert() with auto-detection."""
        trace = tmp_path / "trace.jsonl"
        trace.write_text('{"event": "on_tool_start", "name": "test", "input": {}}\n{"event": "on_tool_end", "output": "done"}')
        
        result = TraceConverter.convert(str(trace), "auto")
        
        assert result.metadata["source"] == "langchain"
    
    def test_convert_unknown_format_raises(self, tmp_path):
        """Test convert() with unknown format raises error."""
        trace = tmp_path / "trace.json"
        trace.write_text("{}")
        
        with pytest.raises(ValueError, match="Unknown format"):
            TraceConverter.convert(str(trace), "unknown_format")


class TestImportTraceCLI:
    """Tests for import-trace CLI command."""
    
    def test_import_trace_help(self):
        """Test import-trace help output."""
        result = runner.invoke(app, ["import-trace", "--help"])
        
        assert result.exit_code == 0
        assert "langchain" in result.stdout.lower()
        assert "openai" in result.stdout.lower()
    
    def test_import_langchain_trace(self, tmp_path):
        """Test importing LangChain trace."""
        trace = tmp_path / "trace.jsonl"
        trace.write_text('{"event": "on_tool_start", "name": "test", "input": {}}\n{"event": "on_tool_end", "output": "done"}')
        
        result = runner.invoke(app, ["import-trace", str(trace), "--format", "langchain"])
        
        assert result.exit_code == 0
        assert "Converted" in result.stdout
        
        # Check output file was created
        output_file = tmp_path / "trace_reco.json"
        assert output_file.exists()
    
    def test_import_with_analyze(self, tmp_path):
        """Test import with immediate analysis."""
        trace = tmp_path / "trace.jsonl"
        trace.write_text('{"event": "on_tool_start", "name": "test", "input": {}}\n{"event": "on_tool_error", "error": "failed"}')
        
        result = runner.invoke(app, ["import-trace", str(trace), "--format", "langchain", "--analyze"])
        
        assert result.exit_code == 0
        assert "AGENT TRACE ANALYSIS" in result.stdout
    
    def test_import_missing_file(self, tmp_path):
        """Test error handling for missing file."""
        result = runner.invoke(app, ["import-trace", "/nonexistent/file.json"])
        
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()
    
    def test_import_custom_output(self, tmp_path):
        """Test custom output path."""
        trace = tmp_path / "trace.jsonl"
        trace.write_text('{"event": "on_tool_start", "name": "test", "input": {}}\n{"event": "on_tool_end", "output": "done"}')
        
        output = tmp_path / "custom_output.json"
        
        result = runner.invoke(app, ["import-trace", str(trace), "--format", "langchain", "--output", str(output)])
        
        assert result.exit_code == 0
        assert output.exists()
