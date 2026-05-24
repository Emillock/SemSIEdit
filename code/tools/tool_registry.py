"""
Tool registry for managing LangChain tools in the feedback system.
"""
import json
import os
import re
import logging
from typing import List, Dict, Any

from langchain_core.tools import BaseTool

try:
    from .dp_rewriting import DPRewritingTool
    from .pii_detection import PIIDetectionTool
    from .web_search import get_web_search_tool
except ImportError:
    from dp_rewriting import DPRewritingTool
    from pii_detection import PIIDetectionTool
    from web_search import get_web_search_tool


def get_langchain_tools() -> List[BaseTool]:
    """Get all available LangChain tools for the editor agent."""
    tools = []

    try:
        tools.append(DPRewritingTool())
        logging.info("DP Rewriting tool registered")
    except Exception as e:
        logging.error(f"Error initializing DP Rewriting tool: {e}")

    if os.environ.get("TAVILY_API_KEY"):
        try:
            tools.append(get_web_search_tool())
            logging.info("Tavily web search tool registered")
        except Exception as e:
            logging.warning(f"Web search tool not available: {e}")
    else:
        logging.info("Tavily web search tool skipped (TAVILY_API_KEY not set)")

    return tools


class ToolRegistry:
    """Registry that wraps LangChain tools and provides parse/execute/docs interface."""

    def __init__(self, tools: List[BaseTool]):
        self._tools: Dict[str, BaseTool] = {t.name: t for t in tools}

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def generate_tools_documentation(self) -> str:
        lines = []
        for name, tool in self._tools.items():
            lines.append(f"Tool: {name}")
            lines.append(f"  Description: {tool.description}")
            if hasattr(tool, 'args_schema') and tool.args_schema:
                schema = tool.args_schema.model_json_schema()
                props = schema.get("properties", {})
                for pname, pinfo in props.items():
                    desc = pinfo.get("description", "")
                    default = pinfo.get("default", "")
                    ptype = pinfo.get("type", "")
                    default_str = f" (default: {default})" if default != "" else ""
                    lines.append(f"  - {pname} ({ptype}): {desc}{default_str}")
            lines.append("")
        return "\n".join(lines)

    def parse_and_execute_tool_call(self, tool_call_text: str, api_token: str = None) -> Dict[str, Any]:
        try:
            match = re.match(r'(\w+)\s*\((.+)\)\s*$', tool_call_text, re.DOTALL)
            if not match:
                return {"error": f"Could not parse tool call: {tool_call_text}"}

            tool_name = match.group(1)
            args_str = match.group(2).strip()

            if tool_name not in self._tools:
                return {"error": f"Unknown tool: {tool_name}. Available: {', '.join(self._tools.keys())}"}

            tool = self._tools[tool_name]
            args = json.loads(args_str)

            if api_token and "api_token" in (tool.args_schema.model_json_schema().get("properties", {}) if tool.args_schema else {}):
                args.setdefault("api_token", api_token)

            result_str = tool._run(**args)
            try:
                result = json.loads(result_str)
            except (json.JSONDecodeError, TypeError):
                result = {"result": str(result_str)}

            result["tool_name"] = tool_name
            return result
        except Exception as e:
            return {"error": str(e)}


def get_tool_registry() -> ToolRegistry:
    """Create a ToolRegistry with all available tools."""
    return ToolRegistry(get_langchain_tools())
