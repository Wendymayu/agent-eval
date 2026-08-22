"""MCP support is an optional extension (not wired in v1).

These stubs let llm_test_case.py / conversational_test_case.py import and
define their pydantic models cleanly. The classes mirror the field
signatures of deepeval/test_case/mcp.py so the data model stays intact,
but the helper functions are no-ops (MCP evaluation is not wired in v1).

Populate the real behaviour from deepeval/test_case/mcp.py when adding
Agent/MCP evaluation.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from pydantic import AnyUrl, BaseModel


@dataclass
class MCPServer:
    server_name: str
    transport: Optional[Literal["stdio", "sse", "streamable-http"]] = None
    available_tools: Optional[List] = None
    available_resources: Optional[List] = None
    available_prompts: Optional[List] = None


class MCPToolCall(BaseModel):
    name: str
    args: Dict
    result: object


class MCPPromptCall(BaseModel):
    name: str
    result: object


class MCPResourceCall(BaseModel):
    uri: AnyUrl
    result: object


def normalize_mcp_servers(mcp_servers: List[Any]) -> List[Any]:
    """No-op in v1: return the servers unchanged."""
    return list(mcp_servers)


def validate_mcp_servers(mcp_servers: Any) -> None:
    """No-op in v1: validation is deferred until MCP is wired."""
    return None


def get_available_mcp_tool_names(mcp_servers: List[MCPServer]) -> List[str]:
    """No-op in v1: no MCP tools are available."""
    return []
