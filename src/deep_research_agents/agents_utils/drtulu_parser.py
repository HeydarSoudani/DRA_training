"""DR-Tulu action parser.

Self-contained extraction of the two classes needed from
dr-tulu/agent/dr_agent/tool_interface/tool_parsers.py:

  ToolCallInfo                  – parsed tool-call metadata
  ToolCallParser                – abstract base
  UnifiedToolCallParserV20250824 – concrete parser for the v20250824 action format

Format handled:
  <call_tool name="tool_name" [param="value" ...]>content</call_tool>
  <call_tool name="tool_name" [param="value" ...]>content</call>   (short close)

Source: dr-tulu/agent/dr_agent/tool_interface/tool_parsers.py
        (UnifiedToolCallParserV20250824, lines 371-472)
"""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Data model ────────────────────────────────────────────────────────────────

class ToolCallInfo(BaseModel):
    """Metadata about a single parsed tool call."""

    content: str                                      # main body / query text
    parameters: Dict[str, Any] = Field(default_factory=dict)  # extra XML attributes
    start_pos: int                                    # byte offset of opening tag
    end_pos: int                                      # byte offset after closing tag


# ── Abstract base ─────────────────────────────────────────────────────────────

class ToolCallParser(ABC):
    """Abstract base class for tool-call parsers."""

    @abstractmethod
    def has_calls(self, text: str, tool_name: str) -> bool:
        """Return True if *text* contains a call to *tool_name*."""

    @abstractmethod
    def parse_call(self, text: str, tool_name: str) -> Optional[ToolCallInfo]:
        """Return the first call to *tool_name* found in *text*, or None."""

    @abstractmethod
    def format_result(self, formatted_output: str, output: Any) -> str:
        """Wrap *formatted_output* in the parser's result envelope."""

    @property
    @abstractmethod
    def stop_sequences(self) -> List[str]:
        """Token sequences the LLM should stop at after emitting a tool call."""


# ── Concrete parser ───────────────────────────────────────────────────────────

class UnifiedToolCallParserV20250824(ToolCallParser):
    """Parser for the DR-Tulu v20250824 action format.

    Handles both closing variants produced by LLMs:
      <call_tool name="tool_name">content</call_tool>
      <call_tool name="tool_name">content</call>
    """

    @property
    def stop_sequences(self) -> List[str]:
        return ["</call_tool>", "</call>"]

    def has_calls(self, text: str, tool_name: str) -> bool:
        for pattern in [
            r'<call_tool\s+name="' + re.escape(tool_name) + r'"[^>]*?>.*?</call_tool>',
            r'<call_tool\s+name="' + re.escape(tool_name) + r'"[^>]*?>.*?</call>',
        ]:
            if re.search(pattern, text, re.DOTALL):
                return True
        return False

    def parse_call(self, text: str, tool_name: str) -> Optional[ToolCallInfo]:
        for pattern in [
            r"<call_tool\s+([^>]*?)>(.*?)</call_tool>",
            r"<call_tool\s+([^>]*?)>(.*?)</call>",
        ]:
            for match in re.finditer(pattern, text, re.DOTALL):
                attr_string = match.group(1)
                content = match.group(2).strip()

                # Parse XML attributes
                attributes: Dict[str, str] = dict(
                    re.findall(r'(\w+)="([^"]*)"', attr_string)
                )

                if attributes.get("name") == tool_name:
                    return ToolCallInfo(
                        content=content,
                        parameters={k: v for k, v in attributes.items() if k != "name"},
                        start_pos=match.start(),
                        end_pos=match.end(),
                    )
        return None

    def format_result(self, formatted_output: str, output: Any) -> str:
        """Wrap snippet XML in a <tool_output> envelope."""
        return f"<tool_output>{formatted_output}</tool_output>"
