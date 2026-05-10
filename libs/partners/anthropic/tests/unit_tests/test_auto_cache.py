"""Unit tests for the auto-caching optimizations module."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from langchain_anthropic import ChatAnthropic
from langchain_anthropic.optimizations import (
    TokenUsageStats,
    _MIN_CACHE_CHARS,
    apply_auto_cache_to_system,
    apply_auto_cache_to_tools,
)

_SHORT = "Hi"
_LONG = "x" * _MIN_CACHE_CHARS  # exactly at threshold


# ---------------------------------------------------------------------------
# apply_auto_cache_to_system
# ---------------------------------------------------------------------------


def test_system_none_returns_none() -> None:
    assert apply_auto_cache_to_system(None) is None


def test_short_string_unchanged() -> None:
    result = apply_auto_cache_to_system(_SHORT)
    assert result == _SHORT


def test_long_string_gets_cache_control() -> None:
    result = apply_auto_cache_to_system(_LONG)
    assert isinstance(result, list)
    assert len(result) == 1
    block = result[0]
    assert block["type"] == "text"
    assert block["text"] == _LONG
    assert block["cache_control"] == {"type": "ephemeral"}


def test_list_short_unchanged() -> None:
    blocks = [{"type": "text", "text": "short"}]
    result = apply_auto_cache_to_system(blocks)
    assert result == blocks


def test_list_long_adds_cache_control_to_last_text_block() -> None:
    blocks = [
        {"type": "text", "text": "x" * (_MIN_CACHE_CHARS // 2)},
        {"type": "text", "text": "x" * (_MIN_CACHE_CHARS // 2)},
    ]
    result = apply_auto_cache_to_system(blocks)
    assert isinstance(result, list)
    # First block should be untouched
    assert "cache_control" not in result[0]
    # Last block should have cache_control
    assert result[-1]["cache_control"] == {"type": "ephemeral"}


def test_list_does_not_overwrite_existing_cache_control() -> None:
    existing_cc = {"type": "ephemeral"}
    blocks = [
        {"type": "text", "text": "x" * _MIN_CACHE_CHARS, "cache_control": existing_cc},
    ]
    result = apply_auto_cache_to_system(blocks)
    # The existing cache_control is preserved (no second block to tag)
    assert result[-1]["cache_control"] is existing_cc


def test_list_does_not_mutate_original() -> None:
    blocks = [{"type": "text", "text": "x" * _MIN_CACHE_CHARS}]
    apply_auto_cache_to_system(blocks)
    # Original must be unchanged
    assert "cache_control" not in blocks[0]


# ---------------------------------------------------------------------------
# apply_auto_cache_to_tools
# ---------------------------------------------------------------------------


def test_tools_empty_returns_unchanged() -> None:
    assert apply_auto_cache_to_tools([]) == []


def test_tools_adds_cache_control_to_last() -> None:
    tools = [
        {"name": "tool_a", "description": "a"},
        {"name": "tool_b", "description": "b"},
    ]
    result = apply_auto_cache_to_tools(tools)
    assert "cache_control" not in result[0]
    assert result[-1]["cache_control"] == {"type": "ephemeral"}


def test_tools_skips_if_last_already_has_cache_control() -> None:
    tools = [{"name": "t", "cache_control": {"type": "ephemeral"}}]
    result = apply_auto_cache_to_tools(tools)
    # Should still be ephemeral, not doubled
    assert result[0]["cache_control"] == {"type": "ephemeral"}


def test_tools_does_not_mutate_original() -> None:
    tools = [{"name": "t", "description": "d"}]
    apply_auto_cache_to_tools(tools)
    assert "cache_control" not in tools[0]


# ---------------------------------------------------------------------------
# TokenUsageStats
# ---------------------------------------------------------------------------


def test_token_usage_stats_empty() -> None:
    stats = TokenUsageStats()
    assert stats.total_tokens == 0
    assert stats.cache_hit_rate == 0.0


def test_token_usage_stats_update() -> None:
    stats = TokenUsageStats()
    stats.update(
        {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": 20,
            "cache_read_input_tokens": 80,
        }
    )
    assert stats.input_tokens == 100
    assert stats.output_tokens == 50
    assert stats.cache_creation_tokens == 20
    assert stats.cache_read_tokens == 80
    assert stats.total_tokens == 150


def test_token_usage_cache_hit_rate() -> None:
    stats = TokenUsageStats()
    stats.update({"input_tokens": 100, "cache_read_input_tokens": 100})
    # 100 / (100 + 100) = 0.5
    assert stats.cache_hit_rate == pytest.approx(0.5)


def test_token_usage_summary_keys() -> None:
    stats = TokenUsageStats()
    summary = stats.summary()
    assert set(summary) == {
        "input_tokens",
        "output_tokens",
        "cache_creation_tokens",
        "cache_read_tokens",
        "total_tokens",
        "cache_hit_rate",
    }


# ---------------------------------------------------------------------------
# ChatAnthropic.auto_cache integration
# ---------------------------------------------------------------------------


def _make_model(**kwargs: object) -> ChatAnthropic:
    return ChatAnthropic(model="claude-opus-4-7", api_key="test-key", **kwargs)  # type: ignore[arg-type]


def test_auto_cache_default_false() -> None:
    model = _make_model()
    assert model.auto_cache is False


def test_auto_cache_field_set() -> None:
    model = _make_model(auto_cache=True)
    assert model.auto_cache is True


def test_auto_cache_injects_into_payload_system() -> None:
    """_get_request_payload must add cache_control to a long system message."""
    model = _make_model(auto_cache=True)
    long_system = "x" * _MIN_CACHE_CHARS
    messages = [SystemMessage(content=long_system), HumanMessage(content="hello")]
    payload = model._get_request_payload(messages)
    system = payload.get("system")
    assert isinstance(system, list), "system should be promoted to a list"
    assert any(b.get("cache_control") for b in system), "cache_control must be present"


def test_auto_cache_skips_short_system() -> None:
    """Short system content must not be promoted."""
    model = _make_model(auto_cache=True)
    messages = [SystemMessage(content="short"), HumanMessage(content="hello")]
    payload = model._get_request_payload(messages)
    system = payload.get("system")
    # Should remain a plain string (not promoted)
    assert isinstance(system, str)


def test_no_auto_cache_leaves_system_unchanged() -> None:
    """With auto_cache=False the system must not be modified."""
    model = _make_model(auto_cache=False)
    long_system = "x" * _MIN_CACHE_CHARS
    messages = [SystemMessage(content=long_system), HumanMessage(content="hello")]
    payload = model._get_request_payload(messages)
    system = payload.get("system")
    # Should remain a string
    assert isinstance(system, str)


def test_auto_cache_injects_into_tools() -> None:
    """auto_cache must tag the last tool definition in the payload."""
    from langchain_anthropic.chat_models import convert_to_anthropic_tool
    from langchain_core.tools import tool

    @tool
    def greet(name: str) -> str:
        """Greet someone by name."""
        return f"Hello, {name}!"

    model = _make_model(auto_cache=True)
    messages = [HumanMessage(content="hello")]
    payload = model._get_request_payload(
        messages,
        tools=[convert_to_anthropic_tool(greet)],
    )
    tools = payload.get("tools", [])
    assert tools, "tools must be present in payload"
    assert tools[-1].get("cache_control") == {"type": "ephemeral"}
