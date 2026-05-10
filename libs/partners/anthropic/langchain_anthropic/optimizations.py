"""Token optimization utilities for the Anthropic connector.

Provides automatic prompt-cache injection and token-budget helpers that
reduce billable token usage without requiring callers to annotate every
request manually.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

# Anthropic enforces a minimum cacheable block size.
# Standard models require ≥1 024 tokens; Haiku requires ≥2 048.
# We use 1 024 as the conservative default (~4 096 chars at ~4 chars/token).
_MIN_CACHE_TOKENS: int = 1_024
_CHARS_PER_TOKEN: int = 4
_MIN_CACHE_CHARS: int = _MIN_CACHE_TOKENS * _CHARS_PER_TOKEN

_EPHEMERAL: dict[str, str] = {"type": "ephemeral"}


def _text_len(block: str | dict[str, Any] | list[Any]) -> int:
    """Estimate character count of an Anthropic content block or system value."""
    if isinstance(block, str):
        return len(block)
    if isinstance(block, dict):
        return len(block.get("text", ""))
    if isinstance(block, list):
        return sum(_text_len(b) for b in block)
    return 0


def apply_auto_cache_to_system(
    system: str | list[dict[str, Any]] | None,
) -> str | list[dict[str, Any]] | None:
    """Return *system* with `cache_control` added to the last eligible block.

    If the total estimated size is below `_MIN_CACHE_CHARS` the value is
    returned unchanged so we never pay the caching overhead for tiny prompts.

    Args:
        system: The Anthropic system parameter – either a plain string, a list
            of content blocks, or `None`.

    Returns:
        The (possibly modified) system value, or `None` when input is `None`.
    """
    if system is None:
        return None

    if isinstance(system, str):
        if len(system) < _MIN_CACHE_CHARS:
            return system
        return [{"type": "text", "text": system, "cache_control": _EPHEMERAL}]

    # list[dict] path
    total_chars = sum(_text_len(b) for b in system)
    if total_chars < _MIN_CACHE_CHARS:
        return system

    # Place a breakpoint on the last text block that does not already have one.
    result = [dict(b) for b in system]  # shallow copy so we don't mutate caller
    for block in reversed(result):
        if block.get("type") == "text" and "cache_control" not in block:
            block["cache_control"] = _EPHEMERAL
            break
    return result


def apply_auto_cache_to_tools(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return *tools* with `cache_control` placed on the last tool definition.

    Tool schemas are static across a session, making them ideal cache
    candidates.  Adding a breakpoint to the final entry covers the entire
    tool list in one shot.

    Args:
        tools: List of Anthropic tool definition dicts.

    Returns:
        The (possibly modified) tools list.
    """
    if not tools:
        return tools

    result = [dict(t) for t in tools]
    last = result[-1]
    if "cache_control" not in last:
        last["cache_control"] = _EPHEMERAL
    return result


@dataclass
class TokenUsageStats:
    """Accumulated token usage across multiple invocations."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, usage: dict[str, Any]) -> None:
        """Merge a single usage dict into the running totals.

        Args:
            usage: Anthropic `UsageMetadata`-like dict with optional keys
                ``input_tokens``, ``output_tokens``,
                ``cache_creation_input_tokens``, and
                ``cache_read_input_tokens``.
        """
        with self._lock:
            self.input_tokens += usage.get("input_tokens", 0)
            self.output_tokens += usage.get("output_tokens", 0)
            self.cache_creation_tokens += usage.get(
                "cache_creation_input_tokens", 0
            )
            self.cache_read_tokens += usage.get("cache_read_input_tokens", 0)

    @property
    def total_tokens(self) -> int:
        """Sum of all input and output tokens."""
        return self.input_tokens + self.output_tokens

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of input tokens served from cache (0.0 – 1.0)."""
        total_input = self.input_tokens + self.cache_read_tokens
        if total_input == 0:
            return 0.0
        return self.cache_read_tokens / total_input

    def summary(self) -> dict[str, Any]:
        """Return a snapshot of all counters.

        Returns:
            A plain dict with keys ``input_tokens``, ``output_tokens``,
            ``cache_creation_tokens``, ``cache_read_tokens``,
            ``total_tokens``, and ``cache_hit_rate``.
        """
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "total_tokens": self.total_tokens,
            "cache_hit_rate": self.cache_hit_rate,
        }
