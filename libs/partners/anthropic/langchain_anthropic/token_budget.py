"""Token-budget-aware connector routing for Anthropic models.

`TokenBudgetRouter` wraps a `ConnectorOrchestrator` and automatically
promotes requests to a cheaper (faster, smaller) model once a configurable
token budget has been consumed.  This is the main tool for staying within
free-tier or cost-cap limits across a session.

Typical setup::

    from langchain_anthropic import ChatAnthropic
    from langchain_anthropic.token_budget import TokenBudgetRouter, ModelTier
    from langchain_core.runnables.orchestrator import ConnectorOrchestrator

    router = TokenBudgetRouter(
        tiers=[
            ModelTier(
                name="opus",
                connector=ChatAnthropic(model="claude-opus-4-7", auto_cache=True),
                budget_tokens=50_000,
            ),
            ModelTier(
                name="haiku",
                connector=ChatAnthropic(model="claude-haiku-4-5", auto_cache=True),
            ),
        ]
    )

    # Works like any Runnable
    response = router.invoke(messages)
    print(router.usage_summary())
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain_core.runnables.base import Runnable
from langchain_core.runnables.config import RunnableConfig, ensure_config
from langchain_core.runnables.orchestrator import ConnectorOrchestrator
from langchain_core.runnables.utils import Input, Output

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator


@dataclass
class ModelTier:
    """A connector with an optional token budget threshold.

    When `budget_tokens` is set, the `TokenBudgetRouter` uses this
    connector until the cumulative session token count exceeds the budget,
    then falls back to the next tier.  The last tier in the list has no
    budget cap and is used for the remainder of the session.

    Attributes:
        name: Human-readable label for logging and the usage report.
        connector: Any LangChain `Runnable` — typically a `ChatAnthropic`
            instance with `auto_cache=True`.
        budget_tokens: Soft token cap for this tier.  `None` means
            unlimited (suitable for the final fallback tier).
    """

    name: str
    connector: Runnable
    budget_tokens: int | None = None


class TokenBudgetRouter(Runnable[Input, Output]):
    """Routes requests to cheaper models as the token budget is consumed.

    Internally uses a `ConnectorOrchestrator` per tier so each tier still
    benefits from circuit-breaking and self-healing.

    Usage counters are updated from `AIMessage.usage_metadata` when
    available; otherwise a rough character-based estimate is used as a
    fallback so the budget is always tracked even without exact counts.

    Args:
        tiers: Ordered list of `ModelTier` objects, from most capable
            (highest budget) to cheapest (final fallback).
        estimate_chars_per_token: Characters-per-token ratio used for the
            rough estimate when exact metadata is unavailable.  Defaults to
            4, which is a reasonable approximation for English text.

    Raises:
        ValueError: If `tiers` is empty or no tier lacks a `budget_tokens`
            cap (i.e., there is no unlimited fallback).
    """

    def __init__(
        self,
        tiers: list[ModelTier],
        *,
        estimate_chars_per_token: int = 4,
    ) -> None:
        if not tiers:
            msg = "At least one ModelTier is required."
            raise ValueError(msg)
        if all(t.budget_tokens is not None for t in tiers):
            msg = (
                "The last tier must have budget_tokens=None so there is always "
                "an unlimited fallback."
            )
            raise ValueError(msg)

        self._tiers = tiers
        self._estimate_chars_per_token = estimate_chars_per_token
        self._session_tokens: int = 0
        self._lock = threading.Lock()
        # Wrap each tier in a single-connector orchestrator so we get
        # circuit-breaking and self-healing for free.
        self._orchestrators: list[ConnectorOrchestrator] = [
            ConnectorOrchestrator(connectors={t.name: t.connector})
            for t in tiers
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _active_orchestrator(self) -> ConnectorOrchestrator:
        """Return the orchestrator for the current budget tier."""
        with self._lock:
            consumed = self._session_tokens
        for tier, orch in zip(self._tiers, self._orchestrators):
            if tier.budget_tokens is None or consumed < tier.budget_tokens:
                return orch
        # All budgeted tiers exhausted — use the last one (unlimited fallback).
        return self._orchestrators[-1]

    def _update_from_result(self, result: Any, fallback_input: Any) -> None:
        """Extract token count from result metadata or fall back to estimation."""
        tokens = 0
        if hasattr(result, "usage_metadata") and result.usage_metadata:
            meta = result.usage_metadata
            tokens = (
                meta.get("input_tokens", 0)
                + meta.get("output_tokens", 0)
                + meta.get("cache_read_input_tokens", 0)
            )
        if not tokens:
            # Rough estimate from input length
            raw = str(fallback_input)
            tokens = max(1, len(raw) // self._estimate_chars_per_token)
        with self._lock:
            self._session_tokens += tokens

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def invoke(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Output:
        """Invoke the active-budget connector.

        Args:
            input: Runnable input (messages, string, dict, …).
            config: Optional `RunnableConfig`.
            **kwargs: Forwarded to the connector.

        Returns:
            Output from the selected connector.
        """
        config = ensure_config(config)
        orch = self._active_orchestrator()
        result = orch.invoke(input, config, **kwargs)
        self._update_from_result(result, input)
        return result

    async def ainvoke(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Output:
        """Async version of :meth:`invoke`.

        Args:
            input: Runnable input.
            config: Optional `RunnableConfig`.
            **kwargs: Forwarded to the connector.

        Returns:
            Output from the selected connector.
        """
        config = ensure_config(config)
        orch = self._active_orchestrator()
        result = await orch.ainvoke(input, config, **kwargs)
        self._update_from_result(result, input)
        return result

    def stream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Output]:
        """Stream from the active-budget connector.

        Args:
            input: Runnable input.
            config: Optional `RunnableConfig`.
            **kwargs: Forwarded to the connector.

        Yields:
            Output chunks.
        """
        config = ensure_config(config)
        orch = self._active_orchestrator()
        chunks: list[Output] = []
        for chunk in orch.stream(input, config, **kwargs):
            chunks.append(chunk)
            yield chunk
        # Use the last chunk for metadata extraction (usually carries usage info).
        if chunks:
            self._update_from_result(chunks[-1], input)

    async def astream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Output]:
        """Async streaming version of :meth:`stream`.

        Args:
            input: Runnable input.
            config: Optional `RunnableConfig`.
            **kwargs: Forwarded to the connector.

        Yields:
            Output chunks.
        """
        config = ensure_config(config)
        orch = self._active_orchestrator()
        last_chunk: Output | None = None
        async for chunk in orch.astream(input, config, **kwargs):
            last_chunk = chunk
            yield chunk
        if last_chunk is not None:
            self._update_from_result(last_chunk, input)

    def reset_budget(self) -> None:
        """Reset the session token counter to zero.

        Useful when starting a fresh conversation that should benefit from
        the full budget again.
        """
        with self._lock:
            self._session_tokens = 0

    def usage_summary(self) -> dict[str, Any]:
        """Return a snapshot of session-level usage and current tier.

        Returns:
            Dict with ``session_tokens``, ``active_tier``, and
            ``tier_budgets`` (list of ``{name, budget, consumed}`` dicts).
        """
        with self._lock:
            consumed = self._session_tokens

        active_name = "unknown"
        for tier, orch in zip(self._tiers, self._orchestrators):
            if tier.budget_tokens is None or consumed < tier.budget_tokens:
                active_name = tier.name
                break

        return {
            "session_tokens": consumed,
            "active_tier": active_name,
            "tier_budgets": [
                {
                    "name": t.name,
                    "budget": t.budget_tokens,
                    "consumed": consumed,
                }
                for t in self._tiers
            ],
        }
