"""Health-score-aware connector orchestrator.

`HealthAwareOrchestrator` extends `ConnectorOrchestrator` with real-time
latency intelligence: instead of routing to connectors in fixed registration
order, it consults an `EndpointHealthChecker` and routes to whichever connector
currently has the highest composite stability score.

The circuit-breaker logic from the base class is preserved — a connector
that trips its breaker is still skipped regardless of its health score.

Quick start::

    import asyncio, os
    from langchain_openai import ChatOpenAI
    from langchain_core.runnables.free_models import FREE_CODING_MODELS, by_tier
    from langchain_core.runnables.health_check import EndpointHealthChecker
    from langchain_core.runnables.health_aware_orchestrator import (
        HealthAwareOrchestrator,
        build_free_model_orchestrator,
    )

    # Convenience builder — creates connectors from catalog metadata
    orch = build_free_model_orchestrator(
        min_tier="A",
        api_keys={
            "OPENROUTER_API_KEY": os.environ["OPENROUTER_API_KEY"],
            "GROQ_API_KEY":       os.environ["GROQ_API_KEY"],
        },
    )

    # Warm up the health data, then invoke
    await orch.warm_up()
    response = await orch.ainvoke("Write a binary search in Python.")
    print(orch.routing_report())
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.runnables.base import Runnable
from langchain_core.runnables.free_models import (
    FREE_CODING_MODELS,
    CodingTierLabel,
    FreeModelEntry,
    by_tier,
    probe_endpoints,
)
from langchain_core.runnables.health_check import EndpointHealthChecker, StabilityMetrics
from langchain_core.runnables.orchestrator import ConnectorOrchestrator

logger = logging.getLogger(__name__)


class HealthAwareOrchestrator(ConnectorOrchestrator):
    """Routes requests to the connector with the highest live stability score.

    Inherits all self-healing behaviour from `ConnectorOrchestrator` (context
    overflow recovery, circuit breaking, exponential back-off) and adds:

    - **Score-based routing** — connectors are ordered by their current
      composite stability score from the `EndpointHealthChecker`.
    - **Warm-up** — call :meth:`warm_up` once to run a quick burst of health
      probes before your first real request.
    - **Background polling** — call :meth:`start_background_polling` to keep
      health data fresh throughout the lifetime of your application.
    - **Routing report** — :meth:`routing_report` shows which connector would
      be chosen and why.

    When no health data is available for a connector it falls back to the
    base-class registration-order routing, so the orchestrator is usable from
    the very first invocation.

    Args:
        connectors: Mapping of name → `Runnable`, same as `ConnectorOrchestrator`.
        health_checker: A configured `EndpointHealthChecker`.  The endpoint
            names in *health_checker* must match the keys in *connectors*.
        **kwargs: Forwarded to `ConnectorOrchestrator.__init__`.
    """

    def __init__(
        self,
        connectors: dict[str, Runnable],
        health_checker: EndpointHealthChecker,
        **kwargs: Any,
    ) -> None:
        super().__init__(connectors=connectors, **kwargs)
        self._health_checker = health_checker

    # ------------------------------------------------------------------
    # Override routing to use health scores
    # ------------------------------------------------------------------

    def _ordered_connectors(self) -> list[tuple[str, Runnable]]:  # type: ignore[override]
        """Order connectors by composite health score (best first).

        Connectors with no health data yet are placed after scored ones in
        registration order.  OPEN-circuit connectors are placed last regardless
        of score (mirrors base-class behaviour).
        """
        import threading

        with self._lock:
            priority = list(self._priority)
            connectors = dict(self._connectors)
            health = dict(self._health)

        all_metrics = self._health_checker.get_all_metrics()

        def _sort_key(name: str) -> tuple[int, float]:
            is_suspended = name in health and not health[name].is_available
            score = all_metrics[name].composite_score if name in all_metrics else -1.0
            # (suspended=1 sorts after non-suspended=0; score negated so higher is first)
            return (int(is_suspended), -score)

        ordered = sorted(
            [
                (n, connectors[n])
                for n in priority
                if n in connectors and n in health
            ],
            key=lambda pair: _sort_key(pair[0]),
        )
        return ordered

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    async def warm_up(self, rounds: int = 3) -> None:
        """Run a quick burst of health probes to populate routing scores.

        Should be called once before the first real request if you want
        score-based routing from the start.

        Args:
            rounds: Number of parallel ping rounds to run.
        """
        for _ in range(rounds):
            await self._health_checker.ping_all()
        logger.info("Health warm-up complete (%d rounds).", rounds)

    async def start_background_polling(
        self, interval_seconds: float = 12.0
    ) -> None:
        """Start continuous background health polling.

        Args:
            interval_seconds: Steady-state seconds between poll rounds.
        """
        await self._health_checker.start_background_polling(
            interval_seconds=interval_seconds
        )

    def stop_background_polling(self) -> None:
        """Stop the background polling task."""
        self._health_checker.stop_background_polling()

    def routing_report(self) -> dict[str, Any]:
        """Return a snapshot showing current health scores and routing order.

        Returns:
            Dict with ``active_connector`` (name that would be chosen next),
            ``scores`` (ordered list of ``{name, score, tier, available}``),
            and ``circuit_breakers`` (health report from base class).
        """
        ordered = self._ordered_connectors()
        all_metrics = self._health_checker.get_all_metrics()

        scores = []
        for name, _ in ordered:
            m = all_metrics.get(name)
            scores.append(
                {
                    "name": name,
                    "composite_score": round(m.composite_score, 3) if m else None,
                    "p95_ms": round(m.p95_ms, 1) if m else None,
                    "availability": round(m.availability_rate, 3) if m else None,
                    "circuit_breaker": self._health.get(name, None) and
                                       self._health[name].summary()["state"],
                }
            )

        return {
            "active_connector": ordered[0][0] if ordered else None,
            "scores": scores,
            "circuit_breakers": self.health_report(),
        }


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------


def build_free_model_orchestrator(
    *,
    min_tier: CodingTierLabel = "A",
    max_models: int = 6,
    api_keys: dict[str, str] | None = None,
    extra_connectors: dict[str, Runnable] | None = None,
    poll_interval_seconds: float = 15.0,
    **orchestrator_kwargs: Any,
) -> HealthAwareOrchestrator:
    """Build a `HealthAwareOrchestrator` from the built-in free model catalog.

    Selects up to *max_models* entries from :data:`~langchain_core.runnables.free_models.FREE_CODING_MODELS`
    at or above *min_tier*, creates OpenAI-compatible `ChatOpenAI` connectors
    for each one (requires ``langchain-openai`` to be installed), and wires in
    a live `EndpointHealthChecker`.

    The function does **not** start background polling automatically.  Call
    :meth:`HealthAwareOrchestrator.start_background_polling` or
    :meth:`HealthAwareOrchestrator.warm_up` after construction.

    Args:
        min_tier: Minimum SWE-bench tier to include (e.g. ``"A"`` or ``"S"``).
        max_models: Cap on the number of connectors to create.
        api_keys: Dict of ``{ENV_VAR_NAME: value}`` overrides.  Falls back to
            ``os.environ`` for any key not supplied.
        extra_connectors: Additional `Runnable` instances (e.g. a
            `ChatAnthropic` with `auto_cache=True`) to add to the pool.
        poll_interval_seconds: Passed through to the health checker; not started
            automatically.
        **orchestrator_kwargs: Forwarded to `HealthAwareOrchestrator.__init__`.

    Returns:
        A configured `HealthAwareOrchestrator` ready for :meth:`~HealthAwareOrchestrator.warm_up`
        or immediate use (with registration-order routing until first pings complete).

    Raises:
        ImportError: If ``langchain-openai`` is not installed.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        msg = (
            "build_free_model_orchestrator() requires 'langchain-openai'. "
            "Install it with: pip install langchain-openai"
        )
        raise ImportError(msg) from exc

    keys = dict(os.environ)
    if api_keys:
        keys.update(api_keys)

    candidates: list[FreeModelEntry] = by_tier(FREE_CODING_MODELS, min_tier)

    # Only include models whose API key env var is actually set.
    available = [m for m in candidates if keys.get(m.api_key_env)][:max_models]

    if not available and not extra_connectors:
        msg = (
            "No free models available — set at least one of: "
            + ", ".join(sorted({m.api_key_env for m in candidates}))
        )
        raise RuntimeError(msg)

    connectors: dict[str, Runnable] = {}
    for entry in available:
        connectors[entry.id] = ChatOpenAI(  # type: ignore[call-arg]
            base_url=entry.api_base,
            model=entry.model_id,
            api_key=keys[entry.api_key_env],
            streaming=True,
        )

    if extra_connectors:
        connectors.update(extra_connectors)

    endpoints = {
        entry.id: entry.probe_url for entry in available
    }
    checker = EndpointHealthChecker(
        endpoints=endpoints,
        connect_timeout=5.0,
        read_timeout=8.0,
    )

    return HealthAwareOrchestrator(
        connectors=connectors,
        health_checker=checker,
        **orchestrator_kwargs,
    )
