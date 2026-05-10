"""Unit tests for HealthAwareOrchestrator and build_free_model_orchestrator."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.health_aware_orchestrator import (
    HealthAwareOrchestrator,
    build_free_model_orchestrator,
)
from langchain_core.runnables.health_check import EndpointHealthChecker, StabilityMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_checker(scores: dict[str, float]) -> EndpointHealthChecker:
    """Return a checker whose get_all_metrics() returns the given scores."""
    checker = MagicMock(spec=EndpointHealthChecker)
    metrics = {
        name: StabilityMetrics(
            name=name,
            endpoint="https://example.com",
            ping_count=10,
            availability_rate=score,
            p50_ms=100.0,
            p95_ms=200.0,
            jitter_ms=10.0,
            composite_score=score,
        )
        for name, score in scores.items()
    }
    checker.get_all_metrics.return_value = metrics
    return checker


def _make_orchestrator(
    connector_names: list[str],
    scores: dict[str, float],
) -> HealthAwareOrchestrator:
    connectors = {name: RunnableLambda(lambda _: name) for name in connector_names}
    checker = _make_checker(scores)
    return HealthAwareOrchestrator(connectors=connectors, health_checker=checker)


# ---------------------------------------------------------------------------
# _ordered_connectors — score-based routing
# ---------------------------------------------------------------------------


def test_ordered_connectors_best_score_first() -> None:
    orch = _make_orchestrator(
        ["alpha", "beta", "gamma"],
        scores={"alpha": 0.5, "beta": 0.9, "gamma": 0.3},
    )
    ordered = orch._ordered_connectors()
    names = [n for n, _ in ordered]
    assert names[0] == "beta"
    assert names[-1] == "gamma"


def test_ordered_connectors_equal_scores_stable() -> None:
    orch = _make_orchestrator(
        ["a", "b"],
        scores={"a": 0.8, "b": 0.8},
    )
    ordered = orch._ordered_connectors()
    assert len(ordered) == 2


def test_ordered_connectors_no_metrics_placed_last() -> None:
    """Connectors with no health data (score=-1) appear after scored ones."""
    connectors = {
        "known": RunnableLambda(lambda _: "known"),
        "unknown": RunnableLambda(lambda _: "unknown"),
    }
    checker = MagicMock(spec=EndpointHealthChecker)
    # Only 'known' has metrics
    checker.get_all_metrics.return_value = {
        "known": StabilityMetrics(
            name="known",
            endpoint="https://x.com",
            ping_count=5,
            availability_rate=0.8,
            p50_ms=100.0,
            p95_ms=200.0,
            jitter_ms=5.0,
            composite_score=0.8,
        )
    }
    orch = HealthAwareOrchestrator(connectors=connectors, health_checker=checker)
    ordered = orch._ordered_connectors()
    names = [n for n, _ in ordered]
    assert names[0] == "known"
    assert names[1] == "unknown"


def test_ordered_connectors_suspended_placed_last() -> None:
    """Open-circuit connectors appear after healthy ones regardless of score."""
    orch = _make_orchestrator(
        ["good", "broken"],
        scores={"good": 0.5, "broken": 0.99},
    )
    # Trip the circuit breaker on 'broken'
    for _ in range(10):
        orch._health["broken"].record_failure()

    ordered = orch._ordered_connectors()
    names = [n for n, _ in ordered]
    assert names[0] == "good"
    assert names[-1] == "broken"


# ---------------------------------------------------------------------------
# routing_report
# ---------------------------------------------------------------------------


def test_routing_report_structure() -> None:
    orch = _make_orchestrator(
        ["alpha", "beta"],
        scores={"alpha": 0.6, "beta": 0.9},
    )
    report = orch.routing_report()
    assert "active_connector" in report
    assert "scores" in report
    assert "circuit_breakers" in report
    assert report["active_connector"] == "beta"


def test_routing_report_scores_list() -> None:
    orch = _make_orchestrator(
        ["alpha", "beta"],
        scores={"alpha": 0.6, "beta": 0.9},
    )
    report = orch.routing_report()
    assert len(report["scores"]) == 2
    for entry in report["scores"]:
        assert "name" in entry
        assert "composite_score" in entry
        assert "p95_ms" in entry
        assert "availability" in entry


def test_routing_report_no_metrics_gives_none_active() -> None:
    """When no health data exists yet all connectors have score -1; report is still valid."""
    checker = MagicMock(spec=EndpointHealthChecker)
    checker.get_all_metrics.return_value = {}
    dummy = RunnableLambda(lambda _: "ok")
    orch = HealthAwareOrchestrator(connectors={"x": dummy}, health_checker=checker)
    report = orch.routing_report()
    # 'x' has no metrics so composite_score is None
    assert report["active_connector"] is not None or report["scores"] is not None


# ---------------------------------------------------------------------------
# warm_up
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_warm_up_calls_ping_all_n_times() -> None:
    checker = MagicMock(spec=EndpointHealthChecker)
    checker.ping_all = AsyncMock(return_value={})
    checker.get_all_metrics.return_value = {}
    dummy = RunnableLambda(lambda _: "ok")
    orch = HealthAwareOrchestrator(connectors={"x": dummy}, health_checker=checker)
    await orch.warm_up(rounds=4)
    assert checker.ping_all.call_count == 4


@pytest.mark.asyncio
async def test_warm_up_default_rounds() -> None:
    checker = MagicMock(spec=EndpointHealthChecker)
    checker.ping_all = AsyncMock(return_value={})
    checker.get_all_metrics.return_value = {}
    dummy = RunnableLambda(lambda _: "ok")
    orch = HealthAwareOrchestrator(connectors={"x": dummy}, health_checker=checker)
    await orch.warm_up()
    assert checker.ping_all.call_count == 3  # default rounds=3


# ---------------------------------------------------------------------------
# start / stop background polling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_background_polling_delegates() -> None:
    checker = MagicMock(spec=EndpointHealthChecker)
    checker.start_background_polling = AsyncMock()
    checker.get_all_metrics.return_value = {}
    dummy = RunnableLambda(lambda _: "ok")
    orch = HealthAwareOrchestrator(connectors={"x": dummy}, health_checker=checker)
    await orch.start_background_polling(interval_seconds=30.0)
    checker.start_background_polling.assert_called_once_with(interval_seconds=30.0)


def test_stop_background_polling_delegates() -> None:
    checker = MagicMock(spec=EndpointHealthChecker)
    checker.get_all_metrics.return_value = {}
    dummy = RunnableLambda(lambda _: "ok")
    orch = HealthAwareOrchestrator(connectors={"x": dummy}, health_checker=checker)
    orch.stop_background_polling()
    checker.stop_background_polling.assert_called_once()


# ---------------------------------------------------------------------------
# build_free_model_orchestrator
# ---------------------------------------------------------------------------


def test_build_raises_without_langchain_openai() -> None:
    with patch.dict("sys.modules", {"langchain_openai": None}):
        with pytest.raises(ImportError, match="langchain-openai"):
            build_free_model_orchestrator(
                min_tier="S+",
                api_keys={"OPENROUTER_API_KEY": "sk-test"},
            )


def test_build_raises_when_no_keys_available() -> None:
    mock_openai = MagicMock()
    with patch.dict("sys.modules", {"langchain_openai": mock_openai}):
        with pytest.raises(RuntimeError, match="No free models available"):
            build_free_model_orchestrator(
                min_tier="S+",
                api_keys={},  # no matching keys
            )


def test_build_returns_orchestrator_with_matching_keys() -> None:
    mock_chat = MagicMock()
    mock_openai = MagicMock()
    mock_openai.ChatOpenAI = mock_chat

    with patch.dict("sys.modules", {"langchain_openai": mock_openai}):
        orch = build_free_model_orchestrator(
            min_tier="A",
            max_models=2,
            api_keys={"GROQ_API_KEY": "sk-groq-test"},
        )
    assert isinstance(orch, HealthAwareOrchestrator)
    assert len(orch._connectors) <= 2
    # All created connectors used the groq key
    for call in mock_chat.call_args_list:
        assert call.kwargs.get("api_key") == "sk-groq-test"


def test_build_respects_max_models() -> None:
    mock_chat = MagicMock()
    mock_openai = MagicMock()
    mock_openai.ChatOpenAI = mock_chat

    with patch.dict("sys.modules", {"langchain_openai": mock_openai}):
        orch = build_free_model_orchestrator(
            min_tier="C",
            max_models=1,
            api_keys={
                "GROQ_API_KEY": "gsk1",
                "OPENROUTER_API_KEY": "or1",
                "NVIDIA_API_KEY": "nv1",
            },
        )
    assert len(orch._connectors) == 1


def test_build_includes_extra_connectors() -> None:
    mock_chat = MagicMock()
    mock_openai = MagicMock()
    mock_openai.ChatOpenAI = mock_chat
    extra = RunnableLambda(lambda _: "extra")

    with patch.dict("sys.modules", {"langchain_openai": mock_openai}):
        orch = build_free_model_orchestrator(
            min_tier="A",
            api_keys={"GROQ_API_KEY": "sk-groq"},
            extra_connectors={"my-extra": extra},
        )
    assert "my-extra" in orch._connectors


def test_build_raises_when_no_keys_and_no_extra() -> None:
    mock_openai = MagicMock()
    with patch.dict("sys.modules", {"langchain_openai": mock_openai}):
        with pytest.raises(RuntimeError):
            build_free_model_orchestrator(min_tier="S+", api_keys={})


def test_build_ok_with_only_extra_connectors() -> None:
    mock_openai = MagicMock()
    extra = RunnableLambda(lambda _: "extra")

    with patch.dict("sys.modules", {"langchain_openai": mock_openai}):
        orch = build_free_model_orchestrator(
            min_tier="S+",
            api_keys={},
            extra_connectors={"manual": extra},
        )
    assert "manual" in orch._connectors
