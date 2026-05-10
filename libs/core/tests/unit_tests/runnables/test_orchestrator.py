"""Unit tests for ConnectorOrchestrator."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.orchestrator import (
    ConnectorOrchestrator,
    _CircuitState,
    _ConnectorHealth,
    _trim_messages_by_fraction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok_runnable(value: str = "ok") -> RunnableLambda:
    return RunnableLambda(lambda _: value)


def _failing_runnable(exc: Exception | None = None) -> RunnableLambda:
    error = exc or RuntimeError("boom")
    def _raise(_: Any) -> Any:
        raise error
    return RunnableLambda(_raise)


# ---------------------------------------------------------------------------
# _ConnectorHealth / circuit breaker
# ---------------------------------------------------------------------------


def test_health_starts_closed() -> None:
    h = _ConnectorHealth(failure_threshold=3, recovery_timeout=60)
    assert h.state == _CircuitState.CLOSED
    assert h.is_available


def test_health_opens_after_threshold() -> None:
    h = _ConnectorHealth(failure_threshold=3, recovery_timeout=60)
    for _ in range(3):
        h.record_failure()
    assert h.state == _CircuitState.OPEN
    assert not h.is_available


def test_health_closes_after_success() -> None:
    h = _ConnectorHealth(failure_threshold=3, recovery_timeout=60)
    for _ in range(3):
        h.record_failure()
    # Simulate recovery timeout elapsed
    h._last_failure_time = 0.0
    h._recovery_timeout = 0.0
    assert h.state == _CircuitState.HALF_OPEN
    h.record_success()
    assert h.state == _CircuitState.CLOSED


def test_health_summary_keys() -> None:
    h = _ConnectorHealth(failure_threshold=3, recovery_timeout=60)
    assert set(h.summary()) == {"state", "consecutive_failures", "success_rate"}


def test_health_success_rate_with_mixed_results() -> None:
    h = _ConnectorHealth(failure_threshold=10, recovery_timeout=60, window=4)
    h.record_success()
    h.record_success()
    h.record_failure()
    h.record_failure()
    assert h.success_rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _trim_messages_by_fraction
# ---------------------------------------------------------------------------


def test_trim_keeps_system_message() -> None:
    msgs = [
        SystemMessage(content="system"),
        HumanMessage(content="a"),
        HumanMessage(content="b"),
        HumanMessage(content="c"),
        HumanMessage(content="d"),
    ]
    trimmed = _trim_messages_by_fraction(msgs, 0.5)
    assert trimmed[0].type == "system"
    assert len(trimmed) < len(msgs)


def test_trim_without_system() -> None:
    msgs = [HumanMessage(content=str(i)) for i in range(8)]
    trimmed = _trim_messages_by_fraction(msgs, 0.25)
    assert len(trimmed) < len(msgs)
    assert len(trimmed) >= 1


def test_trim_single_message_returns_it() -> None:
    msgs = [HumanMessage(content="only")]
    trimmed = _trim_messages_by_fraction(msgs, 0.5)
    assert trimmed == msgs


# ---------------------------------------------------------------------------
# ConnectorOrchestrator construction
# ---------------------------------------------------------------------------


def test_requires_at_least_one_connector() -> None:
    with pytest.raises(ValueError, match="At least one connector"):
        ConnectorOrchestrator(connectors={})


def test_basic_invoke_success() -> None:
    orch = ConnectorOrchestrator(connectors={"a": _ok_runnable("result")})
    assert orch.invoke("input") == "result"


def test_register_new_connector() -> None:
    orch = ConnectorOrchestrator(connectors={"a": _ok_runnable("a")})
    orch.register("b", _ok_runnable("b"))
    assert "b" in orch._connectors
    assert "b" in orch._priority


def test_unregister_connector() -> None:
    orch = ConnectorOrchestrator(connectors={"a": _ok_runnable(), "b": _ok_runnable()})
    orch.unregister("b")
    assert "b" not in orch._connectors


def test_unregister_nonexistent_raises() -> None:
    orch = ConnectorOrchestrator(connectors={"a": _ok_runnable()})
    with pytest.raises(KeyError):
        orch.unregister("missing")


def test_health_report_keys() -> None:
    orch = ConnectorOrchestrator(connectors={"a": _ok_runnable()})
    report = orch.health_report()
    assert "a" in report
    assert set(report["a"]) == {"state", "consecutive_failures", "success_rate"}


# ---------------------------------------------------------------------------
# Failure fallover
# ---------------------------------------------------------------------------


def test_failover_to_second_connector() -> None:
    orch = ConnectorOrchestrator(
        connectors={
            "bad": _failing_runnable(),
            "good": _ok_runnable("rescued"),
        },
        max_retries=2,
    )
    result = orch.invoke("input")
    assert result == "rescued"


def test_all_connectors_fail_raises() -> None:
    orch = ConnectorOrchestrator(
        connectors={"a": _failing_runnable(), "b": _failing_runnable()},
        max_retries=1,
    )
    with pytest.raises(RuntimeError, match="boom"):
        orch.invoke("input")


def test_circuit_breaker_skips_open_connector() -> None:
    """After threshold failures the faulty connector should be deprioritised."""
    call_log: list[str] = []

    def make_tracker(name: str, ok: bool) -> RunnableLambda:
        def _fn(_: Any) -> str:
            call_log.append(name)
            if not ok:
                raise RuntimeError(f"{name} failed")
            return f"{name} ok"
        return RunnableLambda(_fn)

    orch = ConnectorOrchestrator(
        connectors={
            "bad": make_tracker("bad", ok=False),
            "good": make_tracker("good", ok=True),
        },
        failure_threshold=3,
        max_retries=5,
    )
    # Exhaust the bad connector's circuit breaker
    for _ in range(3):
        try:
            orch.invoke("x")
        except RuntimeError:
            pass
    call_log.clear()

    # Now the good connector should be tried first (bad is OPEN)
    result = orch.invoke("x")
    assert result == "good ok"
    # bad connector should NOT have been called
    assert "bad" not in call_log


# ---------------------------------------------------------------------------
# Context-overflow self-healing
# ---------------------------------------------------------------------------


def test_context_overflow_triggers_trim_and_retry() -> None:
    """On ContextOverflowError the orchestrator trims the history and retries."""
    call_count = 0

    def _model(messages: Any) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ContextOverflowError("too long")
        return "ok after trim"

    orch = ConnectorOrchestrator(
        connectors={"llm": RunnableLambda(_model)},
        max_retries=2,
    )
    msgs = [HumanMessage(content=str(i)) for i in range(10)]
    result = orch.invoke(msgs)
    assert result == "ok after trim"
    assert call_count == 2


# ---------------------------------------------------------------------------
# Async paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_invoke_success() -> None:
    orch = ConnectorOrchestrator(connectors={"a": _ok_runnable("async-result")})
    result = await orch.ainvoke("input")
    assert result == "async-result"


@pytest.mark.asyncio
async def test_async_failover() -> None:
    orch = ConnectorOrchestrator(
        connectors={
            "bad": _failing_runnable(),
            "good": _ok_runnable("async-rescued"),
        },
        max_retries=2,
    )
    result = await orch.ainvoke("input")
    assert result == "async-rescued"


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_stream_yields_chunks() -> None:
    orch = ConnectorOrchestrator(connectors={"a": _ok_runnable("chunk")})
    chunks = list(orch.stream("input"))
    assert chunks == ["chunk"]


def test_stream_fallback_on_error() -> None:
    orch = ConnectorOrchestrator(
        connectors={
            "bad": _failing_runnable(),
            "good": _ok_runnable("streamed"),
        },
        max_retries=2,
    )
    chunks = list(orch.stream("input"))
    assert "streamed" in chunks


def test_stream_all_fail_raises() -> None:
    orch = ConnectorOrchestrator(
        connectors={"a": _failing_runnable()},
        max_retries=1,
    )
    with pytest.raises(RuntimeError, match="All connectors failed"):
        list(orch.stream("input"))
