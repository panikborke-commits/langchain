"""Unit tests for EndpointHealthChecker and StabilityMetrics."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.runnables.health_check import (
    EndpointHealthChecker,
    PingResult,
    StabilityMetrics,
    _compute_composite,
)


# ---------------------------------------------------------------------------
# _compute_composite
# ---------------------------------------------------------------------------


def test_composite_perfect_endpoint() -> None:
    # 100% availability, 0ms p95, 0 jitter → max score
    score = _compute_composite(availability=1.0, p95_ms=0.0, jitter_ms=0.0, p50_ms=1.0)
    assert score == pytest.approx(1.0)


def test_composite_dead_endpoint() -> None:
    # 0% availability, maxed latency, max jitter → near-zero score
    score = _compute_composite(availability=0.0, p95_ms=8000.0, jitter_ms=9999.0, p50_ms=1.0)
    assert score == pytest.approx(0.0)


def test_composite_weights_sum_to_one() -> None:
    # With perfect speed and consistency, availability dominates proportionally
    score_full = _compute_composite(1.0, 0.0, 0.0, 1.0)
    score_half_avail = _compute_composite(0.5, 0.0, 0.0, 1.0)
    # availability weight is 0.40
    assert score_full - score_half_avail == pytest.approx(0.40 * 0.5, abs=1e-9)


def test_composite_clamped_at_ceiling() -> None:
    # p95 above ceiling still yields 0 speed component
    score_at = _compute_composite(1.0, 8000.0, 0.0, 1.0)
    score_over = _compute_composite(1.0, 99999.0, 0.0, 1.0)
    assert score_at == pytest.approx(score_over)


def test_composite_p50_zero_consistency() -> None:
    # p50 == 0 → consistency component is 0
    score = _compute_composite(1.0, 0.0, 0.0, p50_ms=0.0)
    # availability (0.40) + speed (0.35) only
    assert score == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# EndpointHealthChecker — setup
# ---------------------------------------------------------------------------


@pytest.fixture()
def checker() -> EndpointHealthChecker:
    return EndpointHealthChecker(
        endpoints={
            "alpha": "https://alpha.example.com/v1/models",
            "beta": "https://beta.example.com/v1/models",
        },
        window_size=5,
        connect_timeout=2.0,
        read_timeout=4.0,
    )


# ---------------------------------------------------------------------------
# add / remove endpoint
# ---------------------------------------------------------------------------


def test_add_endpoint(checker: EndpointHealthChecker) -> None:
    checker.add_endpoint("gamma", "https://gamma.example.com/v1/models")
    assert "gamma" in checker._endpoints
    assert "gamma" in checker._history


def test_add_endpoint_idempotent(checker: EndpointHealthChecker) -> None:
    checker.add_endpoint("alpha", "https://alpha2.example.com/v1/models")
    assert checker._endpoints["alpha"] == "https://alpha2.example.com/v1/models"


def test_remove_endpoint(checker: EndpointHealthChecker) -> None:
    checker.remove_endpoint("alpha")
    assert "alpha" not in checker._endpoints
    assert "alpha" not in checker._history


def test_remove_unknown_raises(checker: EndpointHealthChecker) -> None:
    with pytest.raises(KeyError):
        checker.remove_endpoint("nonexistent")


# ---------------------------------------------------------------------------
# get_metrics — no data yet
# ---------------------------------------------------------------------------


def test_get_metrics_returns_none_before_pings(checker: EndpointHealthChecker) -> None:
    assert checker.get_metrics("alpha") is None


def test_get_all_metrics_empty_before_pings(checker: EndpointHealthChecker) -> None:
    assert checker.get_all_metrics() == {}


# ---------------------------------------------------------------------------
# ping_one — mocked HTTP
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ping_one_success(checker: EndpointHealthChecker) -> None:
    async def _mock_fetch(url: str) -> tuple[int, str | None]:
        return 200, None

    checker._fetch = _mock_fetch  # type: ignore[method-assign]
    result = await checker.ping_one("alpha", "https://alpha.example.com/v1/models")
    assert result.available is True
    assert result.status_code == 200
    assert result.error is None
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_ping_one_401_still_available(checker: EndpointHealthChecker) -> None:
    async def _mock_fetch(url: str) -> tuple[int, str | None]:
        return 401, None

    checker._fetch = _mock_fetch  # type: ignore[method-assign]
    result = await checker.ping_one("alpha", "https://alpha.example.com/v1/models")
    assert result.available is True
    assert result.status_code == 401


@pytest.mark.asyncio
async def test_ping_one_connection_failure(checker: EndpointHealthChecker) -> None:
    async def _mock_fetch(url: str) -> tuple[int, str | None]:
        return -1, "Connection refused"

    checker._fetch = _mock_fetch  # type: ignore[method-assign]
    result = await checker.ping_one("alpha", "https://alpha.example.com/v1/models")
    assert result.available is False
    assert result.status_code == -1
    assert result.error == "Connection refused"


@pytest.mark.asyncio
async def test_ping_one_exception_wrapped(checker: EndpointHealthChecker) -> None:
    async def _mock_fetch(url: str) -> tuple[int, str | None]:
        raise RuntimeError("network error")

    checker._fetch = _mock_fetch  # type: ignore[method-assign]
    result = await checker.ping_one("alpha", "https://alpha.example.com/v1/models")
    assert result.available is False
    assert "network error" in result.error


# ---------------------------------------------------------------------------
# ping_all — mocked, concurrency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ping_all_populates_history(checker: EndpointHealthChecker) -> None:
    async def _mock_fetch(url: str) -> tuple[int, str | None]:
        return 200, None

    checker._fetch = _mock_fetch  # type: ignore[method-assign]
    results = await checker.ping_all()
    assert set(results.keys()) == {"alpha", "beta"}
    assert len(checker._history["alpha"]) == 1
    assert len(checker._history["beta"]) == 1


@pytest.mark.asyncio
async def test_ping_all_respects_window(checker: EndpointHealthChecker) -> None:
    async def _mock_fetch(url: str) -> tuple[int, str | None]:
        return 200, None

    checker._fetch = _mock_fetch  # type: ignore[method-assign]
    for _ in range(10):
        await checker.ping_all()
    # window_size is 5
    assert len(checker._history["alpha"]) == 5
    assert len(checker._history["beta"]) == 5


# ---------------------------------------------------------------------------
# get_metrics / get_all_metrics after pings
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_metrics_after_pings(checker: EndpointHealthChecker) -> None:
    async def _mock_fetch(url: str) -> tuple[int, str | None]:
        return 200, None

    checker._fetch = _mock_fetch  # type: ignore[method-assign]
    await checker.ping_all()
    m = checker.get_metrics("alpha")
    assert m is not None
    assert isinstance(m, StabilityMetrics)
    assert m.ping_count == 1
    assert m.availability_rate == 1.0
    assert m.composite_score > 0


@pytest.mark.asyncio
async def test_get_all_metrics_sorted_best_first(checker: EndpointHealthChecker) -> None:
    call_count = {"n": 0}

    async def _mock_fetch(url: str) -> tuple[int, str | None]:
        call_count["n"] += 1
        # beta always fails, alpha always succeeds
        if "beta" in url:
            return -1, "refused"
        return 200, None

    checker._fetch = _mock_fetch  # type: ignore[method-assign]
    for _ in range(3):
        await checker.ping_all()

    all_m = checker.get_all_metrics()
    names = list(all_m.keys())
    assert names[0] == "alpha"  # higher composite score
    scores = [v.composite_score for v in all_m.values()]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_availability_rate_partial_failures(checker: EndpointHealthChecker) -> None:
    responses = [True, True, False, True, False]  # 3/5 success

    async def _mock_fetch(url: str) -> tuple[int, str | None]:
        ok = responses.pop(0)
        return (200, None) if ok else (-1, "err")

    checker._fetch = _mock_fetch  # type: ignore[method-assign]
    # only alpha endpoint — remove beta first
    checker.remove_endpoint("beta")
    for _ in range(5):
        await checker.ping_all()

    m = checker.get_metrics("alpha")
    assert m is not None
    assert m.availability_rate == pytest.approx(3 / 5)


# ---------------------------------------------------------------------------
# Background polling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_stop_background_polling(checker: EndpointHealthChecker) -> None:
    ping_calls: list[str] = []

    async def _mock_ping_all() -> dict:
        ping_calls.append("called")
        return {}

    checker.ping_all = _mock_ping_all  # type: ignore[method-assign]
    # fast_start_rounds=0 skips the 3-second _POLL_FAST sleep so we go
    # straight to the steady-state loop with a 0.05 s interval.
    await checker.start_background_polling(interval_seconds=0.05, fast_start_rounds=0)
    assert checker._poll_task is not None
    assert not checker._poll_task.done()

    await asyncio.sleep(0.25)
    checker.stop_background_polling()
    assert checker._poll_task is None
    assert len(ping_calls) >= 2  # at least a couple of 0.05 s rounds


@pytest.mark.asyncio
async def test_start_background_polling_idempotent(checker: EndpointHealthChecker) -> None:
    async def _mock_ping_all() -> dict:
        return {}

    checker.ping_all = _mock_ping_all  # type: ignore[method-assign]
    await checker.start_background_polling(interval_seconds=60, fast_start_rounds=0)
    task1 = checker._poll_task
    await checker.start_background_polling(interval_seconds=60, fast_start_rounds=0)
    task2 = checker._poll_task
    assert task1 is task2  # second call is a no-op
    checker.stop_background_polling()


@pytest.mark.asyncio
async def test_stop_when_not_running_is_safe(checker: EndpointHealthChecker) -> None:
    checker.stop_background_polling()  # should not raise


# ---------------------------------------------------------------------------
# _fetch_sync fallback (sync path via urllib)
# ---------------------------------------------------------------------------


def test_fetch_sync_200(checker: EndpointHealthChecker) -> None:
    import urllib.request

    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: mock_resp
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.status = 200

    with patch("urllib.request.urlopen", return_value=mock_resp):
        code, err = checker._fetch_sync("https://example.com")
    assert code == 200
    assert err is None


def test_fetch_sync_401(checker: EndpointHealthChecker) -> None:
    import urllib.error

    http_err = urllib.error.HTTPError(
        url="https://example.com", code=401, msg="Unauthorized", hdrs=None, fp=None
    )
    with patch("urllib.request.urlopen", side_effect=http_err):
        code, err = checker._fetch_sync("https://example.com")
    assert code == 401
    assert err is None  # 4xx means alive, no error string


def test_fetch_sync_connection_error(checker: EndpointHealthChecker) -> None:
    with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
        code, err = checker._fetch_sync("https://example.com")
    assert code == -1
    assert err is not None
