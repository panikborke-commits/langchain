"""Async endpoint health checker with composite stability scoring.

Designed for monitoring LLM API endpoints in parallel without consuming API
quota: a lightweight GET to `/v1/models` (or any probe URL) counts as
"available" whether the server returns 200 or 401 — both prove it is
reachable.  Only connection failures and timeouts count as unavailable.

Typical usage::

    import asyncio
    from langchain_core.runnables.health_check import EndpointHealthChecker

    checker = EndpointHealthChecker(
        endpoints={
            "groq":       "https://api.groq.com/openai/v1/models",
            "openrouter": "https://openrouter.ai/api/v1/models",
        }
    )

    # One-off snapshot
    await checker.ping_all()
    metrics = checker.get_all_metrics()
    print(metrics["groq"].composite_score)

    # Continuous background polling
    await checker.start_background_polling(interval_seconds=15)
    # ... later ...
    checker.stop_background_polling()
"""

from __future__ import annotations

import asyncio
import logging
import math
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Above this latency the speed score saturates to 0.
_LATENCY_CEILING_MS: float = 8_000.0

# Polling intervals in seconds for the adaptive poller.
_POLL_FAST: float = 3.0    # just started / manual refresh
_POLL_NORMAL: float = 12.0  # steady state
_POLL_IDLE: float = 30.0   # no recent activity

# Composite score weights — must sum to 1.0.
_W_AVAILABILITY: float = 0.40
_W_SPEED: float = 0.35
_W_CONSISTENCY: float = 0.25


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class PingResult:
    """Outcome of a single health probe against one endpoint.

    Attributes:
        endpoint: The URL that was probed.
        latency_ms: Round-trip time in milliseconds.
        status_code: HTTP status code, or -1 on connection failure.
        available: `True` when the server responded (any HTTP status counts;
            even a 401 proves the server is alive).
        timestamp: ``time.monotonic()`` value at the moment of the probe.
        error: Exception message if the probe failed, else `None`.
    """

    endpoint: str
    latency_ms: float
    status_code: int
    available: bool
    timestamp: float
    error: str | None = None


@dataclass
class StabilityMetrics:
    """Aggregated health statistics over a rolling window.

    Attributes:
        name: Human-readable connector or endpoint name.
        endpoint: The probed URL.
        ping_count: Number of probes in the window.
        availability_rate: Fraction of probes that got a response (0–1).
        p50_ms: Median latency in milliseconds.
        p95_ms: 95th-percentile latency in milliseconds.
        jitter_ms: Standard deviation of observed latencies.
        composite_score: Weighted score in [0, 1] — higher is better.
            Computed as ``0.40 × availability + 0.35 × speed + 0.25 × consistency``.
    """

    name: str
    endpoint: str
    ping_count: int
    availability_rate: float
    p50_ms: float
    p95_ms: float
    jitter_ms: float
    composite_score: float


def _compute_composite(
    availability: float,
    p95_ms: float,
    jitter_ms: float,
    p50_ms: float,
) -> float:
    """Return a 0–1 composite stability score.

    Higher is better.  The three components are:

    - **Availability** (40 %): fraction of pings that got any HTTP response.
    - **Speed** (35 %): 1 minus the p95 latency normalised against an 8-second
      ceiling.  Anything above 8 s scores 0.
    - **Consistency** (25 %): 1 minus the jitter-to-p50 ratio, clamped to [0, 1].
      A perfectly stable endpoint with no jitter scores 1.

    Args:
        availability: Fraction of successful probes, 0–1.
        p95_ms: 95th-percentile latency in milliseconds.
        jitter_ms: Standard deviation of latency observations.
        p50_ms: Median latency in milliseconds.

    Returns:
        Composite score in [0, 1].
    """
    speed = 1.0 - min(p95_ms / _LATENCY_CEILING_MS, 1.0)
    consistency = 1.0 - min(jitter_ms / p50_ms, 1.0) if p50_ms > 0 else 0.0
    return (
        _W_AVAILABILITY * availability
        + _W_SPEED * speed
        + _W_CONSISTENCY * consistency
    )


# ---------------------------------------------------------------------------
# Core checker
# ---------------------------------------------------------------------------


class EndpointHealthChecker:
    """Parallel async health checker for multiple LLM API endpoints.

    Sends a lightweight GET request to a probe URL (e.g. ``/v1/models``) and
    records whether the server responded, how fast it responded, and how
    consistently it performs over a rolling window.

    The checker never consumes API quota: even a 401 "Unauthorized" response
    proves the server is up and was measured correctly.

    Args:
        endpoints: Mapping of endpoint name → probe URL.
        window_size: Number of recent pings to keep per endpoint.  Older
            results are discarded.  Defaults to 20.
        connect_timeout: Seconds to wait for a TCP connection.
        read_timeout: Seconds to wait for the first response byte.
        headers: Optional extra headers sent with every probe (e.g. to satisfy
            mandatory ``Authorization`` headers).
    """

    def __init__(
        self,
        endpoints: dict[str, str],
        *,
        window_size: int = 20,
        connect_timeout: float = 5.0,
        read_timeout: float = 10.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._endpoints: dict[str, str] = dict(endpoints)
        self._window_size = window_size
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._headers: dict[str, str] = headers or {}
        # Rolling window of PingResults per named endpoint.
        self._history: dict[str, deque[PingResult]] = {
            name: deque(maxlen=window_size) for name in endpoints
        }
        self._poll_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Registry management
    # ------------------------------------------------------------------

    def add_endpoint(self, name: str, url: str) -> None:
        """Register a new endpoint for health monitoring.

        Args:
            name: Unique name for this endpoint.
            url: Probe URL (e.g. ``https://api.example.com/v1/models``).
        """
        self._endpoints[name] = url
        if name not in self._history:
            self._history[name] = deque(maxlen=self._window_size)

    def remove_endpoint(self, name: str) -> None:
        """Stop monitoring an endpoint.

        Args:
            name: Name registered via :meth:`add_endpoint`.

        Raises:
            KeyError: If the name is not registered.
        """
        if name not in self._endpoints:
            msg = f"Endpoint '{name}' is not registered."
            raise KeyError(msg)
        del self._endpoints[name]
        del self._history[name]

    # ------------------------------------------------------------------
    # Core probing
    # ------------------------------------------------------------------

    async def ping_one(self, name: str, url: str) -> PingResult:
        """Send a single lightweight probe to *url* and return the result.

        Uses ``httpx.AsyncClient`` when available, falls back to a
        thread-pool ``urllib`` call so the method works even without httpx
        installed (latency data is still correct in that case).

        Args:
            name: Name of the endpoint (used only for logging).
            url: The URL to probe.

        Returns:
            A :class:`PingResult` describing the outcome.
        """
        t0 = time.monotonic()
        try:
            status_code, error = await self._fetch(url)
        except Exception as exc:  # noqa: BLE001
            elapsed = (time.monotonic() - t0) * 1000
            return PingResult(
                endpoint=url,
                latency_ms=elapsed,
                status_code=-1,
                available=False,
                timestamp=time.monotonic(),
                error=str(exc),
            )

        elapsed = (time.monotonic() - t0) * 1000
        # Any HTTP response (including 4xx/5xx) counts as "available": the
        # server answered.  Only connection failures are "unavailable".
        available = error is None
        return PingResult(
            endpoint=url,
            latency_ms=elapsed,
            status_code=status_code,
            available=available,
            timestamp=time.monotonic(),
            error=error,
        )

    async def _fetch(self, url: str) -> tuple[int, str | None]:
        """Perform the HTTP GET and return (status_code, error_or_None)."""
        try:
            import httpx

            timeout = httpx.Timeout(
                connect=self._connect_timeout, read=self._read_timeout, write=5.0, pool=5.0
            )
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url, headers=self._headers)
                return response.status_code, None
        except ImportError:
            # httpx not installed — fall back to thread pool + urllib
            return await asyncio.get_event_loop().run_in_executor(
                None, self._fetch_sync, url
            )
        except Exception as exc:  # noqa: BLE001
            return -1, str(exc)

    def _fetch_sync(self, url: str) -> tuple[int, str | None]:
        """Synchronous fallback using urllib (no external dependencies)."""
        import urllib.request
        try:
            req = urllib.request.Request(url, headers=self._headers, method="GET")
            with urllib.request.urlopen(req, timeout=self._read_timeout) as resp:
                return resp.status, None
        except urllib.error.HTTPError as exc:
            return exc.code, None  # 4xx/5xx still means "server is alive"
        except Exception as exc:  # noqa: BLE001
            return -1, str(exc)

    async def ping_all(self) -> dict[str, PingResult]:
        """Probe all registered endpoints concurrently.

        Results are stored in the internal rolling window and also returned
        directly for immediate use.

        Returns:
            Dict mapping endpoint name → :class:`PingResult`.
        """
        tasks = {
            name: asyncio.create_task(self.ping_one(name, url))
            for name, url in self._endpoints.items()
        }
        results: dict[str, PingResult] = {}
        for name, task in tasks.items():
            result = await task
            results[name] = result
            if name in self._history:
                self._history[name].append(result)
            else:
                self._history[name] = deque([result], maxlen=self._window_size)
        return results

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self, name: str) -> StabilityMetrics | None:
        """Return aggregated stability metrics for one endpoint.

        Args:
            name: Registered endpoint name.

        Returns:
            :class:`StabilityMetrics` or `None` if no pings recorded yet.
        """
        history = self._history.get(name)
        if not history:
            return None
        return self._compute_metrics(name, self._endpoints.get(name, ""), list(history))

    def get_all_metrics(self) -> dict[str, StabilityMetrics]:
        """Return metrics for every endpoint, sorted by composite score (best first).

        Returns:
            Ordered dict mapping name → :class:`StabilityMetrics`.
        """
        result: dict[str, StabilityMetrics] = {}
        for name, history in self._history.items():
            if history:
                url = self._endpoints.get(name, "")
                result[name] = self._compute_metrics(name, url, list(history))
        return dict(
            sorted(result.items(), key=lambda kv: kv[1].composite_score, reverse=True)
        )

    @staticmethod
    def _compute_metrics(
        name: str, url: str, pings: list[PingResult]
    ) -> StabilityMetrics:
        latencies = [p.latency_ms for p in pings if p.available]
        availability = sum(1 for p in pings if p.available) / len(pings)

        if latencies:
            sorted_lat = sorted(latencies)
            p50 = statistics.median(sorted_lat)
            idx_p95 = max(0, math.ceil(len(sorted_lat) * 0.95) - 1)
            p95 = sorted_lat[idx_p95]
            jitter = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
        else:
            p50 = p95 = jitter = 0.0

        return StabilityMetrics(
            name=name,
            endpoint=url,
            ping_count=len(pings),
            availability_rate=availability,
            p50_ms=p50,
            p95_ms=p95,
            jitter_ms=jitter,
            composite_score=_compute_composite(availability, p95, jitter, p50),
        )

    # ------------------------------------------------------------------
    # Background polling
    # ------------------------------------------------------------------

    async def start_background_polling(
        self,
        interval_seconds: float = _POLL_NORMAL,
        *,
        fast_start_rounds: int = 3,
    ) -> None:
        """Start continuous background health polling as an asyncio task.

        The first `fast_start_rounds` polls run at :data:`_POLL_FAST` interval
        to populate metrics quickly, then switch to `interval_seconds`.

        The task runs until :meth:`stop_background_polling` is called or the
        event loop closes.

        Args:
            interval_seconds: Steady-state seconds between polls.
            fast_start_rounds: How many rapid initial polls to run.
        """
        if self._poll_task and not self._poll_task.done():
            return  # already running

        async def _loop() -> None:
            for _ in range(fast_start_rounds):
                await self.ping_all()
                await asyncio.sleep(_POLL_FAST)
            while True:
                await self.ping_all()
                await asyncio.sleep(interval_seconds)

        self._poll_task = asyncio.create_task(_loop())
        logger.info(
            "Background health polling started (interval=%.0fs, %d endpoints).",
            interval_seconds,
            len(self._endpoints),
        )

    def stop_background_polling(self) -> None:
        """Cancel the background polling task if it is running."""
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            self._poll_task = None
            logger.info("Background health polling stopped.")
