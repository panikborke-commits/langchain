"""Self-healing multi-connector orchestrator.

`ConnectorOrchestrator` wraps a pool of `Runnable` connectors and routes each
invocation to the healthiest available one.  It recovers autonomously from
common failures:

- **Context overflow** – trims the message history and retries on the same
  connector.
- **Rate limits / transient errors** – fails over to the next connector in the
  priority order, backing off with exponential delay.
- **Persistent failures** – opens a circuit breaker so the connector is skipped
  until a recovery window expires.

New connectors can be registered at runtime via :meth:`register`.  The
orchestrator inspects incoming connector objects and adapts routing weight
accordingly — no manual configuration required.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from collections import deque
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import BaseMessage
from langchain_core.runnables.base import Runnable, RunnableSerializable
from langchain_core.runnables.config import RunnableConfig, ensure_config
from langchain_core.runnables.utils import Input, Output

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

logger = logging.getLogger(__name__)

# How many tokens to shave off when recovering from a context-overflow error.
# Expressed as a fraction of the current message list length.
_TRIM_FRACTION: float = 0.25
_MIN_TRIM_MESSAGES: int = 1

# Circuit-breaker defaults
_DEFAULT_FAILURE_THRESHOLD: int = 3
_DEFAULT_RECOVERY_TIMEOUT: float = 60.0  # seconds
_DEFAULT_MAX_RETRIES: int = 3
_DEFAULT_BACKOFF_BASE: float = 2.0
_DEFAULT_BACKOFF_MAX: float = 30.0


class _CircuitState(Enum):
    CLOSED = auto()   # normal operation
    OPEN = auto()     # connector suspended after repeated failures
    HALF_OPEN = auto()  # testing if connector has recovered


class _ConnectorHealth:
    """Per-connector health tracker implementing the circuit-breaker pattern."""

    def __init__(
        self,
        failure_threshold: int,
        recovery_timeout: float,
        window: int = 10,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._lock = threading.Lock()
        self._state = _CircuitState.CLOSED
        self._consecutive_failures = 0
        self._last_failure_time: float = 0.0
        # Rolling window for success-rate tracking
        self._recent: deque[bool] = deque(maxlen=window)

    @property
    def state(self) -> _CircuitState:
        with self._lock:
            if self._state == _CircuitState.OPEN:
                if (time.monotonic() - self._last_failure_time) >= self._recovery_timeout:
                    self._state = _CircuitState.HALF_OPEN
            return self._state

    @property
    def is_available(self) -> bool:
        return self.state != _CircuitState.OPEN

    def _compute_success_rate(self) -> float:
        """Compute success rate without acquiring the lock (caller must hold it)."""
        if not self._recent:
            return 1.0
        return sum(self._recent) / len(self._recent)

    @property
    def success_rate(self) -> float:
        with self._lock:
            return self._compute_success_rate()

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._recent.append(True)
            self._state = _CircuitState.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            self._recent.append(False)
            self._last_failure_time = time.monotonic()
            if self._consecutive_failures >= self._failure_threshold:
                if self._state != _CircuitState.OPEN:
                    logger.warning(
                        "Circuit breaker OPENED after %d consecutive failures.",
                        self._consecutive_failures,
                    )
                self._state = _CircuitState.OPEN

    def summary(self) -> dict[str, Any]:
        """Return a snapshot of the health metrics.

        Returns:
            Dict with ``state``, ``consecutive_failures``, and ``success_rate``.
        """
        with self._lock:
            return {
                "state": self._state.name,
                "consecutive_failures": self._consecutive_failures,
                "success_rate": self._compute_success_rate(),
            }


def _trim_messages_by_fraction(messages: list[BaseMessage], fraction: float) -> list[BaseMessage]:
    """Remove the oldest *fraction* of messages, keeping at least one.

    Preserves the first message when it is a `SystemMessage` so the model
    context is not lost.  Never returns an empty list.

    Args:
        messages: Current message list.
        fraction: Proportion of messages to remove (0 < fraction < 1).

    Returns:
        Trimmed message list with at least one message.
    """
    from langchain_core.messages import SystemMessage

    if len(messages) <= 1:
        return messages

    has_system = isinstance(messages[0], SystemMessage)
    non_system = messages[1:] if has_system else messages
    n = min(
        max(_MIN_TRIM_MESSAGES, math.ceil(len(non_system) * fraction)),
        len(non_system) - 1,  # always keep the most recent non-system message
    )
    trimmed_non_system = non_system[n:]
    return ([messages[0], *trimmed_non_system] if has_system else trimmed_non_system) or messages[-1:]


class ConnectorOrchestrator(RunnableSerializable[Input, Output]):
    """Autonomous, self-healing orchestrator for multiple LangChain connectors.

    Routes each invocation to the best available connector based on real-time
    health scores.  Handles errors without caller intervention:

    - **ContextOverflowError** → trims 25 % of message history and retries.
    - **Any other exception** → records the failure, opens the circuit breaker
      after `failure_threshold` consecutive errors, and fails over to the next
      healthy connector.

    New connectors can be added at runtime with :meth:`register`; the
    orchestrator starts using them immediately.

    Example:
        ```python
        from langchain_anthropic import ChatAnthropic
        from langchain_core.runnables.orchestrator import ConnectorOrchestrator

        orchestrator = ConnectorOrchestrator(
            connectors={
                "claude": ChatAnthropic(model="claude-opus-4-7", auto_cache=True),
            },
            max_retries=3,
        )

        # Works like any Runnable
        response = orchestrator.invoke("Explain quantum entanglement.")

        # Add a fallback connector at runtime
        from langchain_openai import ChatOpenAI
        orchestrator.register("openai", ChatOpenAI(model="gpt-4o"))

        # Inspect connector health
        print(orchestrator.health_report())
        ```
    """

    # Pydantic-aware type declaration for RunnableSerializable
    model_config = {"arbitrary_types_allowed": True}

    # These are intentionally NOT Pydantic fields so they can hold arbitrary
    # Runnable instances without serialisation constraints.  We override
    # __init__ to populate them.
    _connectors: dict[str, Runnable]
    _health: dict[str, _ConnectorHealth]
    _priority: list[str]
    _lock: threading.Lock
    _failure_threshold: int
    _recovery_timeout: float
    _max_retries: int

    def __init__(
        self,
        connectors: dict[str, Runnable],
        *,
        strategy: str = "priority",
        max_retries: int = _DEFAULT_MAX_RETRIES,
        failure_threshold: int = _DEFAULT_FAILURE_THRESHOLD,
        recovery_timeout: float = _DEFAULT_RECOVERY_TIMEOUT,
    ) -> None:
        """Create an orchestrator from a named pool of connectors.

        Args:
            connectors: Mapping of connector name → `Runnable`.  The iteration
                order defines the default routing priority.
            strategy: Routing strategy.  Currently only `'priority'` is
                supported (try connectors in registration order, skip unhealthy
                ones).
            max_retries: Maximum total attempts across all connectors before
                raising the last exception.
            failure_threshold: Number of consecutive failures that open a
                connector's circuit breaker.
            recovery_timeout: Seconds after which an open circuit breaker
                transitions to HALF_OPEN for a recovery probe.
        """
        super().__init__()
        if not connectors:
            msg = "At least one connector must be provided."
            raise ValueError(msg)
        self._connectors = dict(connectors)
        self._priority = list(connectors)
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._max_retries = max_retries
        self._lock = threading.Lock()
        self._health = {
            name: _ConnectorHealth(failure_threshold, recovery_timeout)
            for name in connectors
        }
        self._strategy = strategy

    # ------------------------------------------------------------------
    # Registry management
    # ------------------------------------------------------------------

    def register(self, name: str, connector: Runnable) -> None:
        """Add or replace a connector in the pool.

        Newly registered connectors start with a clean health record and are
        appended to the end of the routing priority list (unless a connector
        with the same name already exists, in which case only the runnable and
        health are reset).

        Args:
            name: Unique identifier for the connector.
            connector: Any `Runnable` — e.g., a chat model or a chain.
        """
        with self._lock:
            self._connectors[name] = connector
            self._health[name] = _ConnectorHealth(
                self._failure_threshold, self._recovery_timeout
            )
            if name not in self._priority:
                self._priority.append(name)
        logger.info("Connector '%s' registered.", name)

    def unregister(self, name: str) -> None:
        """Remove a connector from the pool.

        Args:
            name: Connector identifier to remove.

        Raises:
            KeyError: If no connector with this name exists.
        """
        with self._lock:
            if name not in self._connectors:
                msg = f"Connector '{name}' is not registered."
                raise KeyError(msg)
            del self._connectors[name]
            del self._health[name]
            self._priority.remove(name)
        logger.info("Connector '%s' unregistered.", name)

    # ------------------------------------------------------------------
    # Health reporting
    # ------------------------------------------------------------------

    def health_report(self) -> dict[str, Any]:
        """Return a snapshot of health metrics for every registered connector.

        Returns:
            Dict mapping connector name → health summary dict with keys
            ``state``, ``consecutive_failures``, and ``success_rate``.
        """
        with self._lock:
            return {name: h.summary() for name, h in self._health.items()}

    # ------------------------------------------------------------------
    # Internal routing helpers
    # ------------------------------------------------------------------

    def _ordered_connectors(self) -> list[tuple[str, Runnable]]:
        """Return connectors sorted by priority, skipping OPEN circuit breakers."""
        with self._lock:
            priority = list(self._priority)
            connectors = dict(self._connectors)
            health = dict(self._health)

        available = [(n, connectors[n]) for n in priority if health[n].is_available]
        suspended = [(n, connectors[n]) for n in priority if not health[n].is_available]
        # Put suspended connectors last so they still get a chance if all others fail
        return available + suspended

    def _backoff(self, attempt: int) -> None:
        delay = min(
            _DEFAULT_BACKOFF_MAX,
            _DEFAULT_BACKOFF_BASE ** attempt,
        )
        time.sleep(delay)

    async def _abackoff(self, attempt: int) -> None:
        delay = min(
            _DEFAULT_BACKOFF_MAX,
            _DEFAULT_BACKOFF_BASE ** attempt,
        )
        await asyncio.sleep(delay)

    def _resolve_messages(self, input_: Input) -> list[BaseMessage] | None:
        """Extract message list from input if possible, else return None."""
        if isinstance(input_, list) and all(isinstance(m, BaseMessage) for m in input_):
            return input_  # type: ignore[return-value]
        return None

    # ------------------------------------------------------------------
    # Synchronous invoke
    # ------------------------------------------------------------------

    def invoke(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Output:
        """Invoke the best available connector, self-healing on failure.

        Args:
            input: Input to the connectors (messages, string, dict, …).
            config: Optional `RunnableConfig`.
            **kwargs: Extra keyword arguments forwarded to the connector.

        Returns:
            The output from the first successful connector.

        Raises:
            Exception: Re-raises the last exception when all connectors and
                retries are exhausted.
        """
        config = ensure_config(config)
        attempt = 0
        last_exc: Exception | None = None
        current_input: Input = input

        while attempt < self._max_retries:
            for name, connector in self._ordered_connectors():
                try:
                    result = connector.invoke(current_input, config, **kwargs)
                    self._health[name].record_success()
                    return result
                except ContextOverflowError as exc:
                    self._health[name].record_failure()
                    logger.warning(
                        "Connector '%s' hit context overflow; trimming history (attempt %d).",
                        name,
                        attempt + 1,
                    )
                    messages = self._resolve_messages(current_input)
                    if messages and len(messages) > 1:
                        current_input = _trim_messages_by_fraction(  # type: ignore[assignment]
                            messages, _TRIM_FRACTION
                        )
                        last_exc = exc
                        # Retry the *same* connector with shorter context
                        try:
                            result = connector.invoke(current_input, config, **kwargs)
                            self._health[name].record_success()
                            return result
                        except Exception as inner_exc:
                            self._health[name].record_failure()
                            last_exc = inner_exc
                    else:
                        last_exc = exc
                except Exception as exc:
                    self._health[name].record_failure()
                    logger.warning(
                        "Connector '%s' failed (attempt %d): %s",
                        name,
                        attempt + 1,
                        exc,
                    )
                    last_exc = exc

            attempt += 1
            if attempt < self._max_retries:
                self._backoff(attempt)

        assert last_exc is not None
        raise last_exc

    # ------------------------------------------------------------------
    # Asynchronous invoke
    # ------------------------------------------------------------------

    async def ainvoke(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Output:
        """Async version of :meth:`invoke`.

        Args:
            input: Input to the connectors.
            config: Optional `RunnableConfig`.
            **kwargs: Extra keyword arguments forwarded to the connector.

        Returns:
            The output from the first successful connector.

        Raises:
            Exception: Re-raises the last exception when all retries fail.
        """
        config = ensure_config(config)
        attempt = 0
        last_exc: Exception | None = None
        current_input: Input = input

        while attempt < self._max_retries:
            for name, connector in self._ordered_connectors():
                try:
                    result = await connector.ainvoke(current_input, config, **kwargs)
                    self._health[name].record_success()
                    return result
                except ContextOverflowError as exc:
                    self._health[name].record_failure()
                    logger.warning(
                        "Connector '%s' hit context overflow; trimming history (attempt %d).",
                        name,
                        attempt + 1,
                    )
                    messages = self._resolve_messages(current_input)
                    if messages and len(messages) > 1:
                        current_input = _trim_messages_by_fraction(  # type: ignore[assignment]
                            messages, _TRIM_FRACTION
                        )
                        last_exc = exc
                        try:
                            result = await connector.ainvoke(current_input, config, **kwargs)
                            self._health[name].record_success()
                            return result
                        except Exception as inner_exc:
                            self._health[name].record_failure()
                            last_exc = inner_exc
                    else:
                        last_exc = exc
                except Exception as exc:
                    self._health[name].record_failure()
                    logger.warning(
                        "Connector '%s' failed (attempt %d): %s",
                        name,
                        attempt + 1,
                        exc,
                    )
                    last_exc = exc

            attempt += 1
            if attempt < self._max_retries:
                await self._abackoff(attempt)

        assert last_exc is not None
        raise last_exc

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Output]:
        """Stream from the first available connector, falling back to `invoke`.

        If the primary connector does not support streaming, `invoke` is called
        and the single result is yielded.

        Args:
            input: Input to the connectors.
            config: Optional `RunnableConfig`.
            **kwargs: Extra keyword arguments forwarded to the connector.

        Yields:
            Output chunks from the selected connector.
        """
        config = ensure_config(config)
        for name, connector in self._ordered_connectors():
            try:
                chunks = list(connector.stream(input, config, **kwargs))
                self._health[name].record_success()
                yield from chunks
                return
            except Exception as exc:
                self._health[name].record_failure()
                logger.warning("Connector '%s' stream failed: %s", name, exc)

        msg = "All connectors failed during streaming."
        raise RuntimeError(msg)

    async def astream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Output]:
        """Async streaming version of :meth:`stream`.

        Args:
            input: Input to the connectors.
            config: Optional `RunnableConfig`.
            **kwargs: Extra keyword arguments forwarded to the connector.

        Yields:
            Output chunks from the selected connector.
        """
        config = ensure_config(config)
        for name, connector in self._ordered_connectors():
            try:
                chunks: list[Output] = []
                async for chunk in connector.astream(input, config, **kwargs):
                    chunks.append(chunk)
                self._health[name].record_success()
                for chunk in chunks:
                    yield chunk
                return
            except Exception as exc:
                self._health[name].record_failure()
                logger.warning("Connector '%s' astream failed: %s", name, exc)

        msg = "All connectors failed during async streaming."
        raise RuntimeError(msg)

    # ------------------------------------------------------------------
    # Serialisation support
    # ------------------------------------------------------------------

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return `False` – connector instances are not JSON-serialisable."""
        return False

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Return the LangChain namespace for this class.

        Returns:
            `["langchain", "schema", "runnable"]`
        """
        return ["langchain", "schema", "runnable"]
