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

New connectors can be registered at runtime via :meth:`ConnectorOrchestrator.register`.
The orchestrator starts routing to them immediately — no restart needed.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from collections import deque
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import BaseMessage
from langchain_core.runnables.base import Runnable, RunnableSerializable
from langchain_core.runnables.config import RunnableConfig, ensure_config
from langchain_core.runnables.utils import Input, Output

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

logger = logging.getLogger(__name__)

# Fraction of message history to remove on context-overflow recovery.
_TRIM_FRACTION: float = 0.25
_MIN_TRIM_MESSAGES: int = 1

# Circuit-breaker defaults
_DEFAULT_FAILURE_THRESHOLD: int = 3
_DEFAULT_RECOVERY_TIMEOUT: float = 60.0  # seconds
_DEFAULT_MAX_RETRIES: int = 3
_DEFAULT_BACKOFF_BASE: float = 2.0
_DEFAULT_BACKOFF_MAX: float = 30.0

_SUPPORTED_STRATEGIES: frozenset[str] = frozenset({"priority"})


class _CircuitState(Enum):
    CLOSED = auto()    # normal operation
    OPEN = auto()      # connector suspended after repeated failures
    HALF_OPEN = auto() # testing if connector has recovered


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
        """Return success rate; caller must hold `self._lock`."""
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
        """Return a health snapshot.

        Returns:
            Dict with ``state``, ``consecutive_failures``, and ``success_rate``.
        """
        with self._lock:
            return {
                "state": self._state.name,
                "consecutive_failures": self._consecutive_failures,
                "success_rate": self._compute_success_rate(),
            }


def _trim_messages_by_fraction(
    messages: list[BaseMessage], fraction: float
) -> list[BaseMessage]:
    """Remove the oldest *fraction* of messages, keeping at least one.

    The first `SystemMessage` is always preserved.  The function never
    returns an empty list.

    Args:
        messages: Current message list.
        fraction: Proportion of non-system messages to remove (0 < fraction < 1).

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
        len(non_system) - 1,  # always keep the most recent message
    )
    trimmed = non_system[n:]
    return ([messages[0], *trimmed] if has_system else trimmed) or messages[-1:]


class ConnectorOrchestrator(RunnableSerializable[Input, Output]):
    """Autonomous, self-healing orchestrator for multiple LangChain connectors.

    Routes each invocation to the best available connector based on real-time
    health scores.  Recovers from errors without caller intervention:

    - **ContextOverflowError** → trims 25 % of message history, retries
      the *same* connector.
    - **Any other exception** → records the failure, opens the circuit
      breaker after `failure_threshold` consecutive errors, and fails over
      to the next healthy connector.

    New connectors can be added at runtime via :meth:`register`; the
    orchestrator starts routing to them immediately.

    Example:
        ```python
        from langchain_anthropic import ChatAnthropic
        from langchain_core.runnables.orchestrator import ConnectorOrchestrator

        orch = ConnectorOrchestrator(
            connectors={
                "claude": ChatAnthropic(model="claude-opus-4-7", auto_cache=True),
            },
            max_retries=3,
        )

        response = orch.invoke("Explain quantum entanglement.")

        # Register a fallback at runtime — active immediately
        from langchain_openai import ChatOpenAI
        orch.register("openai", ChatOpenAI(model="gpt-4o"))

        print(orch.health_report())
        ```
    """

    model_config = {"arbitrary_types_allowed": True}

    # Instance attributes are stored as plain Python attributes (not Pydantic
    # fields) because Runnable instances are not JSON-serialisable.
    _connectors: dict[str, Runnable]
    _health: dict[str, _ConnectorHealth]
    _priority: list[str]
    _lock: threading.Lock
    _failure_threshold: int
    _recovery_timeout: float
    _max_retries: int
    _strategy: str

    def __init__(
        self,
        connectors: dict[str, Runnable],
        *,
        strategy: Literal["priority"] = "priority",
        max_retries: int = _DEFAULT_MAX_RETRIES,
        failure_threshold: int = _DEFAULT_FAILURE_THRESHOLD,
        recovery_timeout: float = _DEFAULT_RECOVERY_TIMEOUT,
    ) -> None:
        """Create an orchestrator from a named pool of connectors.

        Args:
            connectors: Mapping of connector name → `Runnable`.  Iteration
                order sets the default routing priority.
            strategy: Routing strategy.  `'priority'` (the only supported
                value) tries connectors in registration order and skips
                unhealthy ones.
            max_retries: Maximum number of full passes through the connector
                pool before the last exception is re-raised.
            failure_threshold: Consecutive failures required to open a
                connector's circuit breaker.
            recovery_timeout: Seconds until an open circuit breaker transitions
                to HALF_OPEN for a recovery probe.

        Raises:
            ValueError: If `connectors` is empty or `strategy` is not
                `'priority'`.
        """
        super().__init__()
        if not connectors:
            msg = "At least one connector must be provided."
            raise ValueError(msg)
        if strategy not in _SUPPORTED_STRATEGIES:
            msg = (
                f"Unsupported strategy '{strategy}'. "
                f"Supported values: {sorted(_SUPPORTED_STRATEGIES)}"
            )
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

        The connector starts with a fresh health record.  If `name` already
        exists only the runnable and health are reset; its position in the
        priority list is preserved.

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
        """Return a snapshot of health metrics for every connector.

        Returns:
            Dict mapping connector name → health summary with keys
            ``state``, ``consecutive_failures``, and ``success_rate``.
        """
        with self._lock:
            return {name: h.summary() for name, h in self._health.items()}

    # ------------------------------------------------------------------
    # Internal routing helpers
    # ------------------------------------------------------------------

    def _ordered_connectors(self) -> list[tuple[str, Runnable]]:
        """Return (name, runnable) pairs sorted by priority.

        Healthy connectors come first; OPEN-circuit ones are placed at the
        end (last-resort fallback).  A snapshot of the registry is taken
        under the lock; the returned list operates on local copies so
        concurrent `register`/`unregister` calls cannot cause KeyErrors.
        """
        with self._lock:
            priority = list(self._priority)
            connectors = dict(self._connectors)
            health = dict(self._health)

        # Use key existence checks in case a connector was unregistered
        # between the lock release and the list comprehension.
        available = [
            (n, connectors[n])
            for n in priority
            if n in health and n in connectors and health[n].is_available
        ]
        suspended = [
            (n, connectors[n])
            for n in priority
            if n in health and n in connectors and not health[n].is_available
        ]
        return available + suspended

    def _record_success(self, name: str) -> None:
        """Record a success, ignoring stale names after unregister."""
        health = self._health.get(name)
        if health is not None:
            health.record_success()

    def _record_failure(self, name: str) -> None:
        """Record a failure, ignoring stale names after unregister."""
        health = self._health.get(name)
        if health is not None:
            health.record_failure()

    def _backoff(self, attempt: int) -> None:
        time.sleep(min(_DEFAULT_BACKOFF_MAX, _DEFAULT_BACKOFF_BASE ** attempt))

    async def _abackoff(self, attempt: int) -> None:
        await asyncio.sleep(
            min(_DEFAULT_BACKOFF_MAX, _DEFAULT_BACKOFF_BASE ** attempt)
        )

    def _resolve_messages(self, input_: Input) -> list[BaseMessage] | None:
        """Extract a message list from *input_* when possible."""
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
            Exception: The last exception raised when all connectors and
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
                    self._record_success(name)
                    return result
                except ContextOverflowError as exc:
                    self._record_failure(name)
                    logger.warning(
                        "Connector '%s' context overflow; trimming (attempt %d).",
                        name, attempt + 1,
                    )
                    messages = self._resolve_messages(current_input)
                    if messages and len(messages) > 1:
                        current_input = _trim_messages_by_fraction(  # type: ignore[assignment]
                            messages, _TRIM_FRACTION
                        )
                        try:
                            result = connector.invoke(current_input, config, **kwargs)
                            self._record_success(name)
                            return result
                        except Exception as inner_exc:
                            self._record_failure(name)
                            last_exc = inner_exc
                    else:
                        last_exc = exc
                except Exception as exc:
                    self._record_failure(name)
                    logger.warning(
                        "Connector '%s' failed (attempt %d): %s",
                        name, attempt + 1, exc,
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
            Exception: The last exception when all retries are exhausted.
        """
        config = ensure_config(config)
        attempt = 0
        last_exc: Exception | None = None
        current_input: Input = input

        while attempt < self._max_retries:
            for name, connector in self._ordered_connectors():
                try:
                    result = await connector.ainvoke(current_input, config, **kwargs)
                    self._record_success(name)
                    return result
                except ContextOverflowError as exc:
                    self._record_failure(name)
                    logger.warning(
                        "Connector '%s' context overflow; trimming (attempt %d).",
                        name, attempt + 1,
                    )
                    messages = self._resolve_messages(current_input)
                    if messages and len(messages) > 1:
                        current_input = _trim_messages_by_fraction(  # type: ignore[assignment]
                            messages, _TRIM_FRACTION
                        )
                        try:
                            result = await connector.ainvoke(current_input, config, **kwargs)
                            self._record_success(name)
                            return result
                        except Exception as inner_exc:
                            self._record_failure(name)
                            last_exc = inner_exc
                    else:
                        last_exc = exc
                except Exception as exc:
                    self._record_failure(name)
                    logger.warning(
                        "Connector '%s' failed (attempt %d): %s",
                        name, attempt + 1, exc,
                    )
                    last_exc = exc

            attempt += 1
            if attempt < self._max_retries:
                await self._abackoff(attempt)

        assert last_exc is not None
        raise last_exc

    # ------------------------------------------------------------------
    # Synchronous streaming
    # ------------------------------------------------------------------

    def stream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Output]:
        """Stream from the first available connector.

        Chunks are yielded as they arrive — no buffering.  Falls over to
        the next connector when the current one raises before yielding the
        first chunk.  A partial stream that fails mid-way re-raises without
        attempting failover (some output has already been sent to the caller).

        Args:
            input: Input to the connectors.
            config: Optional `RunnableConfig`.
            **kwargs: Extra keyword arguments forwarded to the connector.

        Yields:
            Output chunks from the selected connector.

        Raises:
            RuntimeError: When all connectors fail before yielding any chunk.
        """
        config = ensure_config(config)
        for name, connector in self._ordered_connectors():
            try:
                yielded = False
                for chunk in connector.stream(input, config, **kwargs):
                    yielded = True
                    yield chunk
                self._record_success(name)
                return
            except Exception as exc:
                if yielded:
                    # Partial stream already started — re-raise so the caller
                    # isn't silently given an incomplete response.
                    self._record_failure(name)
                    raise
                self._record_failure(name)
                logger.warning("Connector '%s' stream failed: %s", name, exc)

        msg = "All connectors failed during streaming."
        raise RuntimeError(msg)

    # ------------------------------------------------------------------
    # Asynchronous streaming
    # ------------------------------------------------------------------

    async def astream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Output]:
        """Async streaming version of :meth:`stream`.

        Chunks are yielded as they arrive — no buffering.  Falls over to the
        next connector when the current one raises before yielding any chunk.

        Args:
            input: Input to the connectors.
            config: Optional `RunnableConfig`.
            **kwargs: Extra keyword arguments forwarded to the connector.

        Yields:
            Output chunks from the selected connector.

        Raises:
            RuntimeError: When all connectors fail before yielding any chunk.
        """
        config = ensure_config(config)
        for name, connector in self._ordered_connectors():
            try:
                yielded = False
                async for chunk in connector.astream(input, config, **kwargs):
                    yielded = True
                    yield chunk
                self._record_success(name)
                return
            except Exception as exc:
                if yielded:
                    self._record_failure(name)
                    raise
                self._record_failure(name)
                logger.warning("Connector '%s' astream failed: %s", name, exc)

        msg = "All connectors failed during async streaming."
        raise RuntimeError(msg)

    # ------------------------------------------------------------------
    # Serialisation support
    # ------------------------------------------------------------------

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return `False` — connector instances are not JSON-serialisable."""
        return False

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Return the LangChain namespace for this class.

        Returns:
            `["langchain", "schema", "runnable"]`
        """
        return ["langchain", "schema", "runnable"]
