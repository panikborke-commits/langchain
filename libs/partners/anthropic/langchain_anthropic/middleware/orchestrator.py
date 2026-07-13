"""Agent middleware that routes model calls through a `ConnectorOrchestrator`.

This bridges the LangChain agent middleware framework with the autonomous
`ConnectorOrchestrator`, enabling health-aware connector selection, automatic
failover, and self-healing inside any agent built with `create_agent()`.

Requires both ``langchain`` (for the agent framework) and
``langchain-anthropic`` to be installed.

Example:
    ```python
    from langchain_anthropic import ChatAnthropic
    from langchain_anthropic.middleware.orchestrator import OrchestratorMiddleware
    from langchain_core.runnables.orchestrator import ConnectorOrchestrator

    orch = ConnectorOrchestrator(
        connectors={
            "claude-big":  ChatAnthropic(model="claude-opus-4-7",   auto_cache=True),
            "claude-fast": ChatAnthropic(model="claude-haiku-4-5",  auto_cache=True),
        }
    )

    agent = create_agent(
        model=ChatAnthropic(model="claude-opus-4-7"),
        tools=[...],
        middleware=[OrchestratorMiddleware(orch)],
    )
    ```
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

try:
    from langchain.agents.middleware.types import (
        AgentMiddleware,
        ModelCallResult,
        ModelRequest,
        ModelResponse,
    )
except ImportError as e:
    msg = (
        "OrchestratorMiddleware requires 'langchain' to be installed. "
        "Install it with: pip install langchain"
    )
    raise ImportError(msg) from e

from langchain_core.messages import AIMessage
from langchain_core.runnables.orchestrator import ConnectorOrchestrator


class OrchestratorMiddleware(AgentMiddleware):
    """Route agent model calls through a `ConnectorOrchestrator`.

    Instead of calling the model bound to the agent directly, every model
    invocation is delegated to the orchestrator.  The orchestrator selects
    the healthiest available connector, falls over automatically on errors,
    and self-heals context-overflow situations by trimming the history.

    The middleware is transparent to the rest of the agent: it intercepts
    the call via `wrap_model_call`, delegates to the orchestrator, and
    returns a `ModelResponse` in the format the agent framework expects.
    """

    def __init__(self, orchestrator: ConnectorOrchestrator) -> None:
        """Attach an orchestrator to the middleware.

        Args:
            orchestrator: A configured `ConnectorOrchestrator` instance.
        """
        self._orchestrator = orchestrator

    # ------------------------------------------------------------------
    # Synchronous path
    # ------------------------------------------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Intercept the model call and delegate to the orchestrator.

        Builds the message list the orchestrator expects, invokes it, and
        wraps the returned `AIMessage` back into a `ModelResponse`.

        Args:
            request: The model request from the agent framework.
            handler: The original handler (not called; orchestrator takes over).

        Returns:
            A `ModelResponse` wrapping the orchestrator's output.
        """
        messages = _build_messages(request)
        kwargs = dict(request.model_settings or {})

        result = self._orchestrator.invoke(messages, **kwargs)

        ai_msg = result if isinstance(result, AIMessage) else AIMessage(content=str(result))
        return ModelResponse(result=[ai_msg])

    # ------------------------------------------------------------------
    # Asynchronous path
    # ------------------------------------------------------------------

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Async version of :meth:`wrap_model_call`.

        Args:
            request: The model request from the agent framework.
            handler: The original async handler (not called).

        Returns:
            A `ModelResponse` wrapping the orchestrator's output.
        """
        messages = _build_messages(request)
        kwargs = dict(request.model_settings or {})

        result = await self._orchestrator.ainvoke(messages, **kwargs)

        ai_msg = result if isinstance(result, AIMessage) else AIMessage(content=str(result))
        return ModelResponse(result=[ai_msg])


def _build_messages(request: ModelRequest) -> list[Any]:
    """Assemble the full message list from a `ModelRequest`.

    Prepends the system message (when present) so the orchestrator receives a
    complete conversation context in the standard LangChain format.

    Args:
        request: Agent framework model request.

    Returns:
        List of `BaseMessage` objects ready for a chat model.
    """
    messages: list[Any] = []
    if request.system_message is not None:
        messages.append(request.system_message)
    messages.extend(request.messages)
    return messages
