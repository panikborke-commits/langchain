"""Claude (Anthropic) partner package for LangChain."""

from langchain_anthropic._version import __version__
from langchain_anthropic.chat_models import (
    ChatAnthropic,
    convert_to_anthropic_tool,
)
from langchain_anthropic.llms import AnthropicLLM
from langchain_anthropic.optimizations import (
    TokenUsageStats,
    apply_auto_cache_to_system,
    apply_auto_cache_to_tools,
)

__all__ = [
    "AnthropicLLM",
    "ChatAnthropic",
    "TokenUsageStats",
    "__version__",
    "apply_auto_cache_to_system",
    "apply_auto_cache_to_tools",
    "convert_to_anthropic_tool",
]
