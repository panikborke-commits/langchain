"""Curated catalog of free and free-tier LLM coding endpoints.

All listed providers offer a free access tier (no credit card required, or a
meaningful permanently-free quota).  Tier labels map to publicly reported
SWE-bench Verified scores, which are the standard benchmark for coding
capability.

This module intentionally contains **no API keys** and makes **no network
calls**.  It is pure metadata that you use to build your own `Runnable`
instances::

    import os
    from langchain_openai import ChatOpenAI
    from langchain_core.runnables.free_models import FREE_CODING_MODELS, by_tier
    from langchain_core.runnables.orchestrator import ConnectorOrchestrator

    # Pick the three fastest S-tier models
    top = by_tier(FREE_CODING_MODELS, min_tier="S")[:3]

    connectors = {
        m.id: ChatOpenAI(
            base_url=m.api_base,
            model=m.model_id,
            api_key=os.environ.get(m.api_key_env, "none"),
        )
        for m in top
    }

    orch = ConnectorOrchestrator(connectors=connectors)

Together with :class:`~langchain_core.runnables.health_check.EndpointHealthChecker`
and :class:`HealthAwareOrchestrator` you get fully automatic routing to the
best available free endpoint at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Tier system  (based on public SWE-bench Verified leaderboard data)
# ---------------------------------------------------------------------------

CodingTierLabel = Literal["S+", "S", "A+", "A", "A-", "B+", "B", "C"]

# SWE-bench Verified minimum scores per tier label.
_TIER_MIN_SCORES: dict[CodingTierLabel, float] = {
    "S+": 0.70,
    "S":  0.60,
    "A+": 0.50,
    "A":  0.40,
    "A-": 0.30,
    "B+": 0.25,
    "B":  0.20,
    "C":  0.00,
}

_TIER_ORDER: list[CodingTierLabel] = ["S+", "S", "A+", "A", "A-", "B+", "B", "C"]


def tier_rank(label: CodingTierLabel) -> int:
    """Return the numeric rank of a tier label (lower = better).

    Args:
        label: A tier label such as ``"S+"`` or ``"A"``.

    Returns:
        Integer rank where 0 is the best tier (``S+``).
    """
    return _TIER_ORDER.index(label)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FreeModelEntry:
    """Metadata for one free or free-tier LLM endpoint.

    Attributes:
        id: Unique identifier within this catalog (e.g. ``"groq-llama-3-3-70b"``).
        provider: Human-readable provider name (e.g. ``"Groq"``).
        model_id: Exact model identifier to pass to the API.
        api_base: OpenAI-compatible base URL (without trailing slash).
        api_key_env: Name of the environment variable holding the API key.
        context_window: Maximum context tokens (input + output).
        tier: SWE-bench Verified performance tier.
        free_tier_note: Short description of the free plan limits.
        probe_path: Path appended to *api_base* for health probing.
        supports_streaming: Whether the endpoint supports streaming responses.
        supports_tools: Whether the endpoint supports tool/function calling.
        max_output_tokens: Maximum output tokens (``None`` = same as context).
        tags: Additional labels such as ``"reasoning"`` or ``"code-specialist"``.
    """

    id: str
    provider: str
    model_id: str
    api_base: str
    api_key_env: str
    context_window: int
    tier: CodingTierLabel
    free_tier_note: str
    probe_path: str = "/v1/models"
    supports_streaming: bool = True
    supports_tools: bool = True
    max_output_tokens: int | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def probe_url(self) -> str:
        """Full URL used for lightweight health probes."""
        return self.api_base.rstrip("/") + self.probe_path

    @property
    def tier_rank(self) -> int:
        """Numeric tier rank — lower is better."""
        return tier_rank(self.tier)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
# Sources: provider documentation, OpenRouter model list, SWE-bench leaderboard.
# All information is publicly available; no code from any other project was
# used to produce this list.
# ---------------------------------------------------------------------------

FREE_CODING_MODELS: list[FreeModelEntry] = [
    # ------------------------------------------------------------------
    # S+ tier  (SWE-bench Verified ≥ 70 %)
    # ------------------------------------------------------------------
    FreeModelEntry(
        id="openrouter-deepseek-r1-free",
        provider="OpenRouter",
        model_id="deepseek/deepseek-r1:free",
        api_base="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        context_window=164_000,
        tier="S+",
        free_tier_note="Free with rate limits; no credit card required",
        supports_tools=False,
        tags=("reasoning", "free"),
    ),
    FreeModelEntry(
        id="nvidia-deepseek-r1",
        provider="NVIDIA NIM",
        model_id="deepseek-ai/deepseek-r1",
        api_base="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY",
        context_window=128_000,
        tier="S+",
        free_tier_note="Free API key, generous quota via build.nvidia.com",
        probe_path="/v1/models",
        tags=("reasoning", "free"),
    ),
    # ------------------------------------------------------------------
    # S tier  (60 – 70 %)
    # ------------------------------------------------------------------
    FreeModelEntry(
        id="openrouter-deepseek-v3-free",
        provider="OpenRouter",
        model_id="deepseek/deepseek-v3:free",
        api_base="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        context_window=164_000,
        tier="S",
        free_tier_note="Free with rate limits on OpenRouter",
        tags=("coding", "free"),
    ),
    FreeModelEntry(
        id="openrouter-qwen3-235b-free",
        provider="OpenRouter",
        model_id="qwen/qwen3-235b-a22b:free",
        api_base="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        context_window=40_960,
        tier="S",
        free_tier_note="Free with rate limits on OpenRouter",
        tags=("coding", "free"),
    ),
    FreeModelEntry(
        id="nvidia-qwen3-235b",
        provider="NVIDIA NIM",
        model_id="qwen/qwen3-235b-a22b",
        api_base="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY",
        context_window=40_960,
        tier="S",
        free_tier_note="Free quota via build.nvidia.com",
        tags=("coding", "free"),
    ),
    # ------------------------------------------------------------------
    # A+ tier  (50 – 60 %)
    # ------------------------------------------------------------------
    FreeModelEntry(
        id="mistral-codestral",
        provider="Mistral La Plateforme",
        model_id="codestral-latest",
        api_base="https://api.mistral.ai/v1",
        api_key_env="MISTRAL_API_KEY",
        context_window=256_000,
        tier="A+",
        free_tier_note="Codestral free endpoint at codestral.mistral.ai",
        probe_path="/v1/models",
        tags=("code-specialist", "free"),
    ),
    FreeModelEntry(
        id="groq-llama-4-scout",
        provider="Groq",
        model_id="meta-llama/llama-4-scout-17b-16e-instruct",
        api_base="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        context_window=131_072,
        max_output_tokens=8_192,
        tier="A+",
        free_tier_note="Free tier with daily/monthly token limits",
        tags=("fast", "free"),
    ),
    FreeModelEntry(
        id="nvidia-llama-4-scout",
        provider="NVIDIA NIM",
        model_id="meta/llama-4-scout-17b-16e-instruct",
        api_base="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY",
        context_window=131_072,
        tier="A+",
        free_tier_note="Free quota via build.nvidia.com",
        tags=("free",),
    ),
    FreeModelEntry(
        id="openrouter-qwen25-coder-free",
        provider="OpenRouter",
        model_id="qwen/qwen-2.5-coder-32b-instruct:free",
        api_base="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        context_window=131_072,
        tier="A+",
        free_tier_note="Free with rate limits on OpenRouter",
        tags=("code-specialist", "free"),
    ),
    # ------------------------------------------------------------------
    # A tier  (40 – 50 %)
    # ------------------------------------------------------------------
    FreeModelEntry(
        id="groq-llama-3-3-70b",
        provider="Groq",
        model_id="llama-3.3-70b-versatile",
        api_base="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        context_window=128_000,
        max_output_tokens=32_768,
        tier="A",
        free_tier_note="Free tier with daily/monthly token limits; very low latency",
        tags=("fast", "free"),
    ),
    FreeModelEntry(
        id="cerebras-llama-3-3-70b",
        provider="Cerebras",
        model_id="llama3.3-70b",
        api_base="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        context_window=128_000,
        tier="A",
        free_tier_note="Free research tier; extremely fast inference",
        tags=("fast", "free"),
    ),
    FreeModelEntry(
        id="google-gemini-2-0-flash",
        provider="Google AI Studio",
        model_id="gemini-2.0-flash",
        api_base="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key_env="GOOGLE_API_KEY",
        context_window=1_048_576,
        max_output_tokens=8_192,
        tier="A",
        free_tier_note="Free tier at 15 rpm / 1 M tokens/day via AI Studio",
        probe_path="/v1beta/openai/models",
        tags=("long-context", "free"),
    ),
    FreeModelEntry(
        id="nvidia-llama-3-3-70b",
        provider="NVIDIA NIM",
        model_id="meta/llama-3.3-70b-instruct",
        api_base="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY",
        context_window=128_000,
        tier="A",
        free_tier_note="Free quota via build.nvidia.com",
        tags=("free",),
    ),
    # ------------------------------------------------------------------
    # A- tier  (30 – 40 %)
    # ------------------------------------------------------------------
    FreeModelEntry(
        id="groq-gemma-3-27b",
        provider="Groq",
        model_id="gemma2-9b-it",
        api_base="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        context_window=8_192,
        tier="A-",
        free_tier_note="Free tier; very fast",
        tags=("fast", "free"),
    ),
    FreeModelEntry(
        id="cloudflare-llama-3-3-70b",
        provider="Cloudflare Workers AI",
        model_id="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        api_base="https://api.cloudflare.com/client/v4/ai/v1",
        api_key_env="CLOUDFLARE_API_KEY",
        context_window=128_000,
        tier="A-",
        free_tier_note="Free up to 10 000 neurons/day via Workers AI",
        probe_path="/v1/models",
        tags=("edge", "free"),
    ),
    FreeModelEntry(
        id="github-gpt-4o-mini",
        provider="GitHub Models",
        model_id="gpt-4o-mini",
        api_base="https://models.inference.ai.azure.com",
        api_key_env="GITHUB_TOKEN",
        context_window=128_000,
        max_output_tokens=16_384,
        tier="A-",
        free_tier_note="Free for GitHub accounts; modest rate limits",
        tags=("free",),
    ),
    # ------------------------------------------------------------------
    # B tier  (20 – 25 %)
    # ------------------------------------------------------------------
    FreeModelEntry(
        id="groq-llama-3-1-8b",
        provider="Groq",
        model_id="llama-3.1-8b-instant",
        api_base="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        context_window=128_000,
        max_output_tokens=8_192,
        tier="B",
        free_tier_note="Free tier; extremely low latency for small tasks",
        tags=("fast", "small", "free"),
    ),
    FreeModelEntry(
        id="nvidia-mistral-small",
        provider="NVIDIA NIM",
        model_id="mistralai/mistral-small-3.1-24b-instruct",
        api_base="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY",
        context_window=128_000,
        tier="B",
        free_tier_note="Free quota via build.nvidia.com",
        tags=("free",),
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def by_tier(
    catalog: list[FreeModelEntry],
    min_tier: CodingTierLabel = "B",
) -> list[FreeModelEntry]:
    """Filter *catalog* to models at or above *min_tier*, sorted best-first.

    Args:
        catalog: The model list to filter.
        min_tier: The minimum acceptable tier label.

    Returns:
        Filtered, sorted list — S+ models come first.
    """
    min_rank = tier_rank(min_tier)
    return sorted(
        (m for m in catalog if m.tier_rank <= min_rank),
        key=lambda m: m.tier_rank,
    )


def by_provider(
    catalog: list[FreeModelEntry],
    provider: str,
) -> list[FreeModelEntry]:
    """Filter *catalog* to a specific provider, sorted by tier (best first).

    Args:
        catalog: The model list to filter.
        provider: Exact provider string as it appears in
            :attr:`FreeModelEntry.provider`.

    Returns:
        Filtered, sorted list.
    """
    return sorted(
        (m for m in catalog if m.provider == provider),
        key=lambda m: m.tier_rank,
    )


def by_tag(
    catalog: list[FreeModelEntry],
    tag: str,
) -> list[FreeModelEntry]:
    """Filter *catalog* to models that carry a specific tag.

    Args:
        catalog: The model list to filter.
        tag: Tag string (e.g. ``"reasoning"``, ``"fast"``, ``"code-specialist"``).

    Returns:
        Matching entries sorted by tier (best first).
    """
    return sorted(
        (m for m in catalog if tag in m.tags),
        key=lambda m: m.tier_rank,
    )


def probe_endpoints(
    catalog: list[FreeModelEntry],
) -> dict[str, str]:
    """Return a ``{name: probe_url}`` dict ready for :class:`~langchain_core.runnables.health_check.EndpointHealthChecker`.

    Args:
        catalog: The model entries to extract probe URLs from.

    Returns:
        Dict mapping :attr:`FreeModelEntry.id` → :attr:`FreeModelEntry.probe_url`.
    """
    return {m.id: m.probe_url for m in catalog}
