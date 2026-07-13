"""Unit tests for the free model catalog and helper functions."""

from __future__ import annotations

import pytest
from langchain_core.runnables.free_models import (
    FREE_CODING_MODELS,
    CodingTierLabel,
    FreeModelEntry,
    by_provider,
    by_tag,
    by_tier,
    probe_endpoints,
    tier_rank,
)


# ---------------------------------------------------------------------------
# tier_rank
# ---------------------------------------------------------------------------


def test_tier_rank_order() -> None:
    assert tier_rank("S+") < tier_rank("S")
    assert tier_rank("S") < tier_rank("A+")
    assert tier_rank("A+") < tier_rank("A")
    assert tier_rank("A") < tier_rank("A-")
    assert tier_rank("A-") < tier_rank("B+")
    assert tier_rank("B+") < tier_rank("B")
    assert tier_rank("B") < tier_rank("C")


def test_tier_rank_invalid() -> None:
    with pytest.raises(ValueError):
        tier_rank("Z")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# FreeModelEntry
# ---------------------------------------------------------------------------


def test_probe_url_no_trailing_slash() -> None:
    entry = FreeModelEntry(
        id="test",
        provider="Test",
        model_id="test-model",
        api_base="https://api.example.com/v1/",
        api_key_env="TEST_KEY",
        context_window=4096,
        tier="A",
        free_tier_note="free",
        probe_path="/v1/models",
    )
    assert entry.probe_url == "https://api.example.com/v1/v1/models"


def test_probe_url_default_path() -> None:
    entry = FreeModelEntry(
        id="test",
        provider="Test",
        model_id="test-model",
        api_base="https://api.example.com",
        api_key_env="TEST_KEY",
        context_window=4096,
        tier="B",
        free_tier_note="free",
    )
    assert entry.probe_url == "https://api.example.com/v1/models"


def test_entry_is_frozen() -> None:
    entry = FreeModelEntry(
        id="test",
        provider="Test",
        model_id="m",
        api_base="https://x.com",
        api_key_env="K",
        context_window=1000,
        tier="C",
        free_tier_note="free",
    )
    with pytest.raises((AttributeError, TypeError)):
        entry.id = "other"  # type: ignore[misc]


def test_entry_tier_rank_property() -> None:
    entry = FreeModelEntry(
        id="test",
        provider="Test",
        model_id="m",
        api_base="https://x.com",
        api_key_env="K",
        context_window=1000,
        tier="S+",
        free_tier_note="free",
    )
    assert entry.tier_rank == 0


# ---------------------------------------------------------------------------
# FREE_CODING_MODELS catalog
# ---------------------------------------------------------------------------


def test_catalog_is_non_empty() -> None:
    assert len(FREE_CODING_MODELS) > 0


def test_catalog_all_ids_unique() -> None:
    ids = [m.id for m in FREE_CODING_MODELS]
    assert len(ids) == len(set(ids))


def test_catalog_all_have_api_key_env() -> None:
    for m in FREE_CODING_MODELS:
        assert m.api_key_env, f"{m.id} has no api_key_env"


def test_catalog_all_have_valid_tiers() -> None:
    valid_tiers = {"S+", "S", "A+", "A", "A-", "B+", "B", "C"}
    for m in FREE_CODING_MODELS:
        assert m.tier in valid_tiers, f"{m.id} has invalid tier {m.tier}"


def test_catalog_context_windows_positive() -> None:
    for m in FREE_CODING_MODELS:
        assert m.context_window > 0, f"{m.id} has non-positive context_window"


def test_catalog_probe_urls_start_with_https() -> None:
    for m in FREE_CODING_MODELS:
        assert m.probe_url.startswith("https://"), f"{m.id} probe URL is not HTTPS"


# ---------------------------------------------------------------------------
# by_tier
# ---------------------------------------------------------------------------


def test_by_tier_returns_sorted_best_first() -> None:
    results = by_tier(FREE_CODING_MODELS, min_tier="A")
    for i in range(len(results) - 1):
        assert results[i].tier_rank <= results[i + 1].tier_rank


def test_by_tier_filters_below_min() -> None:
    results = by_tier(FREE_CODING_MODELS, min_tier="A")
    for m in results:
        assert m.tier_rank <= tier_rank("A")


def test_by_tier_s_plus_only() -> None:
    results = by_tier(FREE_CODING_MODELS, min_tier="S+")
    for m in results:
        assert m.tier == "S+"


def test_by_tier_all_included_for_c() -> None:
    all_results = by_tier(FREE_CODING_MODELS, min_tier="C")
    assert len(all_results) == len(FREE_CODING_MODELS)


def test_by_tier_empty_catalog() -> None:
    assert by_tier([], min_tier="A") == []


# ---------------------------------------------------------------------------
# by_provider
# ---------------------------------------------------------------------------


def test_by_provider_groq_only() -> None:
    results = by_provider(FREE_CODING_MODELS, "Groq")
    for m in results:
        assert m.provider == "Groq"


def test_by_provider_sorted_best_first() -> None:
    results = by_provider(FREE_CODING_MODELS, "Groq")
    for i in range(len(results) - 1):
        assert results[i].tier_rank <= results[i + 1].tier_rank


def test_by_provider_unknown_returns_empty() -> None:
    assert by_provider(FREE_CODING_MODELS, "NonExistent Inc.") == []


# ---------------------------------------------------------------------------
# by_tag
# ---------------------------------------------------------------------------


def test_by_tag_reasoning() -> None:
    results = by_tag(FREE_CODING_MODELS, "reasoning")
    assert len(results) > 0
    for m in results:
        assert "reasoning" in m.tags


def test_by_tag_fast() -> None:
    results = by_tag(FREE_CODING_MODELS, "fast")
    for m in results:
        assert "fast" in m.tags


def test_by_tag_unknown_returns_empty() -> None:
    assert by_tag(FREE_CODING_MODELS, "nonexistent-tag-xyz") == []


def test_by_tag_sorted_best_first() -> None:
    results = by_tag(FREE_CODING_MODELS, "free")
    for i in range(len(results) - 1):
        assert results[i].tier_rank <= results[i + 1].tier_rank


# ---------------------------------------------------------------------------
# probe_endpoints
# ---------------------------------------------------------------------------


def test_probe_endpoints_returns_all_ids() -> None:
    ep = probe_endpoints(FREE_CODING_MODELS)
    assert set(ep.keys()) == {m.id for m in FREE_CODING_MODELS}


def test_probe_endpoints_values_are_urls() -> None:
    ep = probe_endpoints(FREE_CODING_MODELS)
    for name, url in ep.items():
        assert url.startswith("https://"), f"{name} probe URL not HTTPS"


def test_probe_endpoints_empty_catalog() -> None:
    assert probe_endpoints([]) == {}


def test_probe_endpoints_subset() -> None:
    subset = FREE_CODING_MODELS[:3]
    ep = probe_endpoints(subset)
    assert len(ep) == 3
    for m in subset:
        assert ep[m.id] == m.probe_url
