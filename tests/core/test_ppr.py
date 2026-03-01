"""Unit tests for PPR algorithm (src/mnemotree/core/ppr.py)."""
from __future__ import annotations

import pytest

from mnemotree.core.models import LinkType, MemoryLink
from mnemotree.core.ppr import (
    PPRConfig,
    build_adjacency_from_links,
    personalized_pagerank,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _link(src: str, tgt: str, strength: float = 1.0) -> MemoryLink:
    return MemoryLink(
        source_id=src,
        target_id=tgt,
        link_type=LinkType.REFERENCES,
        strength=strength,
    )


# ---------------------------------------------------------------------------
# personalized_pagerank
# ---------------------------------------------------------------------------


def test_ppr_empty_seeds_returns_empty():
    result = personalized_pagerank({}, {}, PPRConfig())
    assert result == {}


def test_ppr_single_seed_no_neighbours():
    """Isolated node with alpha=1 (pure restart) should keep all weight on itself."""
    scores = personalized_pagerank({"A": 1.0}, {}, PPRConfig(alpha=1.0))
    assert set(scores) == {"A"}
    assert abs(scores["A"] - 1.0) < 1e-6


def test_ppr_converges_two_nodes():
    """A → B with alpha=1 should keep everything on A (pure restart)."""
    seeds = {"A": 1.0}
    adjacency = {"A": [("B", 1.0)]}
    cfg = PPRConfig(alpha=1.0, max_iter=10, tol=1e-9)
    result = personalized_pagerank(seeds, adjacency, cfg)
    assert result["A"] > result.get("B", 0.0)


def test_ppr_propagates_through_chain():
    """A → B → C: C should receive non-zero PPR score."""
    seeds = {"A": 1.0}
    adjacency = {"A": [("B", 1.0)], "B": [("C", 1.0)]}
    cfg = PPRConfig(alpha=0.15, max_iter=100, tol=1e-9)
    result = personalized_pagerank(seeds, adjacency, cfg)
    assert result.get("C", 0.0) > 0.0


def test_ppr_normalised_seeds():
    """Multiple seeds with different activations should be normalised."""
    seeds = {"A": 2.0, "B": 8.0}
    result = personalized_pagerank(seeds, {}, PPRConfig(alpha=1.0))
    assert abs(result["A"] - 0.2) < 1e-6
    assert abs(result["B"] - 0.8) < 1e-6


def test_ppr_weighted_edges():
    """Stronger edges should attract more activation.

    With alpha > 0 the teleportation maintains flow through A, so both B and C
    receive positive steady-state scores and C's should be larger (edge 0.9 > 0.1).
    """
    seeds = {"A": 1.0}
    adjacency = {"A": [("B", 0.1), ("C", 0.9)]}
    cfg = PPRConfig(alpha=0.15, max_iter=500, tol=1e-9)
    result = personalized_pagerank(seeds, adjacency, cfg)
    assert result.get("C", 0.0) > result.get("B", 0.0)


def test_ppr_convergence_tolerance():
    """Result should stabilise within max_iter and seed nodes have positive scores."""
    seeds = {"A": 1.0, "B": 0.5}
    adjacency = {
        "A": [("B", 1.0), ("C", 0.5)],
        "B": [("A", 0.8), ("D", 0.3)],
    }
    cfg = PPRConfig(alpha=0.15, max_iter=200, tol=1e-8)
    result = personalized_pagerank(seeds, adjacency, cfg)
    # Seed nodes should have non-zero scores
    assert result.get("A", 0.0) > 0.0
    assert result.get("B", 0.0) > 0.0
    # Graph scores should be non-negative
    assert all(v >= 0.0 for v in result.values())


# ---------------------------------------------------------------------------
# build_adjacency_from_links
# ---------------------------------------------------------------------------


def test_build_adjacency_basic():
    links = [_link("A", "B"), _link("A", "C"), _link("B", "C")]
    adj = build_adjacency_from_links(links, PPRConfig())
    assert set(adj) == {"A", "B"}
    assert ("B", 1.0) in adj["A"]
    assert ("C", 1.0) in adj["A"]
    assert ("C", 1.0) in adj["B"]


def test_build_adjacency_weight_links_false():
    link = _link("A", "B", strength=0.3)
    adj = build_adjacency_from_links([link], PPRConfig(weight_links=False))
    assert adj["A"] == [("B", 1.0)]


def test_build_adjacency_weight_links_true():
    link = _link("A", "B", strength=0.7)
    adj = build_adjacency_from_links([link], PPRConfig(weight_links=True))
    assert adj["A"] == [("B", 0.7)]


def test_build_adjacency_link_type_filter():
    links = [
        MemoryLink(source_id="A", target_id="B", link_type=LinkType.CAUSES, strength=1.0),
        MemoryLink(source_id="A", target_id="C", link_type=LinkType.PART_OF, strength=1.0),
    ]
    cfg = PPRConfig(link_types=["causes"])
    adj = build_adjacency_from_links(links, cfg)
    assert len(adj["A"]) == 1
    assert adj["A"][0][0] == "B"


def test_build_adjacency_empty_links():
    adj = build_adjacency_from_links([], PPRConfig())
    assert adj == {}
