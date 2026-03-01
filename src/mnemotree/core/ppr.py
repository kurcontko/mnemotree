"""Personalized PageRank (PPR) for graph-augmented memory retrieval.

Based on HippoRAG 2 (ICML 2025): seed retrieval results are used to initialize
a personalized restart distribution, then PPR propagates activation through the
knowledge-graph link structure to surface multi-hop relevant memories.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PPRConfig:
    """Configuration for Personalized PageRank."""

    alpha: float = 0.15
    """Teleportation probability (restart probability). Higher = more weight on seeds."""

    max_iter: int = 50
    """Maximum power-iteration steps."""

    tol: float = 1e-6
    """Convergence tolerance (L1 norm of residual)."""

    weight_links: bool = True
    """Use link.strength as edge weight instead of binary 1.0."""

    max_depth: int = 3
    """Neighbourhood depth to fetch for adjacency construction."""

    ppr_weight: float = 0.5
    """Blend coefficient: final_score = ppr_weight*ppr + (1-ppr_weight)*seed_score."""

    link_types: list[str] = field(default_factory=list)
    """Optional whitelist of link_type values to include (empty = all)."""


def personalized_pagerank(
    seed_scores: dict[str, float],
    adjacency: dict[str, list[tuple[str, float]]],
    cfg: PPRConfig,
) -> dict[str, float]:
    """Power-iteration Personalized PageRank.

    Args:
        seed_scores: memory_id → initial activation (unnormalised).
        adjacency: memory_id → list of (neighbour_id, edge_weight).
        cfg: PPRConfig parameters.

    Returns:
        memory_id → final PPR score (sums to 1 over all nodes that appeared).
    """
    if not seed_scores:
        return {}

    # Normalise seed distribution
    total_seed = sum(seed_scores.values())
    if total_seed <= 0.0:
        return {}
    r0: dict[str, float] = {k: v / total_seed for k, v in seed_scores.items()}

    # Collect all nodes (seeds + neighbours)
    all_nodes: set[str] = set(r0)
    for node, neighbours in adjacency.items():
        all_nodes.add(node)
        for nb, _ in neighbours:
            all_nodes.add(nb)

    # Build column-stochastic transition matrix as sparse dict-of-dicts
    # out_weights[node] = total outgoing weight (for normalisation)
    out_weights: dict[str, float] = {}
    for node, neighbours in adjacency.items():
        total = sum(w for _, w in neighbours)
        if total > 0:
            out_weights[node] = total

    # Current PPR vector (initialise to seed)
    r: dict[str, float] = dict(r0)
    for node in all_nodes:
        if node not in r:
            r[node] = 0.0

    alpha = cfg.alpha

    for _ in range(cfg.max_iter):
        r_new: dict[str, float] = {}

        # Teleportation term: alpha * r0
        for node in all_nodes:
            r_new[node] = alpha * r0.get(node, 0.0)

        # Propagation term: (1-alpha) * A^T * r
        for node, neighbours in adjacency.items():
            w_total = out_weights.get(node, 0.0)
            if w_total <= 0.0:
                continue
            r_val = r.get(node, 0.0)
            for nb, w in neighbours:
                contrib = (1.0 - alpha) * r_val * (w / w_total)
                r_new[nb] = r_new.get(nb, 0.0) + contrib

        # Check convergence (L1)
        delta = sum(abs(r_new.get(n, 0.0) - r.get(n, 0.0)) for n in all_nodes)
        r = r_new
        if delta < cfg.tol:
            break

    return r


def build_adjacency_from_links(
    links: list,  # list[MemoryLink]
    cfg: PPRConfig,
) -> dict[str, list[tuple[str, float]]]:
    """Convert a flat list of MemoryLink objects into an adjacency dict.

    Args:
        links: MemoryLink objects (source_id → target_id).
        cfg: PPRConfig used to decide weighting / filtering.

    Returns:
        adjacency: node_id → [(neighbour_id, weight)]
    """
    allowed = set(cfg.link_types) if cfg.link_types else None
    adjacency: dict[str, list[tuple[str, float]]] = {}

    for link in links:
        if allowed and link.link_type.value not in allowed:
            continue
        weight = link.strength if cfg.weight_links else 1.0
        adjacency.setdefault(link.source_id, []).append((link.target_id, weight))

    return adjacency
