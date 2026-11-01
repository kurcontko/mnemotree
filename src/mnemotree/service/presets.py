"""Search options the service runs with unless configured otherwise."""

from __future__ import annotations

from ..core.models import SearchOptions
from ..engine import EvidenceEngine

# Leave room for the question and answer-model instructions. This is a configurable
# evidence allowance, not a claim about retrieval quality or a provider's context limit.
AML_TOKEN_BUDGET = 8_000


def aml_options(memory: EvidenceEngine) -> SearchOptions:
    """Service defaults: adjacent turns merged into passages, within 8K tokens."""
    return SearchOptions(
        mode="hybrid" if memory.embedder else "lexical",
        limit=100,
        token_budget=AML_TOKEN_BUDGET,
        neighbor_messages=1,
        merge_blocks=True,
        # Select diverse source anchors before completing long messages. Prefer user
        # turns only in assistant-heavy histories; deepen the pool for long messages.
        chunks_per_source=1,
        role_first="user",
        # Retain the measured hard partition; bounded priors require explicit opt-in.
        role_boost=None,
        role_first_min_assistant_share=0.65,
        model_query_policy="combined",
        candidate_limit=100,
        deep_candidate_limit=300,
        deep_history_density=2.0,
    )
