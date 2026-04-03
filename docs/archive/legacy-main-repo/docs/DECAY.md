Decay Code Review

  Critical Issues

  1. models.py:224-230 — Linear decay is fundamentally broken

  decay_amount = self.decay_rate * time_diff  # time_diff in SECONDS
  self.importance = max(0, self.importance - decay_amount)

  With default decay_rate=0.01, one day produces 0.01 * 86400 = 864 units of decay. Since importance is in
  [0, 1], any memory zeroes out in a fraction of a second. The test at test_models.py:47 only works because
  it uses 5 seconds with decay_rate=0.1: 1.0 - 0.1 * 5 = 0.5. This method is effectively unusable at any
  realistic timescale.

  2. Three inconsistent decay models coexist
  ┌───────────────────────┬────────────────────────────────────┬───────────────────┬────────────────────────┐
  │       Location        │              Formula               │     Time unit     │         Notes          │
  ├───────────────────────┼────────────────────────────────────┼───────────────────┼────────────────────────┤
  │ models.py:224         │ Linear: I -= rate * t              │ Seconds (raw)     │ Catastrophically fast  │
  ├───────────────────────┼────────────────────────────────────┼───────────────────┼────────────────────────┤
  │ scoring.py:125        │ Exponential: I * exp(-rate * t /   │ Seconds           │ Very slow with         │
  │                       │ stability)                         │ (normalized)      │ defaults               │
  ├───────────────────────┼────────────────────────────────────┼───────────────────┼────────────────────────┤
  │ adaptive_decay.py:328 │ Linear: I -= rate * (t / 86400)    │ Days              │ Disconnected from      │
  │                       │                                    │                   │ scoring                │
  └───────────────────────┴────────────────────────────────────┴───────────────────┴────────────────────────┘
  These three systems produce wildly different behaviors and are never reconciled. The adaptive system
  computes rates that aren't consumed by the core scoring pipeline.

  3. Exponential decay in scoring.py is too slow with default params

  With decay_rate=0.01 and stability=604800 (7 days):
  - After 7 days: exp(-0.01) ≈ 0.990 — only 1% decay
  - After 30 days: exp(-0.043) ≈ 0.958 — 4.2% decay
  - After 365 days: exp(-0.522) ≈ 0.594 — 40% decay after a full year

  This is extremely conservative. Most research suggests a half-life of 1-2 weeks for general memories, not a
   year.

  Design Issues

  4. Power-law forgetting may be more appropriate than exponential

  The latest research (FSRS, the state-of-the-art open-source spaced repetition algorithm now integrated into
   Anki) uses a power-law forgetting curve:

  R(t, S) = (1 + factor * t/S) ^ (-decay)

  Cognitive science literature shows that while individual item forgetting may be exponential, aggregate
  forgetting across diverse items follows a power law
  (https://memory.psych.upenn.edu/files/pubs/KahaAdle02.pdf). Since a memory system stores heterogeneous
  items, a power-law curve is a better fit. FSRS achieves
  https://github.com/open-spaced-repetition/fsrs4anki/wiki for the same retention, which is directly relevant
   here.

  5. SM-2 implementation is outdated

  The SpacedRepetitionSchedule in adaptive_decay.py:39-88 implements SM-2 (1987). SM-2 has known weaknesses:
  - Doesn't adapt to individual memory patterns
  - Fixed initial intervals (1, 6 days) regardless of difficulty
  - Easiness factor can only decrease in practice

  https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm uses a learned DSR model
  (Difficulty, Stability, Retrievability) with 17-21 parameters that outperforms SM-2 significantly. Even the
   simpler FSRS-v3 with 13 parameters is substantially better.

  6. FadeMem's approach is more principled for AI memory

  https://arxiv.org/html/2601.18642 (Jan 2026) uses a stretched exponential with layer-specific shape
  parameters:

  v(t) = v(0) * exp(-λ * (t - τ)^β)

  Where β=0.8 for long-term (sub-linear, slower fade) and β=1.2 for short-term (super-linear, faster fade).
  The adaptive decay rate λ = λ_base * exp(-μ * I(t)) couples decay to importance. This achieves 82.1%
  retention of critical facts at 55% storage vs 78.4% retention at 100% storage in baseline systems.

  7. decay_and_reinforce() at memory.py is still a stub

  This method returns None, meaning the adaptive decay system has no integration point with the core memory
  pipeline.

  8. Novelty detection is fragile

  adaptive_decay.py:279 uses hash(memory.content.lower()[:200]) — a truncated string hash. This is:
  - Easily fooled by minor rewording
  - Not using the embedding vectors that are already available
  - In-memory only, lost on restart

  Concrete Recommendations

  1. Unify decay into a single model — Remove models.py:decay_importance() or make it delegate to the scoring
   system. There should be one decay formula.
  2. Switch to power-law forgetting — Replace the exponential exp(-rate * t/S) with FSRS-style (1 + factor *
  t/S)^(-decay_power). This has better empirical support and gives a tunable half-life.
  3. Parameterize by half-life instead of raw rates — Half-life is intuitive and tunable. FadeMem uses ~11
  days (LTM) and ~5 days (STM). From half-life, derive the rate: for power-law factor = (0.5^(-1/decay_power)
   - 1) * S / t_half.
  4. Add the shape parameter (β) from FadeMem — Sub-linear decay (β < 1) for important/semantic memories,
  super-linear (β > 1) for working/transient memories. This naturally maps to MemoryType.
  5. Replace SM-2 with FSRS or a simplified DSR model — At minimum, adopt the FSRS forgetting curve. The full
   FSRS optimizer requires review history, but the scheduling formulas are straightforward.
  6. Wire adaptive decay into scoring — The AdaptiveImportanceSystem.calculate_adaptive_decay() output should
   feed into MemoryScoring._calculate_importance_score() as the decay_rate, closing the gap between
  experimental and core.
  7. Use embeddings for novelty — Replace the hash-based novelty check with cosine similarity against recent
  memory embeddings. The infrastructure (cosine_similarity in scoring.py:10) already exists.

  Summary

  The strongest issue is the fragmented decay architecture — three disconnected implementations that produce
  inconsistent behavior. The path forward is to consolidate on a single, research-backed model (power-law
  with shape parameter, parameterized by half-life) and wire the adaptive system into the core scoring
  pipeline.

  ---
  Sources:
  - https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm
  - https://arxiv.org/html/2601.18642
  - https://memory.psych.upenn.edu/files/pubs/KahaAdle02.pdf
  - https://arxiv.org/html/2601.03938v1
  - https://www.sciencedirect.com/science/article/pii/S0950705125019227
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC4492928/
  - https://help.supermemo.org/wiki/SuperMemo_Algorithm

  ═══════════════════════════════════════════════════════════════════════════════
  Research Review — What Can Be Improved (Feb 2026)
  ═══════════════════════════════════════════════════════════════════════════════

  Full details: docs/RESEARCH_REVIEW.md

  Current State Assessment

  What works well:
  - Hybrid retrieval pipeline (Vector + BM25 + NER + RRF) is solid architecture
  - 3-factor scoring (recency + relevance + importance) matches Park et al. 2023 standard
  - Power-law recency in scoring.py is ahead of many systems
  - Lite mode with local embeddings + spaCy NER = zero-LLM ingestion
  - Builder pattern keeps config manageable

  What's broken (confirmed by 2025-2026 research):
  - Three disconnected decay models (the #1 issue)
  - Exponential decay is the wrong curve for aggregate memory
  - decay_and_reinforce() is a stub with no production effect
  - Hash-based novelty misses semantic similarity
  - No memory consolidation (every major 2025-2026 framework has this)

  ─────────────────────────────────────────────────────────────────────────────
  9 Improvements Ranked by Priority (All Zero-LLM-Cost)
  ─────────────────────────────────────────────────────────────────────────────

  P1 — Unify Decay to FSRS Power-Law                              [Medium / High Impact]

    Replace all three decay models with one formula:

      R(t, S) = (1 + 19/81 * t/S) ^ (-0.5)

    Where S = stability in seconds, t = time since last access.
    Defaults: F = 19/81 (~0.2346), C = -0.5.

    Decay behavior with S = 7 days:
    - 1 day:   R = 0.981 (1.9% decay)
    - 7 days:  R = 0.900 (10% decay — by definition of stability)
    - 30 days: R = 0.745 (25.5% decay)
    - 365 days: R = 0.279 (72.1% decay)

    Why: FSRS (15M+ Anki users) and cognitive science (Kahana & Adler 2002)
    show power-law outperforms exponential for aggregate memory. Heavier tail
    = memories persist longer at long timescales but decay faster initially.

    Sources: https://expertium.github.io/Algorithm.html
             https://borretti.me/article/implementing-fsrs-in-100-lines


  P2 — Memory-Type-Specific Decay Profiles                        [Low / Medium Impact]

    ┌──────────────────┬──────────────────┬──────┬─────────────────────────────┐
    │    MemoryType     │ Stability (days) │ beta │          Rationale          │
    ├──────────────────┼──────────────────┼──────┼─────────────────────────────┤
    │ SEMANTIC          │ 30               │ 0.8  │ Facts persist; slow decay   │
    │ EPISODIC          │ 14               │ 1.0  │ Standard decay              │
    │ PROCEDURAL        │ 60               │ 0.7  │ Skills are durable          │
    │ WORKING           │ 1                │ 1.3  │ Transient; fast decay       │
    │ AUTOBIOGRAPHICAL  │ 21               │ 0.8  │ Personal facts; persistent  │
    │ PROSPECTIVE       │ 7                │ 1.2  │ Future intentions; expire   │
    │ PRIMING           │ 3                │ 1.5  │ Exposure effects; rapid     │
    │ CONDITIONING      │ 30               │ 0.8  │ Learned associations        │
    └──────────────────┴──────────────────┴──────┴─────────────────────────────┘

    FadeMem hysteresis: promote to LTM if importance >= 0.7, demote if < 0.3.

    Source: https://arxiv.org/abs/2601.18642 (FadeMem, Jan 2026)


  P3 — Embedding-Based Novelty Detection                          [Low / Medium Impact]

    Replace hash(content[:200]) with cosine similarity against store:

      similar = store.get_similar_memories(query_embedding, top_k=5)
      max_sim = max(cosine_similarity(query_emb, m.embedding) for m in similar)

      > 0.95 → REDUNDANT    > 0.85 → ROUTINE
      > 0.70 → FAMILIAR     else   → NEW

    Survives restarts. Uses existing infrastructure. Zero LLM calls.


  P4 — Upgrade Reranker                                           [Low / Medium Impact]

    Current: ms-marco-TinyBERT-L-2-v2 (low quality, ~20ms)
    Recommended: ms-marco-MiniLM-L-6-v2 (2-3x quality, ~50ms)

    2025 benchmarks show MiniLM-L-6 is the best speed/accuracy trade-off.
    For larger stores, consider ColBERT late interaction (100x faster than
    cross-encoders with comparable quality, pre-computed document reps).

    Source: https://www.zeroentropy.dev/articles/ultimate-guide-to-choosing-the-best-reranking-model-in-2025


  P5 — Memory Consolidation Pipeline                              [High / High Impact]

    Wire decay_and_reinforce() into action:

    1. HDBSCAN on embeddings → cluster similar memories
    2. Per cluster: highest-importance memory = canonical, link others
    3. Cosine > 0.95 = near-duplicate → merge metadata, keep higher importance
    4. Importance < floor AND access_count < 2 AND age > 30d → archive
    5. High importance (>0.7) → stability boost; low (<0.3) → stability cut

    Cost: O(n log n) for HDBSCAN. No LLM calls.

    Sources: https://arxiv.org/abs/2601.18642 (FadeMem)
             https://arxiv.org/abs/2506.07398 (G-Memory, NeurIPS 2025)
             https://arxiv.org/abs/2507.03724 (MemOS)


  P6 — Emotional Context in Scoring                               [Low / Low Impact]

    emotional_valence and emotional_arousal exist on MemoryItem but are
    unused in scoring. Cognitive science: high-arousal memories recalled more.

    Boost: +0.05 if arousal > 0.6, +0.03 if |valence| > 0.7.

    For ingestion without LLM: NRC VAD Lexicon v2 (55K words, dictionary
    lookup per token).

    Source: https://saifmohammad.com/WebPages/nrc-vad.html


  P7 — Selective PRF                                              [Low / Low Impact]

    PRF can cause query drift on weak initial results. Add a gate:

      if top_bm25_score < 2.0 or score_spread < 1.0:
          skip PRF  # initial results too weak, expansion would drift

    Source: https://link.springer.com/article/10.1007/s10791-021-09393-5


  P8 — Contradiction Detection (NLI)                              [Medium / Medium Impact]

    DeBERTa-v3-xsmall NLI model (~86M params, ~50ms per pair).
    At remember() time, for top-3 most similar existing memories:

      if nli_model.predict(content, similar.content)['contradiction'] > 0.8:
          memory.conflicts_with.append(similar.memory_id)

    Populates the existing conflicts_with field (currently never set).

    Source: https://huggingface.co/cross-encoder/nli-deberta-v3-xsmall


  P9 — Binary Embedding Quantization                              [Medium / Medium Impact]

    32-bit → 1-bit per dimension = 32x storage reduction, 25x retrieval
    speedup, >95% accuracy retained (with rescoring on top candidates).

    Source: https://huggingface.co/blog/embedding-quantization

  ─────────────────────────────────────────────────────────────────────────────
  Key Research Papers Referenced
  ─────────────────────────────────────────────────────────────────────────────

  FSRS (v4-v6)           Power-law forgetting, DSR model
                          https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm

  FadeMem (Jan 2026)     Stretched exponential, dual-layer memory
                          https://arxiv.org/abs/2601.18642

  G-Memory (NeurIPS '25) Three-tier hierarchical graph memory
                          https://arxiv.org/abs/2506.07398

  MemOS (Jul 2025)       Memory operating system, MemCubes
                          https://arxiv.org/abs/2507.03724

  ACAN (2025)            Cross-attention replaces LLM importance scoring
                          https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2025.1591618/full

  Memory Survey (Dec '25) Comprehensive taxonomy of agent memory
                          https://arxiv.org/abs/2512.13564

  Kahana & Adler (2002)  Power-law aggregate forgetting
                          https://memory.psych.upenn.edu/files/pubs/KahaAdle02.pdf

  NRC VAD Lexicon v2     55K word valence/arousal/dominance scores
                          https://saifmohammad.com/WebPages/nrc-vad.html

  Park et al. (2023)     Generative Agents memory architecture
                          https://arxiv.org/abs/2304.03442
