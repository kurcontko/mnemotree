import pytest

from mnemotree.evidence import EmbeddingSpec, Memory, Message, SearchOptions


class RecordingEmbedder:
    spec = EmbeddingSpec(provider="test", model="recording", revision="1", dimensions=2)

    def __init__(self):
        self.queries = []

    async def embed_documents(self, texts):
        return [[1.0, 0.0] if "apple" in text else [0.0, 1.0] for text in texts]

    async def embed_query(self, text):
        self.queries.append(text)
        return (await self.embed_documents([text]))[0]


async def test_question_policy_keeps_options_in_lexical_but_not_dense(tmp_path):
    class RecordingReranker:
        def __init__(self):
            self.queries = []

        async def score(self, query, texts):
            self.queries.append(query)
            return [0.0] * len(texts)

    embedder = RecordingEmbedder()
    reranker = RecordingReranker()
    with Memory(tmp_path / "m.db", embedder=embedder, reranker=reranker) as memory:
        await memory.add(
            "u",
            [Message(content="apple", id="question"), Message(content="banana", id="option")],
            request_id="r",
            session_id="s",
        )
        result = await memory.search(
            "u",
            "apple",
            query_options=["banana"],
            options=SearchOptions(mode="hybrid", model_query_policy="question"),
        )
        assert embedder.queries == reranker.queries == ["apple"]
        assert {item.source_id for item in result.evidence} == {"question", "option"}
        option = next(c for c in result.trace.candidates if c.chunk.source_id == "option")
        assert "lexical" in option.channels
        assert result.trace.original_model_query_tokens == memory.counter.count("apple")
        assert not result.trace.model_query_truncated

        await memory.search(
            "u",
            "apple",
            query_options=["banana"],
            options=SearchOptions(mode="hybrid", model_query_policy="combined"),
        )
        assert embedder.queries[-1] == reranker.queries[-1] == "apple\nbanana"


async def test_bounded_role_prior_and_gate(tmp_path):
    with Memory(tmp_path / "m.db") as memory:
        await memory.add(
            "u",
            [
                Message(content="apple", id="a-assistant", role="assistant"),
                Message(content="apple", id="z-user", role="user"),
            ],
            request_id="r",
            session_id="s",
        )
        baseline = await memory.search("u", "apple", options=SearchOptions(mode="lexical"))
        assert baseline.evidence[0].source_id == "a-assistant"

        boosted = await memory.search(
            "u",
            "apple",
            options=SearchOptions(mode="lexical", role_first="user", role_boost=0.002),
        )
        assert boosted.evidence[0].source_id == "z-user"
        assert boosted.trace.role_boost_applied
        assert not boosted.trace.role_first_applied

        gated = await memory.search(
            "u",
            "apple",
            options=SearchOptions(
                mode="lexical",
                role_first="user",
                role_boost=0.002,
                role_first_min_assistant_share=0.65,
            ),
        )
        assert not gated.trace.role_boost_applied
        assert gated.evidence[0].source_id == "a-assistant"


async def test_role_boost_feeds_but_does_not_override_reranker(tmp_path):
    class Reranker:
        async def score(self, query, texts):
            return [float("assistant" in text) for text in texts]

    with Memory(tmp_path / "m.db", reranker=Reranker()) as memory:
        await memory.add(
            "u",
            [
                Message(content="apple assistant", id="a-assistant", role="assistant"),
                Message(content="apple user", id="z-user", role="user"),
            ],
            request_id="r",
            session_id="s",
        )
        result = await memory.search(
            "u",
            "apple",
            options=SearchOptions(mode="lexical", role_first="user", role_boost=0.002),
        )
        assert result.evidence[0].source_id == "a-assistant"
        assert result.trace.role_boost_applied


def test_role_boost_requires_a_role():
    with pytest.raises(ValueError, match="requires role_first"):
        SearchOptions(role_boost=0.002)


async def test_soft_prior_keeps_a_strong_assistant_ahead_of_weak_user(tmp_path):
    with Memory(tmp_path / "m.db") as memory:
        await memory.add(
            "u",
            [Message(content="apple pear", id=f"a{i:02}", role="assistant") for i in range(19)]
            + [Message(content="apple", id="z-user", role="user")],
            request_id="r",
            session_id="s",
        )
        base = SearchOptions(mode="lexical", limit=5, role_first="user")
        hard = await memory.search("u", "apple pear", options=base)
        soft = await memory.search(
            "u", "apple pear", options=base.model_copy(update={"role_boost": 0.002})
        )
        assert hard.evidence[0].source_id == "z-user"
        assert hard.trace.role_first_applied and not hard.trace.role_boost_applied
        assert soft.evidence[0].role == "assistant"
        assert "z-user" not in {item.source_id for item in soft.evidence}
        # The prior adds a fixed amount; it does not multiply RRF or rewrite ranks.
        user = next(c for c in soft.trace.candidates if c.chunk.source_id == "z-user")
        rank = user.channels["lexical"].rank
        assert user.fusion_score == pytest.approx(1 / (base.rrf_k + rank) + 0.002)


@pytest.mark.parametrize("boost", [0, -0.001, 1.001, float("inf"), float("nan")])
def test_role_boost_has_finite_configuration_bounds(boost):
    with pytest.raises(ValueError):
        SearchOptions(role_first="user", role_boost=boost)


def test_policies_preserve_measured_service_defaults(tmp_path):
    from mnemotree.evidence.server import aml_options

    plain = SearchOptions()
    assert plain.role_boost is None and plain.model_query_policy == "combined"
    with Memory(tmp_path / "m.db") as memory:
        service = aml_options(memory)
    assert service.role_first == "user"
    assert service.role_boost is None
    assert service.role_first_min_assistant_share == 0.65
    assert service.model_query_policy == "combined"


async def test_hard_partition_still_applies_after_reranking(tmp_path):
    class AssistantReranker:
        async def score(self, query, texts):
            return [float("assistant" in text) for text in texts]

    with Memory(tmp_path / "m.db", reranker=AssistantReranker()) as memory:
        await memory.add(
            "u",
            [
                Message(content="apple assistant", role="assistant", id="a"),
                Message(content="apple user", role="user", id="z"),
            ],
            request_id="r",
            session_id="s",
        )
        result = await memory.search(
            "u", "apple", options=SearchOptions(mode="lexical", role_first="user")
        )
        assert result.evidence[0].source_id == "z"
        assert result.trace.role_first_applied
        assert not result.trace.role_boost_applied
