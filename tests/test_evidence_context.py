import json

import pytest

from mnemotree.evidence import Memory, Message, SearchOptions
from mnemotree.evidence.evaluation import ablate_context


async def add(memory, namespace="u", request="r", session="s"):
    return await memory.add(
        namespace,
        [
            Message(
                id=request + "question",
                role="assistant",
                content="What was your Boston marathon finish time?",
            ),
            Message(id=request + "answer", role="user", content="3 hours and 42 minutes."),
        ],
        request_id=request,
        session_id=session,
    )


async def test_context_recovers_unmatched_response_with_attribution(tmp_path):
    with Memory(tmp_path / "m.db") as memory:
        await add(memory)
        baseline = await memory.search(
            "u", "Boston", options=SearchOptions(mode="lexical", limit=1)
        )
        assert not baseline.evidence[0].neighbors
        assert "3 hours" not in baseline.context
        result = await memory.search(
            "u", "Boston", options=SearchOptions(mode="lexical", limit=1, neighbor_messages=1)
        )
        assert len(result.evidence) == 1
        anchor = result.evidence[0]
        assert anchor.position == 0 and anchor.request_id == "r"
        neighbor = anchor.neighbors[0]
        assert neighbor.source_id == "ranswer"
        assert neighbor.role == "user" and neighbor.position == 1
        assert neighbor.text == "3 hours and 42 minutes."
        rendered = json.loads(result.context)
        assert rendered["neighbors"][0]["text"] == neighbor.text
        assert memory.counter.count(result.context) == result.trace.packed_tokens


async def test_context_is_bounded_by_namespace_request_and_session(tmp_path):
    with Memory(tmp_path / "m.db") as memory:
        await add(memory)
        await add(memory, namespace="other")
        await add(memory, request="r2", session="another")
        await add(memory, request="r3")
        source = memory.source("u", "rquestion")
        neighbors = memory.store.conversation_context("u", source, neighbors=2, max_chars=1200)
        assert [n.source_id for n in neighbors] == ["ranswer"]
        with pytest.raises(ValueError, match="outside"):
            memory.store.conversation_context("other", source, neighbors=2, max_chars=1200)
        assert memory.forget("u", "ranswer")
        assert not memory.store.conversation_context("u", source, neighbors=2, max_chars=1200)


async def test_context_uses_exact_adjacent_spans_and_preserves_anchor_budget(tmp_path):
    with Memory(tmp_path / "m.db") as memory:
        await memory.add(
            "u",
            [
                Message(id="before", content="😀" * 100 + "ending"),
                Message(id="hit", content="uniquetarget"),
                Message(id="after", content="beginning" + "界" * 100),
            ],
            request_id="r",
            session_id="s",
        )
        options = SearchOptions(mode="lexical", limit=1, neighbor_messages=1, neighbor_chars=10)
        full = await memory.search("u", "uniquetarget", options=options)
        anchor = full.evidence[0]
        assert [n.text for n in anchor.neighbors] == ["😀" * 4 + "ending", "beginning界"]
        for n in anchor.neighbors:
            assert memory.source("u", n.source_id).content[n.start : n.end] == n.text
        anchor_only = anchor.model_copy(update={"neighbors": ()})
        budget = memory.counter.count(anchor_only.render())
        tight = await memory.search(
            "u", "uniquetarget", options=options.model_copy(update={"token_budget": budget})
        )
        assert len(tight.evidence) == 1 and not tight.evidence[0].neighbors
        assert tight.trace.packed_tokens == budget
        assert len(tight.trace.omitted_context_ids) == 2


async def test_neighbors_already_in_selected_anchors_are_not_repeated(tmp_path):
    with Memory(tmp_path / "m.db") as memory:
        await memory.add(
            "u",
            [Message(id="a", content="target one"), Message(id="b", content="target two")],
            request_id="r",
            session_id="s",
        )
        result = await memory.search(
            "u", "target", options=SearchOptions(mode="lexical", limit=2, neighbor_messages=1)
        )
        assert len(result.evidence) == 2
        assert not any(item.neighbors for item in result.evidence)
        assert result.context.count("target one") == 1
        assert result.context.count("target two") == 1


async def test_context_ablation_reports_anchor_and_expansion_recall(tmp_path):
    corpus, queries = tmp_path / "corpus.jsonl", tmp_path / "queries.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "namespace": "u",
                "request_id": "r",
                "session_id": "s",
                "messages": [
                    {"id": "question", "role": "assistant", "content": "Boston marathon?"},
                    {"id": "answer", "content": "3 hours 42 minutes."},
                ],
            }
        )
    )
    queries.write_text(
        json.dumps(
            {"id": "q", "namespace": "u", "query": "Boston", "relevant_source_ids": ["answer"]}
        )
    )
    with Memory(tmp_path / "m.db") as memory:
        report = await ablate_context(
            memory, corpus, queries, SearchOptions(mode="lexical", limit=1)
        )
        off, on = report["summary"]["context_off"], report["summary"]["context_on"]
        assert off["packed_source_recall"] == 0
        assert on["candidate_source_recall"] == 0
        assert on["packed_anchor_source_recall"] == 0
        assert on["expanded_source_recall"] == 1
        assert on["packed_source_recall"] == 1
        assert report["shared_ingestion"]["sources"] == 2
