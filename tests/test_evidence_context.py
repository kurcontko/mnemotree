import json
from datetime import UTC

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
        assert neighbor.offset == 1
        assert result.context == (
            "assistant: What was your Boston marathon finish time?\nuser: 3 hours and 42 minutes."
        )
        assert result.trace.packed_tokens == sum(
            memory.counter.count(item.render()) for item in result.evidence
        )


async def test_context_is_bounded_by_namespace_and_session(tmp_path):
    with Memory(tmp_path / "m.db") as memory:
        await add(memory)
        await add(memory, namespace="other")
        await add(memory, request="r2", session="another")
        source = memory.source("u", "rquestion")
        neighbors = memory.store.conversation_context("u", source, neighbors=2, max_chars=1200)
        assert [n.source_id for n in neighbors] == ["ranswer"]
        with pytest.raises(ValueError, match="outside"):
            memory.store.conversation_context("other", source, neighbors=2, max_chars=1200)
        assert memory.forget("u", "ranswer")
        assert not memory.store.conversation_context("u", source, neighbors=2, max_chars=1200)


async def test_context_crosses_add_requests_within_a_session(tmp_path):
    # AML splits a session into Add requests of at most 20 messages.
    with Memory(tmp_path / "m.db") as memory:
        await add(memory, request="r1")
        await add(memory, request="r2", session="another")
        await add(memory, request="r3")
        source = memory.source("u", "r1answer")
        neighbors = memory.store.conversation_context("u", source, neighbors=2, max_chars=1200)
        assert [(n.source_id, n.offset) for n in neighbors] == [
            ("r1question", -1),
            ("r3question", 1),
            ("r3answer", 2),
        ]
        await memory.add(
            "u",
            [Message(id="unique", content="Splits were even throughout.")],
            request_id="r4",
            session_id="s",
        )
        result = await memory.search(
            "u", "splits", options=SearchOptions(mode="lexical", limit=1, neighbor_messages=1)
        )
        assert result.context.splitlines() == [
            "user: 3 hours and 42 minutes.",
            "user: Splits were even throughout.",
        ]


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


async def test_merged_blocks_count_once_against_the_limit(tmp_path):
    with Memory(tmp_path / "m.db") as memory:
        texts = ["intro chatter", "kiwi first", "kiwi second", "plain filler", "kiwi third"]
        await memory.add(
            "u",
            [Message(id=f"m{i}", content=text) for i, text in enumerate(texts)],
            request_id="r",
            session_id="s",
        )
        await memory.add(
            "u", [Message(id="t0", content="kiwi elsewhere")], request_id="t", session_id="t"
        )
        plain = await memory.search("u", "kiwi", options=SearchOptions(mode="lexical", limit=2))
        assert sum(len(item.members()) for item in plain.evidence) == 2
        merged = await memory.search(
            "u", "kiwi", options=SearchOptions(mode="lexical", limit=2, merge_blocks=True)
        )
        groups = [{member.source_id for member in item.members()} for item in merged.evidence]
        assert len(groups) == 2 and {"m1", "m2"} in groups
        for item in merged.evidence:
            assert len({member.session_id for member in item.members()}) == 1
        block = next(item for item in merged.evidence if "m1" in item.source_ids())
        assert block.render() == "user: kiwi first\nuser: kiwi second"
        assert merged.trace.packed_tokens == sum(
            memory.counter.count(item.render()) for item in merged.evidence
        )


async def test_merge_reach_includes_neighbor_windows(tmp_path):
    with Memory(tmp_path / "m.db") as memory:
        texts = ["kiwi start", "plain middle", "kiwi end"]
        await memory.add(
            "u",
            [Message(id=f"m{i}", content=text) for i, text in enumerate(texts)],
            request_id="r",
            session_id="s",
        )
        apart = await memory.search(
            "u", "kiwi", options=SearchOptions(mode="lexical", merge_blocks=True)
        )
        assert len(apart.evidence) == 2
        joined = await memory.search(
            "u",
            "kiwi",
            options=SearchOptions(mode="lexical", merge_blocks=True, neighbor_messages=1),
        )
        assert len(joined.evidence) == 1
        assert joined.context == "user: kiwi start\nuser: plain middle\nuser: kiwi end"
        block = joined.evidence[0]
        assert {member.source_id for member in block.members()} == {"m0", "m2"}
        assert set(block.source_ids()) == {"m0", "m1", "m2"}


async def test_merged_chunks_of_one_message_render_as_one_line(tmp_path):
    content = "kiwi " * 20
    with Memory(tmp_path / "m.db", chunk_size=20, chunk_overlap=5) as memory:
        await memory.add("u", [Message(id="m", content=content)], request_id="r", session_id="s")
        result = await memory.search(
            "u", "kiwi", options=SearchOptions(mode="lexical", merge_blocks=True)
        )
        assert len(result.evidence) == 1
        assert result.context == "user: " + content


def test_render_marks_skipped_messages():
    from mnemotree.evidence.models import Evidence

    def item(seq, text):
        return Evidence(
            id=text, source_id=text, session_id="s", role="user", timestamp=None,
            start=0, end=len(text), text=text, score=1.0, channels={}, seq=seq,
        )  # fmt: skip

    block = item(1, "first").model_copy(update={"merged": (item(4, "later"), item(2, "next"))})
    assert block.render() == "user: first\nuser: next\n...\nuser: later"


def test_relative_dates_resolve_against_the_message_time():
    from datetime import datetime

    from mnemotree.evidence.temporal import resolve

    # Monday, 8 May 2023.
    said = datetime(2023, 5, 8, 13, 0, tzinfo=UTC)
    assert resolve("I went yesterday and again two weeks ago.", said) == [
        "yesterday = Sunday, 7 May 2023",
        "two weeks ago = week of 24 April 2023",
    ]
    # The longer phrase wins over the "yesterday" inside it.
    assert resolve("It was the day before yesterday.", said) == [
        "the day before yesterday = Saturday, 6 May 2023"
    ]
    assert resolve("We met on Saturday.", said) == ["on saturday = Saturday, 6 May 2023"]
    assert resolve("I'm going to the game on Saturday.", said) == [
        "on saturday = Saturday, 13 May 2023"
    ]
    assert resolve("Last month was busy, next year too.", said) == [
        "last month = April 2023",
        "next year = 2024",
    ]
    assert resolve("yesterday", None) == []


@pytest.mark.parametrize(
    "text",
    [
        "I left 1000000 days ago.",
        "It began 999999999999 days ago.",
        "That was 9999 years ago.",
        "Maybe 99999 months ago.",
    ],
)
def test_dates_out_of_range_get_no_note(text):
    from datetime import datetime

    from mnemotree.evidence.models import render_line
    from mnemotree.evidence.temporal import resolve

    said = datetime(2023, 5, 8, 13, 0, tzinfo=UTC)
    assert resolve(text + " See you tomorrow.", said) == ["tomorrow = Tuesday, 9 May 2023"]
    assert render_line(said, "user", text).endswith(f"user: {text}")


def test_render_appends_resolved_dates():
    from datetime import datetime

    from mnemotree.evidence.models import render_line

    said = datetime(2023, 5, 8, 13, 0, tzinfo=UTC)
    assert render_line(said, "Caroline", "I went yesterday.") == (
        "[2023-05-08 Mon 13:00 UTC] Caroline: I went yesterday. (yesterday = Sunday, 7 May 2023)"
    )
    assert render_line(None, "Caroline", "I went yesterday.") == "Caroline: I went yesterday."


async def test_time_order_returns_selected_items_chronologically(tmp_path):
    from datetime import datetime

    early = datetime(2023, 5, 1, tzinfo=UTC)
    late = datetime(2023, 6, 1, tzinfo=UTC)
    with Memory(tmp_path / "m.db") as memory:
        for request, (when, text) in enumerate(
            [(early, "Bought a kayak."), (late, "Kayak trip, kayak lessons, kayak club.")]
        ):
            await memory.add(
                "u",
                [Message(id=f"m{request}", content=text, timestamp=when)],
                request_id=f"r{request}",
                session_id=f"s{request}",
            )
        by_rank = await memory.search("u", "kayak", options=SearchOptions(mode="lexical"))
        by_time = await memory.search(
            "u", "kayak", options=SearchOptions(mode="lexical", order="time")
        )
    assert [item.source_id for item in by_rank.evidence] == ["m1", "m0"]
    assert [item.source_id for item in by_time.evidence] == ["m0", "m1"]
    assert by_time.context.index("Bought") < by_time.context.index("lessons")
    assert by_time.trace.packed_tokens == by_rank.trace.packed_tokens
