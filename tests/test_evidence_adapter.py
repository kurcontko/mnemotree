import json

import httpx
import pytest

from mnemotree.evidence import Memory, SearchOptions
from mnemotree.evidence.evaluation import evaluate
from mnemotree.evidence.providers import OpenAIEmbedder


async def test_http_contract_and_immediate_visibility(tmp_path):
    pytest.importorskip("fastapi")
    from mnemotree.evidence.server import create_app

    with Memory(tmp_path / "m.db") as memory:
        app = create_app(memory, api_key="test-key")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.get("/health")).status_code == 200
            payload = {
                "request_id": "r",
                "user_id": "u",
                "session_id": "s",
                "messages": [
                    {"role": "user", "content": "I moved to Warsaw", "timestamp": 1704067200000},
                    {"role": "assistant", "content": "Warsaw sounds good"},
                ],
            }
            assert (await client.post("/v1/memories/add", json=payload)).status_code == 401
            query = {"query": "Where did I move?", "user_id": "u"}
            assert (await client.post("/v1/memories/search", json=query)).status_code == 401
            wrong = {"Authorization": "Bearer wrong-key"}
            response = await client.post("/v1/memories/search", json=query, headers=wrong)
            assert response.status_code == 401
            client.headers["Authorization"] = "Bearer test-key"
            for _ in range(2):
                response = await client.post("/v1/memories/add", json=payload)
                assert response.status_code == 200
                assert response.json() == {
                    "success": True,
                    "request_id": "r",
                    "user_id": "u",
                    "session_id": "s",
                }
            request = {
                "query": "Where did I move?",
                "options": ["Warsaw", "Paris"],
                "user_id": "u",
                "top_k": 100,
            }
            response = await client.post("/v1/memories/search", json=request)
            assert response.status_code == 200
            results = response.json()["data"]
            # Service defaults merge adjacent turns into one passage.
            assert len(results) == 1
            assert results[0]["content"] == (
                "[2024-01-01 Mon 00:00 UTC] user: I moved to Warsaw\nassistant: Warsaw sounds good"
            )
            assert results[0]["created_at"] == "2024-01-01T00:00:00+00:00"
            assert isinstance(results[0]["score"], float)
            assert (
                await client.post("/v1/memories/search", json={**request, "user_id": "other"})
            ).json() == {"data": []}
            assert (
                await client.post("/v1/memories/search", json={**request, "top_k": 101})
            ).status_code == 422
            assert (
                await client.post("/v1/memories/search", json={**request, "query": []})
            ).status_code == 422
            changed = {**payload, "session_id": "changed"}
            assert (await client.post("/v1/memories/add", json=changed)).status_code == 409
            blank = {
                **payload,
                "request_id": "blank",
                "messages": [
                    {"role": "user", "content": " ", "timestamp": 1},
                    {"role": "assistant", "content": ""},
                ],
            }
            response = await client.post("/v1/memories/add", json=blank)
            assert response.status_code == 200 and response.json()["request_id"] == "blank"
            mixed = {
                **payload,
                "request_id": "mixed",
                "messages": [{"role": "user", "content": ""}, *payload["messages"][:1]],
            }
            assert (await client.post("/v1/memories/add", json=mixed)).status_code == 200
            assert memory.store.stats("u")["sources"] == 3
            invalid = {
                **payload,
                "messages": [{"role": "user", "content": "x", "timestamp": 10**20}],
            }
            assert (await client.post("/v1/memories/add", json=invalid)).status_code == 422


def test_service_defaults_use_merged_context_in_a_bounded_window(tmp_path):
    pytest.importorskip("fastapi")
    from mnemotree.evidence.server import AML_TOKEN_BUDGET, aml_options

    with Memory(tmp_path / "m.db") as memory:
        options = aml_options(memory)
        assert options.mode == "lexical" and options.limit == 100
        assert options.token_budget == AML_TOKEN_BUDGET < 117_760
        assert options.neighbor_messages == 1 and options.merge_blocks


async def test_provider_transport_batch_order_and_errors(monkeypatch):
    actual_client = httpx.AsyncClient
    seen = []

    def respond(request):
        body = json.loads(request.content)
        seen.append(body)
        assert request.headers["authorization"] == "Bearer test-only"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": index, "embedding": [float(text), 1]}
                    for index, text in reversed(list(enumerate(body["input"])))
                ]
            },
        )

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: actual_client(**kwargs, transport=httpx.MockTransport(respond)),
    )
    embedder = OpenAIEmbedder(
        model="toy", dimensions=2, revision="1", api_key="test-only", batch_size=2
    )
    assert await embedder.embed_documents(["1", "2", "3"]) == [[1, 1], [2, 1], [3, 1]]
    assert sorted(body["input"] for body in seen) == [["1", "2"], ["3"]]
    seen.clear()
    # Concurrent batches still return vectors in input order.
    texts = [str(value) for value in range(1, 40)]
    assert await embedder.embed_documents(texts) == [[float(text), 1] for text in texts]
    assert len(seen) == 20
    assert all(body["dimensions"] == 2 for body in seen)

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: actual_client(
            **kwargs,
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"data": []})),
        ),
    )
    with pytest.raises(ValueError, match="missing or repeated indices"):
        await embedder.embed_query("1")


async def test_reproducible_source_recall_report(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    queries = tmp_path / "queries.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "namespace": "u",
                "request_id": "r",
                "session_id": "s",
                "messages": [
                    {"id": "a", "content": "apple orchard"},
                    {"id": "b", "content": "banana fruit"},
                ],
            }
        )
        + "\n"
    )
    queries.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "id": "q1",
                    "namespace": "u",
                    "query": "apple",
                    "relevant_source_ids": ["a"],
                    "category": "single",
                },
                {
                    "id": "q2",
                    "namespace": "u",
                    "query": "apple",
                    "relevant_source_ids": ["a", "b"],
                    "category": "multi",
                },
                {"id": "q3", "namespace": "u", "query": "unknown", "relevant_source_ids": []},
            ]
        )
    )
    with Memory(tmp_path / "m.db") as memory:
        result = await evaluate(memory, corpus, queries, SearchOptions(mode="lexical"))
        assert result["summary"]["candidate_source_recall"] == 0.75
        assert result["summary"]["packed_all_sources"] == 0.5
        assert result["summary"]["no_gold_queries"] == 1
        assert result["categories"]["unspecified"]["packed_source_recall"] is None
        assert len(result["manifest"]["corpus_sha256"]) == 64
        assert result["manifest"]["search"]["mode"] == "lexical"
        with pytest.raises(ValueError, match="fresh empty"):
            await evaluate(memory, corpus, queries, SearchOptions(mode="lexical"))


async def test_payload_sampling_is_bounded(tmp_path):
    pytest.importorskip("fastapi")
    from mnemotree.evidence.server import create_app

    samples = tmp_path / "samples.jsonl"
    with Memory(tmp_path / "m.db") as memory:
        app = create_app(memory, api_key="k", sample_path=samples, sample_limit=1)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": "Bearer k"},
        ) as client:
            for index in range(3):
                payload = {
                    "request_id": f"r{index}",
                    "user_id": "u",
                    "session_id": "s",
                    "messages": [{"role": "user", "content": "Caroline: hello"}],
                }
                assert (await client.post("/v1/memories/add", json=payload)).status_code == 200
                query = {"query": "hello", "user_id": "u", "top_k": 5}
                assert (await client.post("/v1/memories/search", json=query)).status_code == 200
    rows = [json.loads(line) for line in samples.read_text().splitlines()]
    assert [row["kind"] for row in rows] == ["add", "search"]
    assert rows[0]["messages"][0]["content"] == "Caroline: hello"
