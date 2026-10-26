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
            assert len(results) == 2 and len({item["id"] for item in results}) == 2
            rendered = [json.loads(item["content"]) for item in results]
            assert any(item["timestamp"] == "2024-01-01T00:00:00+00:00" for item in rendered)
            assert all(item["session_id"] == "s" for item in rendered)
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
            invalid = {**payload, "messages": [{"role": "user", "content": " ", "timestamp": 1}]}
            assert (await client.post("/v1/memories/add", json=invalid)).status_code == 422


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
    assert [body["input"] for body in seen] == [["1", "2"], ["3"]]
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
