import asyncio
import hashlib
import json
import threading

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
            assert (await client.get("/openapi.json")).status_code == 404
            # Pre-run probes: HEAD with the key is 200, without it 401, GET stays 405.
            assert (await client.head("/v1/memories/add")).status_code == 401
            for route in ("/v1/memories/add", "/v1/memories/search"):
                probe = await client.head(route, headers={"Authorization": "Bearer test-key"})
                assert probe.status_code == 200 and probe.content == b""
            assert (await client.get("/v1/memories/add")).status_code == 405
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
            # Oversized top_k is clamped (a 4xx is never retried); unknown keys are ignored.
            response = await client.post(
                "/v1/memories/search", json={**request, "top_k": 500, "run_id": "x"}
            )
            assert response.status_code == 200 and len(response.json()["data"]) == 1
            assert (
                await client.post("/v1/memories/search", json={**request, "top_k": 0})
            ).status_code == 422
            replay = {**payload, "extra": {"ignored": True}}
            assert (await client.post("/v1/memories/add", json=replay)).status_code == 200
            for headers in ({"Authorization": "Token test-key"}, {"X-Api-Key": "test-key"}):
                response = await client.post(
                    "/v1/memories/search", json=request, headers={"Authorization": "", **headers}
                )
                assert response.status_code == 200
            for headers in ({"Authorization": "Bearer wrong"}, {"X-Api-Key": "wrong"}):
                response = await client.post(
                    "/v1/memories/search", json=request, headers={"Authorization": "", **headers}
                )
                assert response.status_code == 401
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


def read_log(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


async def test_request_log_records_add_and_search_metadata_only(tmp_path):
    pytest.importorskip("fastapi")
    from mnemotree.evidence.server import create_app

    log = tmp_path / "logs" / "r.jsonl"  # the parent directory does not exist yet
    with Memory(tmp_path / "m.db") as memory:
        app = create_app(memory, api_key="k", request_log=log)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": "Bearer k"},
        ) as client:
            payload = {
                "request_id": "r",
                "user_id": "u",
                "session_id": "s",
                "messages": [
                    {"role": "user", "content": "I moved to Warsaw", "timestamp": 1704067200000},
                    {"role": "assistant", "content": "Warsaw sounds good"},
                    {"role": "user", "content": " "},
                ],
            }
            assert (await client.post("/v1/memories/add", json=payload)).status_code == 200
            query = {"query": "Where did I move?", "options": ["A", "B"], "user_id": "u"}
            response = await client.post("/v1/memories/search", json={**query, "top_k": 500})
            assert response.status_code == 200
            assert (await client.post("/v1/memories/add", json=payload)).status_code == 200
    add, search, replay = read_log(log)
    for row, route in ((add, "add"), (search, "search")):
        assert row["route"] == f"/v1/memories/{route}" and row["method"] == "POST"
        assert row["status"] == 200 and isinstance(row["ms"], float)
        assert row["auth_scheme"] == "bearer" and row["user_id"] == "u"
        assert row["ts"].endswith("+00:00") and len(row["boot"]) == 12
    assert search["seq"] == add["seq"] + 1 and search["boot"] == add["boot"]
    assert add["request_id"] == "r" and add["session_id"] == "s"
    assert add["messages"] == 3 and add["blank_skipped"] == 1
    assert add["words"] == 7 and add["chars"] == len("I moved to WarsawWarsaw sounds good ")
    assert add["roles"] == {"user": 2, "assistant": 1} and add["with_timestamp"] == 1
    assert add["replay"] is False and replay["replay"] is True
    assert isinstance(add["add_ms"], float)
    assert search["query_chars"] == len(query["query"]) and search["options"] == 2
    assert search["top_k"] == 500 and search["limit"] == 100
    assert search["items"] == 1 and search["sources"] == 2
    assert search["packed_tokens"] > 0 and search["candidate_limit"] == 100
    assert search["chunk_density"] == 1.0  # the service gate measured 2 chunks / 2 messages
    assert set(search["channel_counts"]) == {"lexical"}
    assert set(search["timings_ms"]) == {"embedding", "retrieval", "reranking", "packing", "total"}
    assert search["returned_roles"] == {"user": 1}
    ids = [item["id"] for item in response.json()["data"]]
    assert search["ids_digest"] == hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
    text = log.read_text()
    assert "Warsaw" not in text and "Where did I move" not in text and "Bearer" not in text


async def test_request_log_explains_validation_and_auth_failures(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from pydantic import BaseModel

    from mnemotree.evidence.server import create_app

    log = tmp_path / "r.jsonl"
    with Memory(tmp_path / "m.db") as memory:
        app = create_app(memory, api_key="k", request_log=log)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            invalid = {"query": [], "user_id": "secret-user", "top_k": 5}
            response = await client.post(
                "/v1/memories/search", json=invalid, headers={"X-Api-Key": "k"}
            )
            assert response.status_code == 422
            denied = await client.post(
                "/v1/memories/search",
                json={"query": "x", "user_id": "u", "top_k": 1},
                headers={"Authorization": "Token wrong"},
            )
            assert denied.status_code == 401
            assert denied.json() == {"detail": "invalid credentials"}

    # The 422 body is FastAPI's default, produced by an app without the logging hooks.
    class Search(BaseModel):
        query: str
        user_id: str
        top_k: int

    plain = FastAPI()

    @plain.post("/v1/memories/search")
    async def search(request: Search):
        return {}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=plain), base_url="http://test"
    ) as client:
        expected = await client.post("/v1/memories/search", json=invalid)
    assert response.content == expected.content

    validation, unauthorized = read_log(log)
    assert validation["status"] == 422 and validation["auth_scheme"] == "x-api-key"
    assert validation["detail"] == ["body.query: Input should be a valid string"]
    assert unauthorized["status"] == 401 and unauthorized["auth_scheme"] == "token"
    assert unauthorized["detail"] == "invalid credentials"
    assert "secret-user" not in log.read_text() and "wrong" not in log.read_text()


async def test_unlimited_payload_sampling_joins_the_request_log(tmp_path):
    pytest.importorskip("fastapi")
    from mnemotree.evidence.server import create_app

    log, samples = tmp_path / "r.jsonl", tmp_path / "samples" / "p.jsonl"
    with Memory(tmp_path / "m.db") as memory:
        app = create_app(
            memory, api_key="k", sample_path=samples, sample_limit=None, request_log=log
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": "Bearer k"},
        ) as client:
            for index in range(60):
                payload = {
                    "request_id": f"r{index}",
                    "user_id": "u",
                    "session_id": "s",
                    "messages": [{"role": "user", "content": f"note {index}"}],
                }
                assert (await client.post("/v1/memories/add", json=payload)).status_code == 200
            query = {"query": "note", "user_id": "u", "top_k": 5}
            assert (await client.post("/v1/memories/search", json=query)).status_code == 200
    rows, logged = read_log(samples), read_log(log)
    assert len(rows) == len(logged) == 61
    assert [row["kind"] for row in rows] == ["add"] * 60 + ["search"]
    for row, line in zip(rows, logged, strict=True):
        assert (row["seq"], row["boot"], row["ts"]) == (line["seq"], line["boot"], line["ts"])
    assert rows[-1]["query"] == "note" and rows[0]["messages"][0]["content"] == "note 0"


async def test_provider_failure_logs_status_without_provider_body(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from mnemotree.evidence.server import create_app

    log = tmp_path / "r.jsonl"
    secret = "rate limited for key sk-provider-SECRET"
    with Memory(tmp_path / "m.db") as memory:

        async def failing_search(*args, **kwargs):
            request = httpx.Request("POST", "https://provider.test/v1/embeddings")
            response = httpx.Response(
                429, text=secret, headers={"x-request-key": secret}, request=request
            )
            raise httpx.HTTPStatusError(secret, request=request, response=response)

        monkeypatch.setattr(memory, "search", failing_search)
        app = create_app(memory, api_key="k", request_log=log)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": "Bearer k"},
        ) as client:
            query = {"query": "x", "user_id": "u", "top_k": 5}
            response = await client.post("/v1/memories/search", json=query)
            assert response.status_code == 503
            assert response.json() == {"detail": "retrieval failed; retry this request"}
    (row,) = read_log(log)
    assert row["status"] == 503 and row["error"] == "HTTPStatusError"
    assert row["provider_status"] == 429
    assert row["detail"] == "retrieval failed; retry this request"
    text = log.read_text()
    assert "SECRET" not in text and "rate limited" not in text and "provider.test" not in text


def test_cli_names_a_missing_credential_variable(monkeypatch):
    from mnemotree.evidence.cli import secret

    monkeypatch.delenv("NO_SUCH_KEY", raising=False)
    with pytest.raises(SystemExit, match="set the NO_SUCH_KEY environment variable"):
        secret("NO_SUCH_KEY")


async def test_request_log_bounds_unmatched_paths(tmp_path):
    pytest.importorskip("fastapi")
    from mnemotree.evidence.server import MAX_LOGGED_PATH, create_app

    log = tmp_path / "r.jsonl"
    with Memory(tmp_path / "m.db") as memory:
        app = create_app(memory, api_key="k", request_log=log)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.get("/" + "x" * 1000)).status_code == 404
    (row,) = read_log(log)
    assert row["status"] == 404
    assert row["route"] == "/" + "x" * (MAX_LOGGED_PATH - 1)


async def test_search_clamps_top_k_in_the_options_the_engine_receives(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from mnemotree.evidence.server import MAX_TOP_K, create_app

    with Memory(tmp_path / "m.db") as memory:
        limits = []
        search = memory.search

        async def recording_search(namespace, query, *, options, **kwargs):
            limits.append(options.limit)
            return await search(namespace, query, options=options, **kwargs)

        monkeypatch.setattr(memory, "search", recording_search)
        app = create_app(memory, api_key="k")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": "Bearer k"},
        ) as client:
            for top_k in (7, MAX_TOP_K, 500):
                query = {"query": "anything", "user_id": "u", "top_k": top_k}
                assert (await client.post("/v1/memories/search", json=query)).status_code == 200
    assert limits == [7, MAX_TOP_K, MAX_TOP_K]


def test_service_refuses_a_dense_database_without_its_embedder(tmp_path):
    pytest.importorskip("fastapi")
    from mnemotree.evidence import EmbeddingSpec
    from mnemotree.evidence.server import create_app

    class Embedder:
        spec = EmbeddingSpec(provider="test", model="toy", revision="1", dimensions=2)

    Memory(tmp_path / "m.db", embedder=Embedder()).close()
    lexical = SearchOptions(mode="lexical")
    with Memory(tmp_path / "m.db") as memory, pytest.raises(ValueError, match="embedder"):
        create_app(memory, api_key="test-key", options=lexical)


async def test_health_answers_while_a_search_waits_in_the_store(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from mnemotree.evidence.server import create_app

    with Memory(tmp_path / "m.db") as memory:
        entered, release, returned = threading.Event(), threading.Event(), threading.Event()
        lexical = memory.store.lexical

        def held_lexical(*args, **kwargs):
            entered.set()
            # Bounded, so store work on the event loop fails the test instead of hanging it.
            release.wait(timeout=5)
            returned.set()
            return lexical(*args, **kwargs)

        monkeypatch.setattr(memory.store, "lexical", held_lexical)
        app = create_app(memory, api_key="k")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": "Bearer k"},
        ) as client:
            payload = {
                "request_id": "r",
                "user_id": "u",
                "session_id": "s",
                "messages": [{"role": "user", "content": "I moved to Warsaw"}],
            }
            assert (await client.post("/v1/memories/add", json=payload)).status_code == 200
            query = {"query": "Where did I move?", "user_id": "u", "top_k": 5}
            search = asyncio.create_task(client.post("/v1/memories/search", json=query))
            assert await asyncio.to_thread(entered.wait, 5)
            health = await client.get("/health")
            answered_while_held = not returned.is_set()
            release.set()
            assert (await search).status_code == 200
    assert health.status_code == 200
    assert answered_while_held
