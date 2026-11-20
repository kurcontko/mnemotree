"""The HTTP service routes that read and delete stored memories."""

import json

import httpx
import pytest

from mnemotree import EvidenceEngine

pytest.importorskip("fastapi")
from mnemotree.service.app import create_app

ADD = {
    "request_id": "r",
    "user_id": "u",
    "session_id": "s",
    "messages": [
        {"role": "user", "content": "I moved to Warsaw", "timestamp": 1704067200000},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": "Warsaw sounds good"},
    ],
}


def client_for(app, key="k"):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {key}"},
    )


async def test_ids_from_add_and_search_fetch_and_delete_the_stored_messages(tmp_path):
    with EvidenceEngine(tmp_path / "m.db") as engine:
        async with client_for(create_app(engine, api_key="k")) as client:
            added = (await client.post("/v1/memories/add", json=ADD)).json()
            # The blank turn is skipped, so two IDs for the two stored messages, in order.
            first, second = added["ids"]
            query = {"user_id": "u", "query": "Where did I move?", "top_k": 5}
            (hit,) = (await client.post("/v1/memories/search", json=query)).json()["data"]
            assert sorted(hit["source_ids"]) == sorted([first, second])

            response = await client.post(
                "/v1/memories/get", json={"user_id": "u", "ids": [second, "nope", first, second]}
            )
            assert response.status_code == 200
            assert response.json() == {
                "data": [
                    {
                        "id": second,
                        "content": "Warsaw sounds good",
                        "role": "assistant",
                        "session_id": "s",
                        "ingested_at": response.json()["data"][0]["ingested_at"],
                    },
                    {
                        "id": first,
                        "content": "I moved to Warsaw",
                        "role": "user",
                        "session_id": "s",
                        "ingested_at": response.json()["data"][1]["ingested_at"],
                        "created_at": "2024-01-01T00:00:00+00:00",
                    },
                ],
                "missing": ["nope"],
            }

            # Another user's ID space is separate for both verbs.
            other = await client.post("/v1/memories/get", json={"user_id": "v", "ids": [first]})
            assert other.json() == {"data": [], "missing": [first]}
            other = await client.post("/v1/memories/delete", json={"user_id": "v", "id": first})
            assert other.json() == {"deleted": False, "id": first, "user_id": "v"}

            delete = {"user_id": "u", "id": first}
            response = await client.post("/v1/memories/delete", json=delete)
            assert response.json() == {"deleted": True, "id": first, "user_id": "u"}
            response = await client.post("/v1/memories/delete", json=delete)
            assert response.status_code == 200 and response.json()["deleted"] is False
            got = await client.post("/v1/memories/get", json={"user_id": "u", "ids": [first]})
            assert got.json() == {"data": [], "missing": [first]}
            # The deleted message no longer comes back from Search; its neighbor does.
            warsaw = {**query, "query": "Warsaw"}
            (hit,) = (await client.post("/v1/memories/search", json=warsaw)).json()["data"]
            assert hit["source_ids"] == [second]
            assert "Warsaw sounds good" in hit["content"] and "moved" not in hit["content"]
            # Replaying the Add cannot bring the deleted message back: it is a conflict.
            replay = await client.post("/v1/memories/add", json=ADD)
            assert replay.status_code == 409 and "forgotten" in replay.json()["detail"]
            got = await client.post("/v1/memories/get", json={"user_id": "u", "ids": [first]})
            assert got.json()["missing"] == [first]


async def test_management_routes_require_the_key_and_valid_bodies(tmp_path):
    with EvidenceEngine(tmp_path / "m.db") as engine:
        app = create_app(engine, api_key="k")
        async with client_for(app, key="wrong") as client:
            for route, body in (
                ("get", {"user_id": "u", "ids": ["a"]}),
                ("delete", {"user_id": "u", "id": "a"}),
                ("list", {"user_id": "u"}),
                ("stats", {"user_id": "u"}),
            ):
                response = await client.post(f"/v1/memories/{route}", json=body)
                assert response.status_code == 401, route
        async with client_for(app) as client:
            for route, body in (
                ("get", {"user_id": "u", "ids": []}),
                ("get", {"user_id": "u", "ids": [str(i) for i in range(101)]}),
                ("get", {"user_id": "u", "ids": [""]}),
                ("get", {"user_id": "", "ids": ["a"]}),
                ("delete", {"user_id": "u", "id": ""}),
                ("delete", {"user_id": "u", "id": "x" * 257}),
                ("delete", {"user_id": "u"}),
                ("list", {"user_id": "u", "cursor": "abc"}),
                ("list", {"user_id": "u", "cursor": "-1"}),
                ("list", {"user_id": "u", "limit": 0}),
                ("list", {"user_id": "u", "session_id": ""}),
                ("stats", {}),
                ("stats", {"user_id": " "}),
            ):
                response = await client.post(f"/v1/memories/{route}", json=body)
                assert response.status_code == 422, (route, body)
            # GET and DELETE on these paths are not routes; the verbs are POST bodies.
            assert (await client.get("/v1/memories/get")).status_code == 405


async def test_request_log_records_management_routes_without_ids_or_content(tmp_path):
    log = tmp_path / "r.jsonl"
    with EvidenceEngine(tmp_path / "m.db") as engine:
        async with client_for(create_app(engine, api_key="k", request_log=log)) as client:
            first, _ = (await client.post("/v1/memories/add", json=ADD)).json()["ids"]
            await client.post("/v1/memories/get", json={"user_id": "u", "ids": [first, "zz"]})
            await client.post("/v1/memories/delete", json={"user_id": "u", "id": first})
            await client.post("/v1/memories/list", json={"user_id": "u", "limit": 500})
    _, get, delete, listed = [json.loads(line) for line in log.read_text().splitlines()]
    assert listed["limit"] == 100 and listed["items"] == 1 and listed["session"] is False
    assert "Warsaw" not in log.read_text()
    assert get["route"] == "/v1/memories/get" and get["status"] == 200
    assert get["user_id"] == "u" and get["ids"] == 2 and get["found"] == 1
    assert delete["route"] == "/v1/memories/delete" and delete["deleted"] is True
    assert first not in log.read_text()


async def test_list_pages_a_users_memories_oldest_first(tmp_path):
    with EvidenceEngine(tmp_path / "m.db") as engine:
        async with client_for(create_app(engine, api_key="k")) as client:
            first, second = (await client.post("/v1/memories/add", json=ADD)).json()["ids"]
            later = {**ADD, "request_id": "r2", "session_id": "t", "messages": ADD["messages"][:1]}
            (third,) = (await client.post("/v1/memories/add", json=later)).json()["ids"]
            await client.post("/v1/memories/add", json={**ADD, "user_id": "v"})

            page = (
                await client.post("/v1/memories/list", json={"user_id": "u", "limit": 2})
            ).json()
            assert [item["id"] for item in page["data"]] == [first, second]
            assert page["data"][0]["content"] == "I moved to Warsaw"
            body = {"user_id": "u", "limit": 2, "cursor": page["next_cursor"]}
            page = (await client.post("/v1/memories/list", json=body)).json()
            assert [item["id"] for item in page["data"]] == [third]
            assert page["next_cursor"] is None

            body = {"user_id": "u", "session_id": "t", "limit": 500}
            page = (await client.post("/v1/memories/list", json=body)).json()
            assert [item["id"] for item in page["data"]] == [third]
            empty = await client.post("/v1/memories/list", json={"user_id": "w"})
            assert empty.json() == {"data": [], "next_cursor": None}


async def test_stats_count_one_users_memories(tmp_path):
    with EvidenceEngine(tmp_path / "m.db") as engine:
        async with client_for(create_app(engine, api_key="k")) as client:
            first, _ = (await client.post("/v1/memories/add", json=ADD)).json()["ids"]
            await client.post("/v1/memories/add", json={**ADD, "user_id": "v"})
            stats = (await client.post("/v1/memories/stats", json={"user_id": "u"})).json()
            assert stats == {"user_id": "u", "memories": 2, "chunks": 2, "requests": 1}
            await client.post("/v1/memories/delete", json={"user_id": "u", "id": first})
            stats = (await client.post("/v1/memories/stats", json={"user_id": "u"})).json()
            assert stats == {"user_id": "u", "memories": 1, "chunks": 1, "requests": 1}
            empty = (await client.post("/v1/memories/stats", json={"user_id": "w"})).json()
            assert empty == {"user_id": "w", "memories": 0, "chunks": 0, "requests": 0}
            # There is no route that erases a whole user.
            gone = await client.post("/v1/memories/delete_all", json={"user_id": "u"})
            assert gone.status_code == 404
