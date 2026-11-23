"""Call the HTTP service from your own backend.

Start the service (the ``serve`` extra), then run this file:

    pip install "mnemotree[serve]" httpx
    export MNEMOTREE_API_KEY=replace-with-your-service-key
    mnemotree-evidence --db service.sqlite serve --port 8000
    python examples/service_backend.py

Your backend authenticates its users and holds the key. Anyone with the key can read
every user's memories, so it never goes to a browser or a mobile app.
"""

import asyncio
import os
import uuid
from typing import Any

import httpx


async def add(
    client: httpx.AsyncClient, user_id: str, session_id: str, messages: list[dict[str, str]]
) -> list[str]:
    """Store consecutive turns of one conversation. Returns the new memory IDs."""
    body = {
        # Send the same request_id again when you retry, and the turns are stored once.
        "request_id": uuid.uuid4().hex,
        "user_id": user_id,
        "session_id": session_id,
        "messages": messages,
    }
    response = await client.post("/v1/memories/add", json=body)
    response.raise_for_status()
    ids: list[str] = response.json()["ids"]
    return ids


async def search(
    client: httpx.AsyncClient, user_id: str, query: str, top_k: int = 5
) -> list[dict[str, Any]]:
    """Ranked passages for one user, each with the IDs of the messages it came from."""
    body = {"user_id": user_id, "query": query, "top_k": top_k}
    response = await client.post("/v1/memories/search", json=body)
    response.raise_for_status()
    hits: list[dict[str, Any]] = response.json()["data"]
    return hits


async def main(client: httpx.AsyncClient) -> None:
    ids = await add(
        client,
        user_id="alice",
        session_id="support-42",
        messages=[
            {"role": "user", "content": "My order arrived with a cracked screen."},
            {"role": "assistant", "content": "Sorry about that. A replacement ships Monday."},
        ],
    )
    print("stored:", ids)
    for hit in await search(client, "alice", "What happened with the cracked screen?"):
        print(f"{hit['score']:.3f}  {hit['content']}")


async def run() -> None:
    async with httpx.AsyncClient(
        base_url=os.environ.get("MNEMOTREE_URL", "http://localhost:8000"),
        headers={"Authorization": f"Bearer {os.environ['MNEMOTREE_API_KEY']}"},
        timeout=30,
    ) as client:
        await main(client)


if __name__ == "__main__":
    asyncio.run(run())
