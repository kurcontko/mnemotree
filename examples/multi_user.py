"""One SQLite file, one memory per user. Runs offline with no API key.

    pip install mnemotree
    python examples/multi_user.py

A bot serving many users opens one engine and binds a ``Memory`` to each user's
namespace. Your code decides who the user is. The namespace keeps their data apart;
it does not authenticate anyone.
"""

import asyncio
import tempfile
from pathlib import Path

from mnemotree import EvidenceEngine, IngestConflict, Memory


async def main(db: Path) -> None:
    engine = EvidenceEngine(db)  # no embedder, so recall matches keywords

    def memory_for(user_id: str) -> Memory:
        return Memory.from_engine(engine, namespace=user_id)

    alice, bob = memory_for("alice"), memory_for("bob")
    try:
        await alice.remember("I'm allergic to peanuts.")
        await bob.remember("Peanuts are my favourite snack.")

        # Each user recalls only their own messages.
        for name, memory in (("alice", alice), ("bob", bob)):
            recall = await memory.recall("peanuts")
            print(f"{name}: {[item.content for item in recall.items]}")

        # Chat platforms redeliver webhooks. The same request_id stores the message
        # once and returns the first result again.
        first = await alice.remember("My flight to Lisbon leaves on Friday.", request_id="msg-1")
        again = await alice.remember("My flight to Lisbon leaves on Friday.", request_id="msg-1")
        print("redelivery stored once:", first.id == again.id)

        # Reusing a request_id for different text is a bug in the caller, so it raises.
        try:
            await alice.remember("My flight leaves on Sunday.", request_id="msg-1")
        except IngestConflict as conflict:
            print("conflict:", conflict)

        # "Delete what you know about my flight."
        print("forgotten:", await alice.forget(first.id))
        print("after forget:", len((await alice.recall("flight to Lisbon")).items), "memories")
    finally:
        engine.close()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as folder:
        asyncio.run(main(Path(folder) / "bot.sqlite"))
