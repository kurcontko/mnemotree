import json

import httpx
import pytest

from mnemotree.evidence.providers import OpenAIEmbedder, _RetryAfterOrBackoff

ACTUAL_CLIENT = httpx.AsyncClient


def fake_client(monkeypatch, respond):
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: ACTUAL_CLIENT(**kwargs, transport=httpx.MockTransport(respond)),
    )


def vectors(request):
    body = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            "data": [{"index": i, "embedding": [float(t), 1]} for i, t in enumerate(body["input"])]
        },
    )


async def test_embedder_retries_rate_limits_outages_and_transport_errors(monkeypatch):
    calls = []

    def respond(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        if len(calls) == 2:
            return httpx.Response(503, text="upstream down")
        if len(calls) == 3:
            raise httpx.ConnectError("connection reset")
        return vectors(request)

    fake_client(monkeypatch, respond)
    embedder = OpenAIEmbedder(
        model="toy", dimensions=2, revision="1", api_key="k", backoff_seconds=0
    )
    assert await embedder.embed_documents(["7"]) == [[7.0, 1]]
    assert len(calls) == 4


async def test_embedder_gives_up_after_max_attempts_and_never_retries_client_errors(monkeypatch):
    calls = []
    fake_client(monkeypatch, lambda request: (calls.append(1), httpx.Response(500))[1])
    embedder = OpenAIEmbedder(
        model="toy", dimensions=2, revision="1", api_key="k", max_attempts=3, backoff_seconds=0
    )
    with pytest.raises(httpx.HTTPStatusError):
        await embedder.embed_documents(["1"])
    assert len(calls) == 3

    calls.clear()
    fake_client(monkeypatch, lambda request: (calls.append(1), httpx.Response(401))[1])
    with pytest.raises(httpx.HTTPStatusError):
        await embedder.embed_documents(["1"])
    assert len(calls) == 1

    with pytest.raises(ValueError):
        OpenAIEmbedder(model="toy", dimensions=2, revision="1", api_key="k", max_attempts=0)


def test_retry_after_header_sets_the_wait_for_rate_limits():
    class Outcome:
        def __init__(self, exc):
            self.exc = exc

        def exception(self):
            return self.exc

    class State:
        def __init__(self, exc):
            self.outcome = Outcome(exc)

    wait = _RetryAfterOrBackoff(lambda state: 1.5)
    request = httpx.Request("POST", "http://x/embeddings")

    def error(status, headers=None):
        response = httpx.Response(status, headers=headers, request=request)
        return httpx.HTTPStatusError("e", request=request, response=response)

    assert wait(State(error(429, {"retry-after": "7"}))) == 7.0
    assert wait(State(error(429, {"retry-after": "900"}))) == 60.0
    assert wait(State(error(429, {"retry-after": "soon"}))) == 1.5
    assert wait(State(error(503, {"retry-after": "7"}))) == 1.5
    assert wait(State(httpx.ConnectError("x"))) == 1.5
