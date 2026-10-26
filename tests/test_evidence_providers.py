import asyncio
import json
import logging
from datetime import UTC, datetime
from functools import partial

import httpx
import pytest

from mnemotree.evidence.providers import OpenAIEmbedder, _RetryAfterOrBackoff

ACTUAL_CLIENT = httpx.AsyncClient


def fake_client(monkeypatch, respond):
    monkeypatch.setattr(
        httpx, "AsyncClient", partial(ACTUAL_CLIENT, transport=httpx.MockTransport(respond))
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


class FakeTime:
    """Virtual clock: a sleeper yields once, then time jumps to its wake-up."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def clock(self):
        return self.now

    async def sleep(self, seconds):
        self.sleeps.append(seconds)
        wake = self.now + seconds
        await asyncio.sleep(0)
        self.now = max(self.now, wake)


def embedder_with(fake_time, **kwargs):
    embedder = OpenAIEmbedder(model="toy", dimensions=2, revision="1", api_key="k", **kwargs)
    embedder._clock, embedder._sleep = fake_time.clock, fake_time.sleep
    return embedder


async def test_burst_of_rate_limits_keeps_every_vector_in_order(monkeypatch):
    calls = []

    def respond(request):
        calls.append(1)
        if len(calls) <= 7:
            return httpx.Response(429, json={"error": "rate limited"})
        return vectors(request)

    fake_client(monkeypatch, respond)
    embedder = embedder_with(FakeTime(), batch_size=2, backoff_seconds=0)
    texts = [str(value) for value in range(1, 41)]
    assert await embedder.embed_documents(texts) == [[float(text), 1] for text in texts]
    assert len(calls) == 20 + 7


async def test_read_timeout_on_a_batch_is_retried(monkeypatch):
    calls = []

    def respond(request):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow upstream", request=request)
        return vectors(request)

    fake_client(monkeypatch, respond)
    embedder = embedder_with(FakeTime(), backoff_seconds=0)
    assert await embedder.embed_query("5") == [5.0, 1]
    assert len(calls) == 2


def test_retry_after_http_date_and_reset_headers():
    request = httpx.Request("POST", "http://x/embeddings")

    class State:
        def __init__(self, headers):
            response = httpx.Response(429, headers=headers, request=request)
            error = httpx.HTTPStatusError("e", request=request, response=response)
            self.outcome = type("Outcome", (), {"exception": lambda _: error})()

    now = datetime(2015, 10, 21, 7, 27, 48, tzinfo=UTC)
    wait = _RetryAfterOrBackoff(lambda state: 1.5, now=lambda: now)
    assert wait(State({"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"})) == 12.0
    assert wait(State({"retry-after": "Wed, 21 Oct 2015 09:00:00 GMT"})) == 60.0
    assert wait(State({"retry-after": "Wed, 21 Oct 2015 07:00:00 GMT"})) == 1.5
    assert wait(State({"retry-after": "-3"})) == 1.5
    assert wait(State({"retry-after": "nan"})) == 1.5
    assert wait(State({"retry-after": "0"})) == 0.0
    assert (
        wait(State({"x-ratelimit-reset-requests": "2", "x-ratelimit-reset-tokens": "4.5"})) == 4.5
    )
    assert wait(State({"x-ratelimit-reset-tokens": "6m0s"})) == 1.5
    assert wait(State({"retry-after": "3", "x-ratelimit-reset-tokens": "9"})) == 3.0


async def test_rate_limit_puts_every_concurrent_call_into_a_shared_cool_down(monkeypatch, caplog):
    fake_time = FakeTime()
    posts = []

    def respond(request):
        posts.append(fake_time.now)
        if len(posts) == 1:
            return httpx.Response(429, headers={"retry-after": "5"})
        return vectors(request)

    fake_client(monkeypatch, respond)
    embedder = embedder_with(fake_time, batch_size=1, max_concurrency=1)
    embedder.api_key = "sk-test-secret"
    caplog.set_level(logging.INFO, logger="mnemotree.evidence.providers")
    first, second = await asyncio.gather(
        embedder.embed_documents(["1001", "1002"]), embedder.embed_documents(["2001", "2002"])
    )
    assert first == [[1001.0, 1], [1002.0, 1]]
    assert second == [[2001.0, 1], [2002.0, 1]]
    assert len(posts) == 4 + 1
    assert posts[0] == 0.0
    assert all(at >= 5.0 for at in posts[1:])

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == ["embedding retry attempt=1 status=429 wait_s=5.00 batch_size=1"]
    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert infos == ["embedding cool-down ended cooldown_s=5.00"]
    for secret in ("sk-test-secret", "1001", "1002", "2001", "2002", "Bearer"):
        assert secret not in caplog.text


async def test_concurrency_limit_is_shared_by_all_calls_on_the_instance(monkeypatch):
    in_flight, peak = [0], [0]

    async def respond(request):
        in_flight[0] += 1
        peak[0] = max(peak[0], in_flight[0])
        for _ in range(3):
            await asyncio.sleep(0)
        in_flight[0] -= 1
        return vectors(request)

    fake_client(monkeypatch, respond)
    embedder = embedder_with(FakeTime(), batch_size=1, max_concurrency=2)
    texts = [str(value) for value in range(10)]
    results = await asyncio.gather(*(embedder.embed_documents(texts) for _ in range(3)))
    assert results == [[[float(text), 1] for text in texts]] * 3
    assert peak[0] == 2


async def test_total_wait_budget_raises_the_last_error_instead_of_hanging(monkeypatch):
    calls = []

    def respond(request):
        calls.append(1)
        return httpx.Response(429, headers={"retry-after": "30"})

    fake_client(monkeypatch, respond)
    fake_time = FakeTime()
    embedder = embedder_with(fake_time, max_total_wait=50)
    with pytest.raises(httpx.HTTPStatusError) as raised:
        await embedder.embed_query("1")
    assert raised.value.response.status_code == 429
    assert len(calls) == 2
    assert sum(fake_time.sleeps) <= 50

    with pytest.raises(ValueError):
        OpenAIEmbedder(model="toy", dimensions=2, revision="1", api_key="k", max_total_wait=-1)
