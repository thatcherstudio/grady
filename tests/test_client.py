from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from grady.client import ChatClient, Completion
from grady.config import ModelSpec

pytestmark = pytest.mark.anyio

MODEL = ModelSpec(label="m", base_url="http://fake.local/v1", model="test-model")
MESSAGES = [{"role": "user", "content": "hi"}]


def ok(content: str | None = "hello", usage: dict[str, int] | None = None) -> httpx.Response:
    body: dict[str, object] = {
        "choices": [{"message": {"role": "assistant", "content": content}}],
    }
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body)


class Recorder:
    """A MockTransport handler that replays responses in order and records requests."""

    def __init__(self, *responses: httpx.Response | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item


async def chat(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    retries: int = 3,
    api_key: str | None = None,
    sleeps: list[float] | None = None,
) -> Completion:
    recorded = sleeps if sleeps is not None else []

    async def fake_sleep(seconds: float) -> None:
        recorded.append(seconds)

    client = ChatClient(
        timeout=5,
        retries=retries,
        transport=httpx.MockTransport(handler),
        sleep=fake_sleep,
        rng=lambda: 0.0,
    )
    async with client:
        return await client.chat(MODEL, MESSAGES, temperature=0.0, max_tokens=10, api_key=api_key)


async def test_success_parses_text_usage_and_sends_request() -> None:
    handler = Recorder(ok("hello", {"prompt_tokens": 3, "completion_tokens": 7}))
    result = await chat(handler)
    assert result.error is None
    assert result.text == "hello"
    assert (result.prompt_tokens, result.completion_tokens, result.attempts) == (3, 7, 1)
    assert result.latency_s is not None and result.latency_s >= 0

    (request,) = handler.requests
    assert str(request.url) == "http://fake.local/v1/chat/completions"
    assert json.loads(request.content) == {
        "model": "test-model",
        "messages": MESSAGES,
        "temperature": 0.0,
        "max_tokens": 10,
        "stream": False,
    }
    assert "authorization" not in request.headers


async def test_api_key_goes_in_bearer_header() -> None:
    handler = Recorder(ok())
    await chat(handler, api_key="sk-secret")
    assert handler.requests[0].headers["authorization"] == "Bearer sk-secret"


async def test_missing_usage_gives_none_tokens() -> None:
    result = await chat(Recorder(ok("x")))
    assert (result.prompt_tokens, result.completion_tokens) == (None, None)


async def test_null_content_is_empty_text_not_error() -> None:
    result = await chat(Recorder(ok(None)))
    assert result.error is None
    assert result.text == ""


async def test_retries_5xx_then_succeeds() -> None:
    sleeps: list[float] = []
    result = await chat(Recorder(httpx.Response(503), ok("fine")), sleeps=sleeps)
    assert result.error is None
    assert result.text == "fine"
    assert result.attempts == 2
    assert sleeps == [1.0]


async def test_backoff_doubles_and_gives_up() -> None:
    sleeps: list[float] = []
    handler = Recorder(httpx.Response(500, text="boom"))
    result = await chat(handler, retries=3, sleeps=sleeps)
    assert len(handler.requests) == 4
    assert sleeps == [1.0, 2.0, 4.0]  # no sleep after the final attempt
    assert result.text is None
    assert result.error == "HTTP 500: boom (gave up after 4 attempts)"
    assert result.attempts == 4


async def test_retry_after_is_honored_and_capped() -> None:
    sleeps: list[float] = []
    responses = (
        httpx.Response(429, headers={"Retry-After": "2"}),
        httpx.Response(429, headers={"Retry-After": "600"}),
        ok(),
    )
    result = await chat(Recorder(*responses), sleeps=sleeps)
    assert result.error is None
    assert sleeps == [2.0, 60.0]


async def test_client_errors_are_not_retried() -> None:
    handler = Recorder(httpx.Response(400, json={"error": {"message": "bad model"}}))
    result = await chat(handler)
    assert len(handler.requests) == 1
    assert result.error is not None and result.error.startswith("HTTP 400:")
    assert "bad model" in result.error


async def test_timeouts_retry_then_fail() -> None:
    sleeps: list[float] = []
    handler = Recorder(httpx.ReadTimeout("slow"))
    result = await chat(handler, retries=2, sleeps=sleeps)
    assert len(handler.requests) == 3
    assert result.error == "timed out after 5s (gave up after 3 attempts)"


async def test_connection_errors_are_reported() -> None:
    result = await chat(Recorder(httpx.ConnectError("Connection refused")), retries=0)
    assert result.error == "ConnectError: Connection refused (gave up after 1 attempt)"


async def test_api_key_is_redacted_from_error_text() -> None:
    echo = httpx.Response(401, text="invalid key sk-secret-123 for this project")
    result = await chat(Recorder(echo), api_key="sk-secret-123")
    assert result.error is not None
    assert "sk-secret-123" not in result.error
    assert "[redacted]" in result.error


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (httpx.Response(200, text="<html>not json</html>"), "invalid JSON in response"),
        (httpx.Response(200, json={"choices": []}), "response has no choices"),
        (httpx.Response(200, json={"error": {"message": "overloaded"}}), "overloaded"),
        (
            httpx.Response(200, json={"choices": [{"message": {"content": [{"text": "x"}]}}]}),
            "unexpected content type list",
        ),
    ],
)
async def test_malformed_responses_are_errors(response: httpx.Response, expected: str) -> None:
    result = await chat(Recorder(response))
    assert result.error is not None and expected in result.error
    assert result.text is None


async def test_long_error_bodies_are_truncated() -> None:
    result = await chat(Recorder(httpx.Response(404, text="x" * 1000)))
    assert result.error is not None and len(result.error) < 230
