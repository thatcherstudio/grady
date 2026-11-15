"""Async client for OpenAI-compatible ``/chat/completions`` endpoints.

One :class:`ChatClient` per run shares a connection pool across every model. It retries
transient failures (HTTP 408/429/5xx, connection errors, timeouts) with exponential backoff
and jitter, honors ``Retry-After``, and never raises for a failed request: failures come back
as a :class:`Completion` with ``error`` set, so the runner can record them.

API keys are passed per call, sent only in the ``Authorization`` header, and redacted from
any error text before it leaves this module.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import httpx

from grady.config import ModelSpec

RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})
ERROR_BODY_CHARS = 200

Message = dict[str, str]


@dataclass(frozen=True)
class Completion:
    text: str | None
    latency_s: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    attempts: int
    error: str | None = None


class ChatClient:
    def __init__(
        self,
        *,
        timeout: float,
        retries: int,
        backoff_base: float = 1.0,
        backoff_cap: float = 30.0,
        retry_after_cap: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self.timeout = timeout
        self.retries = retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.retry_after_cap = retry_after_cap
        self._sleep = sleep
        self._rng = rng
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(timeout), transport=transport)

    async def __aenter__(self) -> ChatClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._http.aclose()

    async def chat(
        self,
        model: ModelSpec,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        api_key: str | None = None,
    ) -> Completion:
        url = f"{model.base_url}/chat/completions"
        body = {
            "model": model.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        attempts = self.retries + 1
        last_error = ""

        for attempt in range(attempts):
            retry_after: str | None = None
            start = time.perf_counter()
            try:
                response = await self._http.post(url, json=body, headers=headers)
            except httpx.TimeoutException:
                last_error = f"timed out after {self.timeout:g}s"
            except httpx.TransportError as exc:
                last_error = _snippet(f"{type(exc).__name__}: {exc}", api_key)
            else:
                latency = time.perf_counter() - start
                if response.status_code < 400:
                    return _parse(response, latency, attempt + 1, api_key)
                last_error = _http_error(response, api_key)
                if response.status_code not in RETRY_STATUS:
                    return _failed(last_error, attempt + 1)
                retry_after = response.headers.get("retry-after")

            if attempt + 1 < attempts:
                await self._sleep(self._delay(attempt, retry_after))

        suffix = "attempt" if attempts == 1 else "attempts"
        return _failed(f"{last_error} (gave up after {attempts} {suffix})", attempts)

    def _delay(self, attempt: int, retry_after: str | None) -> float:
        if retry_after is not None:
            try:
                seconds = float(retry_after)
            except ValueError:
                seconds = -1.0  # HTTP-date form: fall back to backoff
            if seconds >= 0:
                return min(seconds, self.retry_after_cap)
        delay = self.backoff_base * 2.0**attempt + self.backoff_base * self._rng()
        return min(delay, self.backoff_cap)


def _parse(response: httpx.Response, latency: float, attempts: int, key: str | None) -> Completion:
    try:
        data: Any = response.json()
    except ValueError:
        return _failed(f"invalid JSON in response: {_snippet(response.text, key)}", attempts)

    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        detail = data.get("error") if isinstance(data, dict) and "error" in data else data
        return _failed(f"response has no choices: {_snippet(str(detail), key)}", attempts)

    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if content is not None and not isinstance(content, str):
        return _failed(f"unexpected content type {type(content).__name__}", attempts)

    usage = data.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    return Completion(
        text=content or "",
        latency_s=latency,
        prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
        completion_tokens=_int_or_none(usage.get("completion_tokens")),
        attempts=attempts,
    )


def _failed(error: str, attempts: int) -> Completion:
    return Completion(
        text=None,
        latency_s=None,
        prompt_tokens=None,
        completion_tokens=None,
        attempts=attempts,
        error=error,
    )


def _http_error(response: httpx.Response, key: str | None) -> str:
    return f"HTTP {response.status_code}: {_snippet(response.text, key)}"


def _snippet(text: str, key: str | None) -> str:
    text = " ".join(text.split())
    if key:
        text = text.replace(key, "[redacted]")
    if len(text) > ERROR_BODY_CHARS:
        text = text[: ERROR_BODY_CHARS - 1] + "…"
    return text or "(empty body)"


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
