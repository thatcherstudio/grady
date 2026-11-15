"""A tiny OpenAI-compatible chat-completions server for tests. No real model, no secrets.

Answers come from a prompt -> answer mapping; anything else gets ``default_answer``. It records
every request and the peak number of requests in flight, and can be told to fail the next N
requests with given status codes.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any


class FakeServer:
    def __init__(
        self,
        answers: dict[str, str] | None = None,
        default_answer: str = "I don't know.",
        delay: float = 0.0,
    ) -> None:
        self.answers = dict(answers or {})
        self.default_answer = default_answer
        self.delay = delay
        self.fail_next: list[int] = []
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.max_in_flight = 0
        self._in_flight = 0
        self._lock = threading.Lock()
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}/v1"

    def __enter__(self) -> FakeServer:
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:  # keep test output quiet
                pass

            def do_POST(self) -> None:
                if self.path != "/v1/chat/completions":
                    self._send(404, {"error": {"message": f"no route {self.path}"}})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
                with server._lock:
                    server.requests.append(body)
                    server.headers.append(dict(self.headers))
                    server._in_flight += 1
                    server.max_in_flight = max(server.max_in_flight, server._in_flight)
                    status = server.fail_next.pop(0) if server.fail_next else 200
                try:
                    if server.delay:
                        time.sleep(server.delay)
                    if status != 200:
                        self._send(status, {"error": {"message": "injected failure"}})
                        return
                    prompt = body["messages"][-1]["content"]
                    answer = server.answers.get(prompt, server.default_answer)
                    self._send(
                        200,
                        {
                            "id": "chatcmpl-fake",
                            "object": "chat.completion",
                            "model": body["model"],
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": answer},
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": len(prompt.split()),
                                "completion_tokens": len(answer.split()),
                                "total_tokens": len(prompt.split()) + len(answer.split()),
                            },
                        },
                    )
                finally:
                    with server._lock:
                        server._in_flight -= 1

            def _send(self, status: int, payload: dict[str, Any]) -> None:
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler
