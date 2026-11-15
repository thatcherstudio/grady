"""End-to-end: the real CLI against the fake OpenAI-compatible server. No model, no secrets."""

from __future__ import annotations

import json
import socket
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from grady.cli import app
from tests.fake_server import FakeServer

SUITE = """
suite: e2e
defaults: { system: "Be brief." }
cases:
  - id: add
    prompt: "What is 2 + 2?"
    score: [{ exact: "4" }]
  - id: planet
    prompt: "Largest planet?"
    score: [{ contains: { value: "jupiter", ignore_case: true } }]
  - id: date
    prompt: "ISO date for March 3, 2027?"
    score: [{ regex: '^2027-03-03$' }]
"""

ANSWERS = {
    "What is 2 + 2?": "4",
    "Largest planet?": "Jupiter.",
    "ISO date for March 3, 2027?": "March 3rd, 2027",  # wrong on purpose
}


@pytest.fixture
def server() -> Iterator[FakeServer]:
    with FakeServer(answers=ANSWERS) as s:
        yield s


def write_files(tmp_path: Path, *base_urls: str, key_env: str | None = None) -> tuple[Path, Path]:
    suite = tmp_path / "suite.yaml"
    suite.write_text(SUITE, encoding="utf-8")
    models = tmp_path / "models.yaml"
    entries = []
    for i, url in enumerate(base_urls):
        entry = f"  - {{ label: model-{i}, base_url: '{url}', model: fake-{i}"
        if key_env:
            entry += f", api_key_env: {key_env}"
        entries.append(entry + " }")
    models.write_text("models:\n" + "\n".join(entries) + "\n", encoding="utf-8")
    return suite, models


def run(*args: str | Path) -> tuple[int, str, str]:
    result = CliRunner().invoke(app, ["run", *map(str, args)])
    return result.exit_code, result.stdout, result.stderr


def read_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def closed_port_url() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1"


def test_happy_path_writes_records_meta_and_summary(tmp_path: Path, server: FakeServer) -> None:
    suite, models = write_files(tmp_path, server.base_url)
    out = tmp_path / "out" / "run.jsonl"
    code, stdout, stderr = run(suite, "--models", models, "--out", out)

    assert code == 0, stderr
    records = read_records(out)
    assert len(records) == 3
    by_case = {r["case_id"]: r for r in records}
    assert by_case["add"]["passed"] is True
    assert by_case["planet"]["passed"] is True
    assert by_case["date"]["passed"] is False
    assert by_case["date"]["scores"] == [
        {
            "scorer": "regex",
            "passed": False,
            "detail": "no match for /^2027-03-03$/ in 'March 3rd, 2027'",
        }
    ]
    add = by_case["add"]
    assert add["model"] == "model-0"
    assert add["repeat"] == 0
    assert add["error"] is None
    assert add["attempts"] == 1
    assert add["completion_tokens"] == 1
    assert isinstance(add["latency_s"], float)
    assert isinstance(add["tokens_per_s"], float)

    # One untimed warm-up request first, then the 3 jobs with the suite's system prompt.
    warmup_request, *job_requests = server.requests
    assert warmup_request["max_tokens"] == 1
    assert len(job_requests) == 3
    assert all(r["messages"][0] == {"role": "system", "content": "Be brief."} for r in job_requests)

    meta = json.loads((tmp_path / "out" / "run.meta.json").read_text())
    assert meta["totals"] == {"jobs": 3, "passed": 2, "failed": 1, "errors": 0, "pass_rate": 2 / 3}
    assert meta["models"] == [{"label": "model-0", "base_url": server.base_url, "model": "fake-0"}]
    assert meta["finished_at"]
    assert meta["settings"]["warmup"] is True
    (warmup,) = meta["warmup"]
    assert warmup["label"] == "model-0" and warmup["error"] is None
    assert "Warm-up model-0: ready in" in stderr

    assert "pass 2 fail 1" in stderr  # final progress line
    assert "model-0" in stdout and "67%" in stdout  # summary table


def test_threshold_gates_exit_code(tmp_path: Path, server: FakeServer) -> None:
    suite, models = write_files(tmp_path, server.base_url)
    code, _, stderr = run(
        suite, "--models", models, "--out", tmp_path / "a.jsonl", "--min-pass-rate", "0.9"
    )
    assert code == 1
    assert "Pass rate 66.7% is below --min-pass-rate 90.0%" in stderr

    code, _, _ = run(
        suite, "--models", models, "--out", tmp_path / "b.jsonl", "--min-pass-rate", "0.6"
    )
    assert code == 0


def test_multiple_models_and_selection(tmp_path: Path, server: FakeServer) -> None:
    suite, models = write_files(tmp_path, server.base_url, server.base_url)
    code, _, _ = run(suite, "--models", models, "--out", tmp_path / "all.jsonl")
    assert code == 0
    assert len(read_records(tmp_path / "all.jsonl")) == 6

    code, _, _ = run(
        suite, "--models", models, "--model", "model-1", "--out", tmp_path / "one.jsonl"
    )
    assert code == 0
    assert {r["model"] for r in read_records(tmp_path / "one.jsonl")} == {"model-1"}


def test_unreachable_endpoint_records_errors_and_exits_3(tmp_path: Path) -> None:
    suite, models = write_files(tmp_path, closed_port_url())
    out = tmp_path / "down.jsonl"
    code, _, stderr = run(suite, "--models", models, "--out", out, "--retries", "0")
    assert code == 3
    records = read_records(out)
    assert len(records) == 3
    assert all(r["error"] and r["passed"] is False and r["output"] is None for r in records)
    assert "err 3" in stderr
    assert "ConnectError" in stderr
    assert "3 request(s) failed after retries" in stderr
    assert "Warm-up model-0 failed: ConnectError" in stderr


def test_request_errors_outrank_threshold(tmp_path: Path, server: FakeServer) -> None:
    server.fail_next = [400]  # one non-retryable failure, hitting the first job
    suite, models = write_files(tmp_path, server.base_url)
    code, _, _ = run(
        suite,
        "--models",
        models,
        "--out",
        tmp_path / "x.jsonl",
        "--min-pass-rate",
        "0.0",
        "--no-warmup",
    )
    assert code == 3


def test_no_warmup_sends_only_the_jobs(tmp_path: Path, server: FakeServer) -> None:
    suite, models = write_files(tmp_path, server.base_url)
    out = tmp_path / "x.jsonl"
    code, _, stderr = run(suite, "--models", models, "--out", out, "--no-warmup")
    assert code == 0
    assert len(server.requests) == 3
    assert all(r["max_tokens"] != 1 for r in server.requests)
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["settings"]["warmup"] is False
    assert meta["warmup"] == []
    assert "Warm-up" not in stderr


def test_warmup_is_one_per_model_and_never_scored(tmp_path: Path, server: FakeServer) -> None:
    suite, models = write_files(tmp_path, server.base_url, server.base_url)
    out = tmp_path / "x.jsonl"
    code, _, _ = run(suite, "--models", models, "--out", out)
    assert code == 0
    assert len(server.requests) == 2 + 6
    assert [r["model"] for r in server.requests[:2]] == ["fake-0", "fake-1"]
    assert len(read_records(out)) == 6  # warm-ups never appear in results


def test_failed_warmup_is_reported_and_the_run_continues(
    tmp_path: Path, server: FakeServer
) -> None:
    server.fail_next = [400]  # the warm-up fails; every job then succeeds
    suite, models = write_files(tmp_path, server.base_url)
    out = tmp_path / "x.jsonl"
    code, _, stderr = run(suite, "--models", models, "--out", out)
    assert code == 0  # results are complete, so the warm-up failure doesn't fail the run
    assert "Warm-up model-0 failed: HTTP 400" in stderr
    assert "Running anyway" in stderr
    assert len(read_records(out)) == 3
    (warmup,) = json.loads(out.with_suffix(".meta.json").read_text())["warmup"]
    assert warmup["error"].startswith("HTTP 400")


def test_missing_api_key_fails_before_sending(
    tmp_path: Path, server: FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GRADY_E2E_KEY", raising=False)
    suite, models = write_files(tmp_path, server.base_url, key_env="GRADY_E2E_KEY")
    out = tmp_path / "x.jsonl"
    code, _, stderr = run(suite, "--models", models, "--out", out)
    assert code == 2
    assert "$GRADY_E2E_KEY" in stderr
    assert not out.exists()
    assert server.requests == []


def test_api_key_is_sent_but_never_written(
    tmp_path: Path, server: FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GRADY_E2E_KEY", "sk-e2e-secret")
    suite, models = write_files(tmp_path, server.base_url, key_env="GRADY_E2E_KEY")
    out = tmp_path / "x.jsonl"
    code, stdout, stderr = run(suite, "--models", models, "--out", out)
    assert code == 0
    assert server.headers[0]["Authorization"] == "Bearer sk-e2e-secret"
    for text in (out.read_text(), out.with_suffix(".meta.json").read_text(), stdout, stderr):
        assert "sk-e2e-secret" not in text


def test_config_errors_exit_2(tmp_path: Path, server: FakeServer) -> None:
    suite, models = write_files(tmp_path, server.base_url)
    code, _, stderr = run(suite, "--models", models, "--model", "nope")
    assert code == 2
    assert "unknown model label(s): nope" in stderr

    bad_suite = tmp_path / "bad.yaml"
    bad_suite.write_text("suite: x\ncases: [{id: a, prompt: p, score: [{containz: y}]}]")
    code, _, stderr = run(bad_suite, "--models", models)
    assert code == 2
    assert "unknown scorer 'containz'" in stderr
    assert server.requests == []


def test_existing_output_is_never_overwritten(tmp_path: Path, server: FakeServer) -> None:
    suite, models = write_files(tmp_path, server.base_url)
    out = tmp_path / "keep.jsonl"
    out.write_text("precious\n")
    code, _, stderr = run(suite, "--models", models, "--out", out)
    assert code == 2
    assert "already exists" in stderr
    assert out.read_text() == "precious\n"


def test_concurrency_limit_is_respected(tmp_path: Path) -> None:
    with FakeServer(answers=ANSWERS, delay=0.2) as slow:
        suite, models = write_files(tmp_path, slow.base_url, slow.base_url)  # 6 jobs
        code, _, _ = run(
            suite, "--models", models, "--out", tmp_path / "c.jsonl", "--concurrency", "2"
        )
        assert code == 0
        assert slow.max_in_flight == 2


def test_default_output_path(
    tmp_path: Path, server: FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite, models = write_files(tmp_path, server.base_url)
    monkeypatch.chdir(tmp_path)
    code, _, _ = run(suite, "--models", models)
    assert code == 0
    (out,) = (tmp_path / "results").glob("e2e-*.jsonl")
    assert len(read_records(out)) == 3
    assert out.with_suffix(".meta.json").exists()


def test_version() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.startswith("grady ")
