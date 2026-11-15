"""Building the judge-check set against two fake servers, one per model. No model, no secrets."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from grady.judgeset import main
from tests.fake_server import FakeServer

PROMPTS = {f"case{i}": f"Prompt number {i}" for i in range(10)}


def rubric_yaml(cases: dict[str, str]) -> str:
    lines = ["temperature: 0.5", "max_tokens: 99", "cases:"]
    for case_id, prompt in cases.items():
        lines += [f"  - id: {case_id}", f"    prompt: {prompt!r}", "    rubric: 'Be right.'"]
    return "\n".join(lines) + "\n"


@pytest.fixture
def servers() -> Iterator[tuple[FakeServer, FakeServer]]:
    strong = FakeServer(answers={p: f"good answer to {p}" for p in PROMPTS.values()})
    weak = FakeServer(answers={p: f"weak answer to {p}" for p in PROMPTS.values()})
    with strong, weak:
        yield strong, weak


def setup_files(tmp_path: Path, *base_urls: str, cases: str | None = None) -> list[str]:
    (tmp_path / "cases.yaml").write_text(cases or rubric_yaml(PROMPTS), encoding="utf-8")
    entries = [
        f"  - {{ label: {name}, base_url: '{url}', model: fake }}"
        for name, url in zip(("strong", "weak", "tiny"), base_urls, strict=False)
    ]
    (tmp_path / "models.yaml").write_text("models:\n" + "\n".join(entries) + "\n", "utf-8")
    return [
        str(tmp_path / "cases.yaml"),
        "--models",
        str(tmp_path / "models.yaml"),
        "--out",
        str(tmp_path / "out" / "candidates.jsonl"),
        "--retries",
        "0",
        "--seed",
        "4",
    ]


def read_out(tmp_path: Path) -> list[dict[str, str]]:
    text = (tmp_path / "out" / "candidates.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def test_every_prompt_gets_one_answer_per_model(
    tmp_path: Path, servers: tuple[FakeServer, FakeServer]
) -> None:
    strong, weak = servers
    args = setup_files(tmp_path, strong.base_url, weak.base_url)

    assert main(args) == 0

    rows = read_out(tmp_path)
    assert len(rows) == 20
    for row in rows:
        expected = "good" if row["source"] == "strong" else "weak"
        assert row["output"] == f"{expected} answer to {row['prompt']}"
        assert row["rubric"] == "Be right."
    for case_id in PROMPTS:
        mine = [r for r in rows if r["case_id"] == case_id]
        assert sorted(r["id"] for r in mine) == [f"{case_id}-1", f"{case_id}-2"]
        assert sorted(r["source"] for r in mine) == ["strong", "weak"]


def test_ids_do_not_reveal_the_model(
    tmp_path: Path, servers: tuple[FakeServer, FakeServer]
) -> None:
    strong, weak = servers
    assert main(setup_files(tmp_path, strong.base_url, weak.base_url)) == 0
    first = {r["source"] for r in read_out(tmp_path) if r["id"].endswith("-1")}
    assert first == {"strong", "weak"}


def test_settings_and_warm_up_are_sent(
    tmp_path: Path, servers: tuple[FakeServer, FakeServer]
) -> None:
    strong, weak = servers
    assert main(setup_files(tmp_path, strong.base_url, weak.base_url)) == 0
    for server in servers:
        warm, *jobs = server.requests
        assert warm["max_tokens"] == 1
        assert len(jobs) == 10
        assert {(j["temperature"], j["max_tokens"]) for j in jobs} == {(0.5, 99)}
        assert all(len(j["messages"]) == 1 for j in jobs)  # no system prompt


def test_existing_output_is_kept_unless_forced(
    tmp_path: Path, servers: tuple[FakeServer, FakeServer], capsys: pytest.CaptureFixture[str]
) -> None:
    strong, weak = servers
    args = setup_files(tmp_path, strong.base_url, weak.base_url)
    out = tmp_path / "out" / "candidates.jsonl"
    out.parent.mkdir()
    out.write_text("keep me\n", encoding="utf-8")

    assert main(args) == 2
    assert "already exists" in capsys.readouterr().err
    assert out.read_text(encoding="utf-8") == "keep me\n"

    assert main([*args, "--force"]) == 0
    assert len(read_out(tmp_path)) == 20


def test_a_failed_request_writes_nothing(
    tmp_path: Path, servers: tuple[FakeServer, FakeServer], capsys: pytest.CaptureFixture[str]
) -> None:
    strong, weak = servers
    weak.fail_next = [200, 200, 400]  # warm-up and the first prompt succeed, then a hard fail
    args = setup_files(tmp_path, strong.base_url, weak.base_url)

    assert main(args) == 3
    err = capsys.readouterr().err
    assert "error: weak on case1: HTTP 400" in err
    assert "Nothing was written." in err
    assert not (tmp_path / "out" / "candidates.jsonl").exists()


def test_a_failed_warm_up_stops_before_any_prompt(
    tmp_path: Path, servers: tuple[FakeServer, FakeServer], capsys: pytest.CaptureFixture[str]
) -> None:
    strong, _ = servers
    strong.fail_next = [400]
    args = setup_files(tmp_path, strong.base_url)

    assert main(args) == 3
    assert "error: warm-up strong: HTTP 400" in capsys.readouterr().err
    assert len(strong.requests) == 1


def test_add_answers_only_the_missing_model(
    tmp_path: Path, servers: tuple[FakeServer, FakeServer]
) -> None:
    strong, weak = servers
    assert main(setup_files(tmp_path, strong.base_url, weak.base_url)) == 0
    before = read_out(tmp_path)
    sent = (len(strong.requests), len(weak.requests))

    with FakeServer(default_answer="tiny answer") as tiny:
        args = setup_files(tmp_path, strong.base_url, weak.base_url, tiny.base_url)
        assert main([*args, "--add"]) == 0
        assert len(tiny.requests) == 11  # warm-up + 10 prompts

    after = read_out(tmp_path)
    assert after[:20] == before
    assert (len(strong.requests), len(weak.requests)) == sent
    added = after[20:]
    assert {r["source"] for r in added} == {"tiny"}
    assert sorted(r["id"] for r in added) == sorted(f"{c}-3" for c in PROMPTS)


def test_add_answers_a_new_prompt_with_every_model(
    tmp_path: Path, servers: tuple[FakeServer, FakeServer]
) -> None:
    strong, weak = servers
    assert main(setup_files(tmp_path, strong.base_url, weak.base_url)) == 0
    more = rubric_yaml({**PROMPTS, "extra": "One more prompt"})
    args = setup_files(tmp_path, strong.base_url, weak.base_url, cases=more)

    assert main([*args, "--add"]) == 0

    added = read_out(tmp_path)[20:]
    assert sorted(r["id"] for r in added) == ["extra-1", "extra-2"]
    assert sorted(r["source"] for r in added) == ["strong", "weak"]


def test_add_with_nothing_missing_changes_nothing(
    tmp_path: Path, servers: tuple[FakeServer, FakeServer], capsys: pytest.CaptureFixture[str]
) -> None:
    strong, weak = servers
    args = setup_files(tmp_path, strong.base_url, weak.base_url)
    assert main(args) == 0
    before = (tmp_path / "out" / "candidates.jsonl").read_text(encoding="utf-8")
    sent = len(strong.requests)

    assert main([*args, "--add"]) == 0

    assert "Nothing to add" in capsys.readouterr().err
    assert (tmp_path / "out" / "candidates.jsonl").read_text(encoding="utf-8") == before
    assert len(strong.requests) == sent  # not even a warm-up


def test_add_refuses_a_changed_rubric(
    tmp_path: Path, servers: tuple[FakeServer, FakeServer], capsys: pytest.CaptureFixture[str]
) -> None:
    strong, weak = servers
    assert main(setup_files(tmp_path, strong.base_url, weak.base_url)) == 0
    changed = rubric_yaml(PROMPTS).replace("Be right.", "Be very right.", 1)
    args = setup_files(tmp_path, strong.base_url, weak.base_url, cases=changed)

    assert main([*args, "--add"]) == 2
    assert "case 'case0' has changed since its answers were made" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("cases", "message"),
    [
        (
            "cases:\n  - {id: a, prompt: p, rubric: r}\n  - {id: a, prompt: q, rubric: r}\n",
            "duplicate case ids: a",
        ),
        ("cases:\n  - {id: a, prompt: p}\n", "cases.0.rubric: Field required"),
        ("cases:\n  - {id: a, prompt: p, rubric: r, extra: 1}\n", "cases.0.extra: unknown key"),
    ],
)
def test_bad_rubric_files_are_config_errors(
    tmp_path: Path,
    servers: tuple[FakeServer, FakeServer],
    capsys: pytest.CaptureFixture[str],
    cases: str,
    message: str,
) -> None:
    strong, _ = servers
    assert main(setup_files(tmp_path, strong.base_url, cases=cases)) == 2
    assert message in capsys.readouterr().err
    assert strong.requests == []


def test_the_shipped_rubric_file_loads() -> None:
    from grady.judgeset import load_rubric_set

    rubrics = load_rubric_set(Path(__file__).parent.parent / "judge-check" / "cases.yaml")
    assert len(rubrics.cases) >= 12
