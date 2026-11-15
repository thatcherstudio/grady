from __future__ import annotations

import json
from pathlib import Path

import pytest

from grady.config import (
    ConfigError,
    load_models,
    load_suite,
    resolve_api_keys,
    select_models,
)

EXAMPLES = Path(__file__).parent.parent / "examples"


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def suite_error(tmp_path: Path, text: str, name: str = "s.yaml") -> str:
    with pytest.raises(ConfigError) as info:
        load_suite(write(tmp_path, name, text))
    return str(info.value)


GOOD_SUITE = """
suite: demo
defaults: { temperature: 0.2, max_tokens: 100, system: "Be brief." }
cases:
  - id: a
    prompt: "Say hi"
    score:
      - contains: "hi"
  - id: b
    prompt: "Say bye"
    system: "Be terse."
    temperature: 0.9
    max_tokens: 5
    score:
      - exact: "bye"
      - regex: "(?i)bye"
"""


# --- shipped examples must always load ----------------------------------------------------------


def test_example_suite_loads() -> None:
    suite = load_suite(EXAMPLES / "basic.yaml")
    assert suite.suite == "basic"
    assert len(suite.cases) == 8


def test_example_models_load() -> None:
    labels = [m.label for m in load_models(EXAMPLES / "models.local.yaml")]
    assert "ollama-gemma3-4b" in labels


# --- suites -------------------------------------------------------------------------------------


def test_yaml_suite_and_resolved_settings(tmp_path: Path) -> None:
    suite = load_suite(write(tmp_path, "s.yaml", GOOD_SUITE))
    a, b = suite.cases
    assert [s.kind for s in b.score] == ["exact", "regex"]
    assert (suite.system_for(a), suite.temperature_for(a), suite.max_tokens_for(a)) == (
        "Be brief.",
        0.2,
        100,
    )
    assert (suite.system_for(b), suite.temperature_for(b), suite.max_tokens_for(b)) == (
        "Be terse.",
        0.9,
        5,
    )


def test_builtin_defaults(tmp_path: Path) -> None:
    suite = load_suite(
        write(tmp_path, "s.yml", "suite: x\ncases: [{id: a, prompt: p, score: [{exact: y}]}]")
    )
    case = suite.cases[0]
    assert (suite.system_for(case), suite.temperature_for(case), suite.max_tokens_for(case)) == (
        None,
        0.0,
        512,
    )


def test_jsonl_suite_takes_name_from_file(tmp_path: Path) -> None:
    lines = [
        json.dumps({"id": "a", "prompt": "p1", "score": [{"exact": "1"}]}),
        "",
        json.dumps({"id": "b", "prompt": "p2", "score": [{"contains": ["x", "y"]}]}),
    ]
    suite = load_suite(write(tmp_path, "bulk-cases.jsonl", "\n".join(lines)))
    assert suite.suite == "bulk-cases"
    assert [c.id for c in suite.cases] == ["a", "b"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "suite: x\ncases:\n  - {id: a, prompt: p, score: [{exact: y}]}\n"
            "  - {id: a, prompt: q, score: [{exact: y}]}",
            "duplicate case id 'a' (cases[0] and cases[1])",
        ),
        (
            "suite: x\ncases:\n  - {id: ok, prompt: p, score: [{exact: y}]}\n"
            "  - {id: capital-02, prompt: p, score: [{containz: y}]}",
            "s.yaml: cases[1] (id 'capital-02'): score[0]: unknown scorer 'containz' "
            "(known: contains, exact, regex)",
        ),
        (
            "suite: x\ncases: [{id: a, prompt: p, score: [{regex: '(oops'}]}]",
            "cases[0] (id 'a'): score[0]: regex: invalid pattern '(oops'",
        ),
        ("suite: x\ncases: [{id: a, prompt: p, score: []}]", "cases[0] (id 'a'): score:"),
        ("suite: x\ncases: [{id: a, prompt: p}]", "cases[0] (id 'a'): score: Field required"),
        (
            "suite: x\ncases: [{id: a, prompt: p, score: [{exact: y, contains: z}]}]",
            "score[0]: each entry must be a one-key mapping",
        ),
        (
            "suite: x\ncases: [{id: a, prompt: p, repeats: 3, score: [{exact: y}]}]",
            "cases[0] (id 'a'): repeats: unknown key",
        ),
        (
            "suite: x\ncases: [{id: 'has space', prompt: p, score: [{exact: y}]}]",
            "cases[0] (id 'has space'): id: String should match pattern",
        ),
        ("suite: x\ncases: [{id: a, prompt: p, score: [{contains: 1250}]}]", '(quote it: "1250")'),
        ("suite: x\ncases: []", "'cases' must be a non-empty list"),
        ("suite: x\nfoo: 1\ncases: [{id: a, prompt: p, score: [{exact: y}]}]", "foo: unknown key"),
        ("cases: [{id: a, prompt: p, score: [{exact: y}]}]", "suite: Field required"),
        ("- just\n- a list", "expected a mapping with 'suite' and 'cases' keys"),
        ("suite: [unclosed", "invalid YAML"),
    ],
)
def test_yaml_suite_errors(tmp_path: Path, text: str, expected: str) -> None:
    assert expected in suite_error(tmp_path, text)


def test_jsonl_errors_name_the_line(tmp_path: Path) -> None:
    good = json.dumps({"id": "a", "prompt": "p", "score": [{"exact": "1"}]})
    message = suite_error(tmp_path, good + "\n{not json", name="c.jsonl")
    assert message.startswith(f"{tmp_path / 'c.jsonl'}:2: invalid JSON")

    bad_case = json.dumps({"id": "b", "prompt": "p", "score": [{"nope": "1"}]})
    message = suite_error(tmp_path, good + "\n" + bad_case, name="c.jsonl")
    assert ":2 (id 'b'): score[0]: unknown scorer 'nope'" in message

    dup = suite_error(tmp_path, good + "\n" + good, name="c.jsonl")
    assert "duplicate case id 'a' (line 1 and line 2)" in dup


def test_missing_and_unsupported_files(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="file not found"):
        load_suite(tmp_path / "nope.yaml")
    with pytest.raises(ConfigError, match=r"unsupported suite file type '\.txt'"):
        load_suite(write(tmp_path, "s.txt", "x"))


# --- models -------------------------------------------------------------------------------------

MODELS = """
models:
  - { label: local, base_url: "http://localhost:11434/v1/", model: m1 }
  - { label: hosted, base_url: "https://api.example.com/v1", model: m2,
      api_key_env: GRADY_TEST_KEY }
"""


def test_models_load_and_trailing_slash_is_trimmed(tmp_path: Path) -> None:
    models = load_models(write(tmp_path, "m.yaml", MODELS))
    assert models[0].base_url == "http://localhost:11434/v1"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "models:\n  - {label: a, base_url: 'http://x', model: m}\n"
            "  - {label: a, base_url: 'http://y', model: m}",
            "duplicate model label 'a'",
        ),
        ("models:\n  - {label: a, base_url: 'localhost:1', model: m}", "must start with http://"),
        ("models:\n  - {label: a, base_url: 'http://x', model: m, key: s}", "unknown key"),
        ("models: []", "models:"),
    ],
)
def test_models_errors(tmp_path: Path, text: str, expected: str) -> None:
    with pytest.raises(ConfigError) as info:
        load_models(write(tmp_path, "m.yaml", text))
    assert expected in str(info.value)


def test_select_models(tmp_path: Path) -> None:
    models = load_models(write(tmp_path, "m.yaml", MODELS))
    assert [m.label for m in select_models(models, [])] == ["local", "hosted"]
    assert [m.label for m in select_models(models, ["hosted", "local", "hosted"])] == [
        "hosted",
        "local",
    ]
    with pytest.raises(
        ConfigError, match=r"unknown model label\(s\): nope \(known: local, hosted\)"
    ):
        select_models(models, ["nope"])


def test_api_keys_resolve_from_env_and_fail_fast(tmp_path: Path) -> None:
    models = load_models(write(tmp_path, "m.yaml", MODELS))
    assert resolve_api_keys(models, {"GRADY_TEST_KEY": "sk-test"}) == {
        "local": None,
        "hosted": "sk-test",
    }
    with pytest.raises(ConfigError, match=r"needs an API key in \$GRADY_TEST_KEY"):
        resolve_api_keys(models, {})
    with pytest.raises(ConfigError):
        resolve_api_keys(models, {"GRADY_TEST_KEY": ""})
