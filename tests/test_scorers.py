from __future__ import annotations

import pytest

from grady.scorers import REGISTRY, ScoreInput, ScorerArgError, build_scorer

pytestmark = pytest.mark.anyio


async def score(kind: str, arg: object, output: str) -> tuple[bool, str]:
    result = await build_scorer(kind, arg).score(ScoreInput(prompt="p", output=output))
    return result.passed, result.detail


# --- pass / fail fixtures for every scorer -----------------------------------------------------

CASES = [
    # (kind, arg, output, expected_pass)
    ("exact", "42", "42", True),
    ("exact", "42", "  42\n", True),  # surrounding whitespace is ignored
    ("exact", "42", "42.", False),
    ("exact", "Paris", "paris", False),  # case-sensitive by default
    ("exact", {"value": "Paris", "ignore_case": True}, "PARIS", True),
    ("exact", "42", "", False),
    ("contains", "1250", "Total: $1250.00", True),
    ("contains", "1250", "Total: $1,250.00", False),
    ("contains", ["red", "green", "blue"], "red, green, blue", True),
    ("contains", ["red", "green", "blue"], "red, green", False),
    ("contains", {"value": "jupiter", "ignore_case": True}, "Jupiter is largest.", True),
    ("contains", "jupiter", "Jupiter", False),
    ("contains", "x", "", False),
    ("regex", r"^\s*2027-03-03\s*$", "2027-03-03\n", True),
    ("regex", r"^\s*2027-03-03\s*$", "The date is 2027-03-03", False),
    ("regex", r"(?i)\bok\b", "Ok!", True),
    ("regex", r"\d+", "", False),
]


@pytest.mark.parametrize(("kind", "arg", "output", "expected"), CASES)
async def test_pass_and_fail(kind: str, arg: object, output: str, expected: bool) -> None:
    passed, detail = await score(kind, arg, output)
    assert passed is expected
    assert detail


def test_every_registered_scorer_has_fixtures() -> None:
    assert {kind for kind, *_ in CASES} == set(REGISTRY)


# --- details say why ---------------------------------------------------------------------------


async def test_exact_detail_shows_expected_and_got() -> None:
    _, detail = await score("exact", "42", "forty-two")
    assert detail == "expected '42', got 'forty-two'"


async def test_exact_ignore_case_detail_keeps_original_text() -> None:
    _, detail = await score("exact", {"value": "Paris", "ignore_case": True}, "Paris.")
    assert detail == "expected 'Paris', got 'Paris.'"


async def test_contains_detail_lists_only_missing() -> None:
    _, detail = await score("contains", ["red", "green", "blue"], "red only")
    assert detail == "missing 'green', 'blue'"


async def test_regex_detail_shows_match() -> None:
    passed, detail = await score("regex", r"\d{4}", "year 2027 ok")
    assert passed
    assert detail == "matched '2027'"


async def test_long_output_is_truncated_in_detail() -> None:
    _, detail = await score("exact", "short", "word " * 100)
    assert len(detail) < 120
    assert detail.endswith("…'")


# --- argument validation (happens at load time) ------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "arg", "message"),
    [
        ("containz", "x", "unknown scorer 'containz' (known: contains, exact, regex)"),
        ("exact", 42, 'expected a string; got int 42 (quote it: "42")'),
        ("exact", ["a"], "expected a string; got list"),
        ("exact", {"ignore_case": True}, "missing 'value'"),
        ("exact", {"value": "a", "ignorecase": True}, "unknown option(s) ignorecase"),
        ("exact", {"value": "a", "ignore_case": "yes"}, "ignore_case must be true or false"),
        ("contains", 1250.0, "(quote it"),
        ("contains", [], "list must not be empty"),
        ("contains", ["a", 1], "got int 1"),
        ("contains", "", "empty string always matches"),
        ("regex", "(unclosed", "invalid pattern '(unclosed'"),
        ("regex", 5, "expected a pattern string"),
    ],
)
def test_bad_arguments_are_rejected(kind: str, arg: object, message: str) -> None:
    with pytest.raises(ScorerArgError) as info:
        build_scorer(kind, arg)
    assert message in str(info.value)
