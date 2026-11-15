"""The scorer interface shared by every scorer.

Scorers are async so model-graded scorers (``llm_judge``, M2) fit the same interface as the
string scorers. Each scorer is built from its suite-file argument with ``from_arg``, which
raises :class:`ScorerArgError` for a bad argument. The loader calls it, so a bad suite fails
before any request is sent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol


@dataclass(frozen=True)
class ScoreInput:
    """What a scorer sees for one job."""

    prompt: str
    output: str


@dataclass(frozen=True)
class ScoreResult:
    passed: bool
    detail: str


class Scorer(Protocol):
    name: ClassVar[str]

    async def score(self, item: ScoreInput) -> ScoreResult: ...


class ScorerArgError(ValueError):
    """A scorer's argument in the suite file is invalid."""


def preview(text: str, limit: int = 80) -> str:
    """One-line, length-capped repr of ``text`` for score details."""
    one_line = " ".join(text.split())
    if len(one_line) > limit:
        one_line = one_line[: limit - 1] + "…"
    return repr(one_line)


def text_arg(scorer: str, arg: object, *, allow_list: bool) -> tuple[str | list[str], bool]:
    """Parse the ``"text"`` / ``[...]`` / ``{value: ..., ignore_case: ...}`` argument forms.

    Returns ``(value, ignore_case)``. Non-string values are rejected rather than coerced, so an
    unquoted YAML number like ``contains: 1250.00`` fails loudly instead of matching "1250.0".
    """
    ignore_case = False
    value: object = arg
    if isinstance(arg, dict):
        unknown = set(arg) - {"value", "ignore_case"}
        if unknown:
            raise ScorerArgError(
                f"{scorer}: unknown option(s) {', '.join(sorted(map(str, unknown)))} "
                "(allowed: value, ignore_case)"
            )
        if "value" not in arg:
            raise ScorerArgError(f"{scorer}: missing 'value'")
        value = arg["value"]
        flag = arg.get("ignore_case", False)
        if not isinstance(flag, bool):
            raise ScorerArgError(f"{scorer}: ignore_case must be true or false")
        ignore_case = flag

    if isinstance(value, str):
        return value, ignore_case
    if allow_list and isinstance(value, list):
        if not value:
            raise ScorerArgError(f"{scorer}: list must not be empty")
        for item in value:
            if not isinstance(item, str):
                raise ScorerArgError(f"{scorer}: {_not_a_string(item)}")
        return list(value), ignore_case
    expected = "a string or a list of strings" if allow_list else "a string"
    raise ScorerArgError(f"{scorer}: expected {expected}; {_not_a_string(value)}")


def _not_a_string(value: object) -> str:
    hint = f' (quote it: "{value}")' if isinstance(value, int | float) else ""
    return f"got {type(value).__name__} {value!r}{hint}"
