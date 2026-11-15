"""Scorer registry. Add a scorer by writing a module with a ``from_arg`` class and listing it."""

from __future__ import annotations

from collections.abc import Callable

from grady.scorers.base import ScoreInput, Scorer, ScorerArgError, ScoreResult
from grady.scorers.contains import Contains
from grady.scorers.exact import Exact
from grady.scorers.regex import Regex

REGISTRY: dict[str, Callable[[object], Scorer]] = {
    Exact.name: Exact.from_arg,
    Contains.name: Contains.from_arg,
    Regex.name: Regex.from_arg,
}


def build_scorer(kind: str, arg: object) -> Scorer:
    """Build a scorer from a suite-file score entry, or raise :class:`ScorerArgError`."""
    factory = REGISTRY.get(kind)
    if factory is None:
        known = ", ".join(sorted(REGISTRY))
        raise ScorerArgError(f"unknown scorer {kind!r} (known: {known})")
    return factory(arg)


__all__ = ["REGISTRY", "ScoreInput", "ScoreResult", "Scorer", "ScorerArgError", "build_scorer"]
