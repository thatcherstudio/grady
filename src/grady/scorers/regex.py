from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

from grady.scorers.base import ScoreInput, ScorerArgError, ScoreResult, preview


@dataclass(frozen=True)
class Regex:
    """Passes when ``re.search(pattern, output)`` finds a match. Use inline flags like ``(?i)``."""

    name: ClassVar[str] = "regex"
    pattern: re.Pattern[str]

    @classmethod
    def from_arg(cls, arg: object) -> Regex:
        if not isinstance(arg, str):
            raise ScorerArgError(
                f"regex: expected a pattern string, got {type(arg).__name__} {arg!r}"
            )
        try:
            return cls(pattern=re.compile(arg))
        except re.error as exc:
            raise ScorerArgError(f"regex: invalid pattern {arg!r}: {exc}") from None

    async def score(self, item: ScoreInput) -> ScoreResult:
        match = self.pattern.search(item.output)
        if match:
            return ScoreResult(True, f"matched {preview(match.group(0))}")
        return ScoreResult(
            False, f"no match for /{self.pattern.pattern}/ in {preview(item.output)}"
        )
