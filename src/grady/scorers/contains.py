from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from grady.scorers.base import ScoreInput, ScorerArgError, ScoreResult, text_arg


@dataclass(frozen=True)
class Contains:
    """Passes when every given string appears in the output."""

    name: ClassVar[str] = "contains"
    values: tuple[str, ...]
    ignore_case: bool = False

    @classmethod
    def from_arg(cls, arg: object) -> Contains:
        value, ignore_case = text_arg(cls.name, arg, allow_list=True)
        values = (value,) if isinstance(value, str) else tuple(value)
        if any(v == "" for v in values):
            raise ScorerArgError("contains: empty string always matches; remove it")
        return cls(values=values, ignore_case=ignore_case)

    async def score(self, item: ScoreInput) -> ScoreResult:
        haystack = item.output.casefold() if self.ignore_case else item.output
        missing = [
            v for v in self.values if (v.casefold() if self.ignore_case else v) not in haystack
        ]
        if missing:
            return ScoreResult(False, "missing " + ", ".join(repr(v) for v in missing))
        return ScoreResult(True, "found " + ", ".join(repr(v) for v in self.values))
