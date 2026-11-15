from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from grady.scorers.base import ScoreInput, ScoreResult, preview, text_arg


@dataclass(frozen=True)
class Exact:
    """Passes when the output equals the value after stripping surrounding whitespace."""

    name: ClassVar[str] = "exact"
    value: str
    ignore_case: bool = False

    @classmethod
    def from_arg(cls, arg: object) -> Exact:
        value, ignore_case = text_arg(cls.name, arg, allow_list=False)
        if not isinstance(value, str):  # text_arg only returns lists when allow_list=True
            raise TypeError("exact: expected a string value")
        return cls(value=value, ignore_case=ignore_case)

    async def score(self, item: ScoreInput) -> ScoreResult:
        got, want = item.output.strip(), self.value.strip()
        same = got.casefold() == want.casefold() if self.ignore_case else got == want
        if same:
            return ScoreResult(True, "matched")
        return ScoreResult(False, f"expected {preview(want)}, got {preview(got)}")
