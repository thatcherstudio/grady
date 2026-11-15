"""Hand labels for judge-check: the answers to grade, your labels, and the grading loop.

A candidate is one model's answer to one rubric prompt (``python -m grady.judgeset`` writes
them). ``grady label`` shows candidates one card at a time and saves after every key. Grading
is blind: the model behind an answer is never shown, and two answers to the same prompt are
spread apart, so each is graded against the rubric rather than against the other. M4's
``grady judge-check`` reads the labels file and reports how often the LLM judge agrees.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from grady.config import ID_PATTERN, ConfigError, _read_text, format_validation_error
from grady.records import utc_now

Verdict = Literal["pass", "fail"]
Outcome = Literal["done", "quit"]

# Below this share for the rarer verdict, agreement mostly measures the easy calls.
MIN_MINORITY_SHARE = 0.25

PASS_KEYS = frozenset({"\x1b[C", "\x1bOC", "p"})  # right arrow, or p
FAIL_KEYS = frozenset({"\x1b[D", "\x1bOD", "f"})  # left arrow, or f
KEY_HELP = "← fail   → pass   ? unsure   n note   u undo   q quit"


class Candidate(BaseModel):
    """One answer to grade. ``source`` names the model that wrote it and is never shown."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=ID_PATTERN)
    case_id: str = Field(pattern=ID_PATTERN)
    prompt: str = Field(min_length=1)
    rubric: str = Field(min_length=1)
    output: str
    source: str


class Label(Candidate):
    """A candidate plus your verdict. Self-contained, so judge-check needs only this file."""

    label: Verdict
    unsure: bool = False
    note: str | None = None
    labeled_at: str

    def candidate(self) -> Candidate:
        return Candidate.model_validate(self.model_dump(include=set(Candidate.model_fields)))


C = TypeVar("C", bound=Candidate)


def read_jsonl(path: Path, model: type[C]) -> list[C]:
    """Parse one record per line. Errors name the file and line; ids must be unique."""
    items: list[C] = []
    seen: dict[str, int] = {}
    for lineno, line in enumerate(_read_text(path).splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path}:{lineno}: invalid JSON: {exc.msg}") from None
        try:
            item = model.model_validate(raw)
        except ValidationError as exc:
            raise ConfigError(f"{path}:{lineno}: {format_validation_error(exc)}") from None
        if item.id in seen:
            raise ConfigError(
                f"{path}: duplicate id {item.id!r} (lines {seen[item.id]} and {lineno})"
            )
        seen[item.id] = lineno
        items.append(item)
    return items


def save_labels(path: Path, labels: Sequence[Label]) -> None:
    """Write the whole file through a temp file and a rename, so a crash can't truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(label.model_dump_json() + "\n" for label in labels), encoding="utf-8")
    tmp.replace(path)


def grading_order(candidates: Sequence[Candidate], rng: random.Random) -> list[Candidate]:
    """Shuffle, keeping answers to the same prompt as far apart as possible.

    Each step takes a random answer from the prompt with the most answers left, skipping the
    prompt just shown. With two answers per prompt, every prompt appears once before any
    appears twice.
    """
    by_case: dict[str, list[Candidate]] = {}
    for candidate in rng.sample(list(candidates), k=len(candidates)):
        by_case.setdefault(candidate.case_id, []).append(candidate)
    ordered: list[Candidate] = []
    previous: str | None = None
    while any(by_case.values()):
        left = [case for case, items in by_case.items() if items]
        choices = [case for case in left if case != previous] or left
        most = max(len(by_case[case]) for case in choices)
        case = rng.choice([c for c in choices if len(by_case[c]) == most])
        ordered.append(by_case[case].pop())
        previous = case
    return ordered


@dataclass(frozen=True)
class LabelStats:
    total: int
    labeled: int
    passed: int
    failed: int
    unsure: int

    def describe(self) -> str:
        return (
            f"{self.labeled}/{self.total} labeled: {self.passed} pass, {self.failed} fail, "
            f"{self.unsure} unsure"
        )

    def balance_warning(self) -> str | None:
        """Only for a finished set: a lopsided one can't show whether the judge agrees."""
        if not self.labeled or self.labeled < self.total:
            return None
        minority, verdict = min((self.passed, "pass"), (self.failed, "fail"))
        if minority / self.labeled >= MIN_MINORITY_SHARE:
            return None
        return (
            f"Only {minority} of {self.labeled} labels are {verdict}. Agreement on a set this "
            f"lopsided mostly measures the easy calls, so add answers that come out {verdict}."
        )


@dataclass
class LabelSession:
    """Grading state: what's left, what's done, and the flags on the current card."""

    candidates: list[Candidate]  # file order, which is also the save order
    labels: dict[str, Label]
    queue: list[Candidate]  # still to grade, in grading order
    history: list[str]  # labeled ids, oldest first, so undo works across sessions
    orphans: list[Label]  # labels whose candidate is gone: kept in the file, never shown
    unsure: bool = False
    note: str | None = None

    @property
    def current(self) -> Candidate | None:
        return self.queue[0] if self.queue else None

    def mark(self, verdict: Verdict, labeled_at: str) -> Label:
        candidate = self.queue.pop(0)
        label = Label.model_validate(
            {
                **candidate.model_dump(),
                "label": verdict,
                "unsure": self.unsure,
                "note": self.note,
                "labeled_at": labeled_at,
            }
        )
        self.labels[candidate.id] = label
        self.history.append(candidate.id)
        self.unsure, self.note = False, None
        return label

    def undo(self) -> Label | None:
        """Take back the most recent label and show that card again, flags and note intact."""
        if not self.history:
            return None
        label = self.labels.pop(self.history.pop())
        self.queue.insert(0, label.candidate())
        self.unsure, self.note = label.unsure, label.note
        return label

    def ordered_labels(self) -> list[Label]:
        labeled = [self.labels[c.id] for c in self.candidates if c.id in self.labels]
        return labeled + self.orphans

    def stats(self) -> LabelStats:
        labels = list(self.labels.values())
        return LabelStats(
            total=len(self.candidates),
            labeled=len(labels),
            passed=sum(label.label == "pass" for label in labels),
            failed=sum(label.label == "fail" for label in labels),
            unsure=sum(label.unsure for label in labels),
        )


def load_session(candidates_path: Path, labels_path: Path, rng: random.Random) -> LabelSession:
    """Start or resume. A label given to a different answer than the current one is an error."""
    candidates = read_jsonl(candidates_path, Candidate)
    if not candidates:
        raise ConfigError(f"{candidates_path}: no candidates found")
    existing = read_jsonl(labels_path, Label) if labels_path.exists() else []

    by_id = {c.id: c for c in candidates}
    labels: dict[str, Label] = {}
    orphans: list[Label] = []
    for label in existing:
        candidate = by_id.get(label.id)
        if candidate is None:
            orphans.append(label)
        elif label.candidate() != candidate:
            raise ConfigError(
                f"{labels_path}: the label for {label.id!r} was given to a different answer "
                f"than the one now in {candidates_path}. Restore the old candidates, or move "
                f"{labels_path} aside to start over."
            )
        else:
            labels[label.id] = label

    history = [label.id for label in sorted(labels.values(), key=lambda lab: lab.labeled_at)]
    queue = grading_order([c for c in candidates if c.id not in labels], rng)
    return LabelSession(
        candidates=candidates, labels=labels, queue=queue, history=history, orphans=orphans
    )


def normalize_key(raw: str) -> str:
    """Arrow keys arrive as escape sequences and stay whole. Otherwise only the first key
    counts, so keys typed while a card was drawing can't grade a card you haven't seen."""
    return raw if raw.startswith("\x1b") else raw[:1].lower()


def grade(
    session: LabelSession,
    *,
    console: Console,
    read_key: Callable[[], str],
    read_note: Callable[[str | None], str],
    save: Callable[[], None],
    clock: Callable[[], str] = utc_now,
) -> Outcome:
    """Show cards until every candidate is labeled or you quit. Saves after each change."""
    status = ""
    while (candidate := session.current) is not None:
        console.clear()
        console.print(render_card(session, candidate, status))
        status = ""
        try:
            key = normalize_key(read_key())
        except EOFError:
            return "quit"
        if not key or key == "q":
            return "quit"
        if key in PASS_KEYS or key in FAIL_KEYS:
            verdict: Verdict = "pass" if key in PASS_KEYS else "fail"
            session.mark(verdict, clock())
            save()
            status = f"Saved: {verdict}."
        elif key == "?":
            session.unsure = not session.unsure
        elif key == "n":
            session.note = read_note(session.note).strip() or None
        elif key == "u":
            undone = session.undo()
            if undone is None:
                status = "Nothing to undo."
            else:
                save()
                status = f"Undid {undone.label}. Grade this one again."
        else:
            status = f"Unknown key. Keys: {KEY_HELP}"
    return "done"


def render_card(session: LabelSession, candidate: Candidate, status: str) -> RenderableType:
    """The rubric, prompt and answer, exactly as the judge will see them. Never the source."""
    done = len(session.labels)
    total = len(session.candidates)
    header = Table.grid(padding=(0, 2))
    header.add_row(
        Text(f"{done + 1}/{total}", style="bold"),
        ProgressBar(total=total, completed=done, width=30),
        Text(candidate.id, style="dim"),
    )

    body = Table.grid(padding=(0, 2))
    body.add_column(style="bold", no_wrap=True)
    body.add_column()
    body.add_row("RUBRIC", Text(candidate.rubric))
    body.add_row("", "")
    body.add_row("PROMPT", Text(candidate.prompt))
    body.add_row("", "")
    answer = (
        Text(candidate.output)
        if candidate.output.strip()
        else Text("(empty answer)", style="dim italic")
    )
    body.add_row("ANSWER", Panel(answer, border_style="dim"))

    flags = Text()
    if session.unsure:
        flags.append("unsure ", style="bold yellow")
    if session.note:
        flags.append(f"note: {session.note}", style="cyan")

    parts: list[RenderableType] = [header, "", body, ""]
    if flags:
        parts.append(flags)
    if status:
        parts.append(Text(status, style="green" if status.startswith("Saved") else "yellow"))
    parts.append(Text(KEY_HELP, style="dim"))
    return Group(*parts)
