"""Build the judge-check set: answer each rubric prompt with several models, for hand grading.

    uv run python -m grady.judgeset --models examples/models.local.yaml \\
        --model ollama-gemma3-4b --model ollama-gemma3-1b

Reads ``judge-check/cases.yaml`` and writes ``judge-check/candidates.jsonl``, one line per
prompt x model. ``--add`` keeps the existing answers (and their labels) and answers only the
missing pairs, for new prompts or a new model. Ids are numbered in a shuffled model order;
the ``source`` field names the model, so grade with ``grady label`` before opening the file.
Use models of different strength: a set where every answer passes can't show whether the
judge agrees with you.

Any failed request stops the run and nothing is written. A set with holes would quietly drop
exactly the prompts a model struggles with.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
from collections import Counter
from collections.abc import Sequence, Set
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from grady.client import ChatClient
from grady.config import (
    ID_PATTERN,
    ConfigError,
    ModelSpec,
    _read_yaml,
    format_validation_error,
    load_models,
    resolve_api_keys,
    select_models,
)
from grady.labeling import Candidate, read_jsonl
from grady.runner import warm_up

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_REQUEST_ERRORS = 3


class RubricCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=ID_PATTERN)
    prompt: str = Field(min_length=1)
    rubric: str = Field(min_length=1)


class RubricSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=1)
    cases: list[RubricCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> RubricSet:
        ids = [case.id for case in self.cases]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate case ids: {', '.join(dupes)}")
        return self


class RequestFailedError(Exception):
    pass


def load_rubric_set(path: Path) -> RubricSet:
    data = _read_yaml(path)
    try:
        return RubricSet.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"{path}: {format_validation_error(exc)}") from None


async def answer_all(
    rubrics: RubricSet,
    models: Sequence[ModelSpec],
    api_keys: dict[str, str | None],
    *,
    timeout: float,
    retries: int,
    have: Set[tuple[str, str]] = frozenset(),
) -> dict[tuple[str, str], str]:
    """Every (prompt, model) pair not in ``have``, one request at a time.

    Raises on the first failed request.
    """
    answers: dict[tuple[str, str], str] = {}
    async with ChatClient(timeout=timeout, retries=retries) as chat:
        for model in models:
            todo = [case for case in rubrics.cases if (case.id, model.label) not in have]
            if not todo:
                continue
            key = api_keys[model.label]
            warm = await warm_up(chat, model, key)
            if warm.error is not None:
                raise RequestFailedError(f"warm-up {model.label}: {warm.error}")
            for case in todo:
                completion = await chat.chat(
                    model,
                    [{"role": "user", "content": case.prompt}],
                    temperature=rubrics.temperature,
                    max_tokens=rubrics.max_tokens,
                    api_key=key,
                )
                if completion.error is not None or completion.text is None:
                    raise RequestFailedError(f"{model.label} on {case.id}: {completion.error}")
                answers[(case.id, model.label)] = completion.text
                _log(f"{model.label} {case.id}: {completion.latency_s or 0:.1f}s")
    return answers


def build_candidates(
    rubrics: RubricSet,
    answers: dict[tuple[str, str], str],
    rng: random.Random,
    existing: Sequence[Candidate] = (),
) -> list[Candidate]:
    """Number each prompt's new answers in a shuffled model order, after any existing ones."""
    taken = Counter(c.case_id for c in existing)
    candidates: list[Candidate] = []
    for case in rubrics.cases:
        sources = [source for case_id, source in answers if case_id == case.id]
        for n, source in enumerate(rng.sample(sources, k=len(sources)), start=1):
            candidates.append(
                Candidate(
                    id=f"{case.id}-{taken[case.id] + n}",
                    case_id=case.id,
                    prompt=case.prompt,
                    rubric=case.rubric,
                    output=answers[(case.id, source)],
                    source=source,
                )
            )
    return candidates


def check_existing(rubrics: RubricSet, existing: Sequence[Candidate], path: Path) -> None:
    """Answers kept by --add must still match their prompt and rubric."""
    cases = {case.id: case for case in rubrics.cases}
    for candidate in existing:
        case = cases.get(candidate.case_id)
        if case is not None and (case.prompt, case.rubric) != (candidate.prompt, candidate.rubric):
            raise ConfigError(
                f"{path}: case {case.id!r} has changed since its answers were made; restore it, "
                "or give the changed case a new id"
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        rubrics = load_rubric_set(args.cases)
        models = select_models(load_models(args.models), args.model or [])
        api_keys = resolve_api_keys(models, os.environ)
        existing: list[Candidate] = []
        if args.out.exists():
            if args.add:
                existing = read_jsonl(args.out, Candidate)
                check_existing(rubrics, existing, args.out)
            elif not args.force:
                raise ConfigError(
                    f"{args.out} already exists and labels are tied to its answers; pass --add "
                    "to answer only what's missing, or --force to replace it"
                )
    except ConfigError as exc:
        _log(f"error: {exc}")
        return EXIT_CONFIG
    if len(models) < 2:
        _log("note: with one model, each prompt gets one answer; two or more models give a mix")

    have = {(c.case_id, c.source) for c in existing}
    try:
        answers = asyncio.run(
            answer_all(
                rubrics, models, api_keys, timeout=args.timeout, retries=args.retries, have=have
            )
        )
    except RequestFailedError as exc:
        _log(f"error: {exc}. Nothing was written.")
        return EXIT_REQUEST_ERRORS
    if not answers:
        _log(f"Nothing to add: {args.out} has an answer from every model for every prompt.")
        return EXIT_OK

    new = build_candidates(rubrics, answers, random.Random(args.seed), existing)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text("".join(c.model_dump_json() + "\n" for c in [*existing, *new]), "utf-8")
    tmp.replace(args.out)
    _log(
        f"Wrote {len(new)} new candidates to {args.out} ({len(existing) + len(new)} in all). "
        "Grade them with `grady label` before opening the file: it names the model behind "
        "each answer."
    )
    return EXIT_OK


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m grady.judgeset", description=__doc__.split("\n\n")[0] if __doc__ else ""
    )
    parser.add_argument(
        "cases", nargs="?", type=Path, default=Path("judge-check/cases.yaml"), help="Rubric file."
    )
    parser.add_argument("--models", type=Path, required=True, help="Models file (YAML).")
    parser.add_argument(
        "--model", action="append", help="Model label to use. Repeatable. Default: every model."
    )
    parser.add_argument(
        "--out", type=Path, default=Path("judge-check/candidates.jsonl"), help="Output file."
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for id numbering.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Seconds per request.")
    parser.add_argument("--retries", type=int, default=2, help="Retries after the first try.")
    parser.add_argument(
        "--add", action="store_true", help="Keep existing answers; answer only what's missing."
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing output file.")
    return parser


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
