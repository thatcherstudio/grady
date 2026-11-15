"""Records written by ``grady run``: one result per job, plus one metadata file per run."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel


def utc_now() -> str:
    """ISO-8601 UTC timestamp with milliseconds, e.g. ``2026-11-14T20:15:02.311Z``."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class ScoreRecord(BaseModel):
    scorer: str
    passed: bool
    detail: str


class ResultRecord(BaseModel):
    """One JSONL line: one case x model x repeat."""

    run_id: str
    suite: str
    case_id: str
    model: str
    repeat: int
    prompt: str
    output: str | None
    scores: list[ScoreRecord]
    passed: bool
    latency_s: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    tokens_per_s: float | None
    attempts: int
    error: str | None
    started_at: str


class ModelInfo(BaseModel):
    """What the metadata records about a model. Deliberately has no field for a key."""

    label: str
    base_url: str
    model: str


class RunSettings(BaseModel):
    concurrency: int
    timeout: float
    retries: int
    min_pass_rate: float
    warmup: bool


class WarmupRecord(BaseModel):
    """The untimed request sent to each model before the run, so load time isn't scored."""

    label: str
    latency_s: float | None
    attempts: int
    error: str | None


class Totals(BaseModel):
    jobs: int
    passed: int
    failed: int
    errors: int
    pass_rate: float

    @classmethod
    def from_records(cls, records: list[ResultRecord]) -> Totals:
        errors = sum(1 for r in records if r.error is not None)
        passed = sum(1 for r in records if r.passed)
        jobs = len(records)
        return cls(
            jobs=jobs,
            passed=passed,
            failed=jobs - passed - errors,
            errors=errors,
            pass_rate=passed / jobs if jobs else 0.0,
        )


class RunMeta(BaseModel):
    grady_version: str
    run_id: str
    suite: str
    suite_path: str
    suite_sha256: str
    models: list[ModelInfo]
    settings: RunSettings
    git_sha: str | None
    started_at: str
    warmup: list[WarmupRecord] = []
    finished_at: str | None = None
    totals: Totals | None = None
