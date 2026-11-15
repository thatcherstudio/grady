"""Run a suite: expand jobs, send them under a concurrency limit, score, and write results.

Each record is appended and flushed the moment its job finishes, so a crash or Ctrl-C keeps
everything completed so far. Nothing fails silently: a request that still fails after
retries, or a scorer that raises, becomes a record that says so.
"""

from __future__ import annotations

import asyncio
import hashlib
import secrets
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

from grady.client import ChatClient, Message
from grady.config import Case, ModelSpec, Suite
from grady.progress import Outcome, RunProgress
from grady.records import ResultRecord, ScoreRecord, WarmupRecord, utc_now
from grady.scorers import ScoreInput


@dataclass(frozen=True)
class Job:
    case: Case
    model: ModelSpec
    repeat: int = 0


def new_run_id() -> str:
    return f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


def build_jobs(suite: Suite, models: Sequence[ModelSpec]) -> list[Job]:
    # Repeats arrive in M2; the record already carries a repeat index so the schema won't change.
    return [Job(case=case, model=model) for model in models for case in suite.cases]


def build_messages(suite: Suite, case: Case) -> list[Message]:
    messages: list[Message] = []
    system = suite.system_for(case)
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": case.prompt})
    return messages


async def execute_job(
    job: Job, suite: Suite, client: ChatClient, api_key: str | None, run_id: str
) -> ResultRecord:
    started_at = utc_now()
    completion = await client.chat(
        job.model,
        build_messages(suite, job.case),
        temperature=suite.temperature_for(job.case),
        max_tokens=suite.max_tokens_for(job.case),
        api_key=api_key,
    )
    scores: list[ScoreRecord] = []
    if completion.error is None and completion.text is not None:
        item = ScoreInput(prompt=job.case.prompt, output=completion.text)
        for spec in job.case.score:
            try:
                result = await spec.build().score(item)
                scores.append(
                    ScoreRecord(scorer=spec.kind, passed=result.passed, detail=result.detail)
                )
            except Exception as exc:  # a scorer bug must not sink the run or pass silently
                scores.append(
                    ScoreRecord(
                        scorer=spec.kind,
                        passed=False,
                        detail=f"scorer error: {type(exc).__name__}: {exc}",
                    )
                )

    tokens_per_s = None
    if completion.completion_tokens is not None and completion.latency_s:
        tokens_per_s = round(completion.completion_tokens / completion.latency_s, 2)

    return ResultRecord(
        run_id=run_id,
        suite=suite.suite,
        case_id=job.case.id,
        model=job.model.label,
        repeat=job.repeat,
        prompt=job.case.prompt,
        output=completion.text,
        scores=scores,
        passed=completion.error is None and bool(scores) and all(s.passed for s in scores),
        latency_s=round(completion.latency_s, 3) if completion.latency_s is not None else None,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
        tokens_per_s=tokens_per_s,
        attempts=completion.attempts,
        error=completion.error,
        started_at=started_at,
    )


WARMUP_MESSAGES: list[Message] = [{"role": "user", "content": "Reply with OK."}]


async def warm_up(client: ChatClient, model: ModelSpec, api_key: str | None) -> WarmupRecord:
    """Send one tiny untimed request so a cold local server loads the model before timing.

    The answer is not scored or written to results. A failure is reported, not fatal: the
    jobs still run and each records its own error.
    """
    completion = await client.chat(
        model, WARMUP_MESSAGES, temperature=0.0, max_tokens=1, api_key=api_key
    )
    latency = completion.latency_s
    return WarmupRecord(
        label=model.label,
        latency_s=round(latency, 3) if latency is not None else None,
        attempts=completion.attempts,
        error=completion.error,
    )


def outcome_of(record: ResultRecord) -> Outcome:
    if record.error is not None:
        return "error"
    return "pass" if record.passed else "fail"


async def run_jobs(
    jobs: Sequence[Job],
    suite: Suite,
    client: ChatClient,
    api_keys: dict[str, str | None],
    run_id: str,
    out: TextIO,
    progress: RunProgress,
    concurrency: int,
) -> list[ResultRecord]:
    """Run every job with at most ``concurrency`` requests in flight. Returns all records."""
    semaphore = asyncio.Semaphore(concurrency)
    records: list[ResultRecord] = []

    async def run_one(job: Job) -> None:
        async with semaphore:
            try:
                record = await execute_job(job, suite, client, api_keys[job.model.label], run_id)
            except Exception as exc:  # recorded and counted as an error, never swallowed
                record = _internal_error_record(job, suite, run_id, exc)
        out.write(record.model_dump_json() + "\n")
        out.flush()
        records.append(record)
        progress.record(outcome_of(record))

    await asyncio.gather(*(run_one(job) for job in jobs))
    return records


def _internal_error_record(job: Job, suite: Suite, run_id: str, exc: Exception) -> ResultRecord:
    return ResultRecord(
        run_id=run_id,
        suite=suite.suite,
        case_id=job.case.id,
        model=job.model.label,
        repeat=job.repeat,
        prompt=job.case.prompt,
        output=None,
        scores=[],
        passed=False,
        latency_s=None,
        prompt_tokens=None,
        completion_tokens=None,
        tokens_per_s=None,
        attempts=0,
        error=f"internal error: {type(exc).__name__}: {exc}",
        started_at=utc_now(),
    )


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_sha() -> str | None:
    """HEAD of the working directory's git repo, or None outside one."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = result.stdout.strip()
    return sha if result.returncode == 0 and sha else None
