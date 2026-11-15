"""Command-line interface. M1 ships ``grady run``; report, compare and judge-check follow."""

from __future__ import annotations

import asyncio
import os
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from grady import __version__
from grady.client import ChatClient
from grady.config import (
    ConfigError,
    ModelSpec,
    Suite,
    load_models,
    load_suite,
    resolve_api_keys,
    select_models,
)
from grady.progress import RunProgress
from grady.records import ModelInfo, ResultRecord, RunMeta, RunSettings, Totals, utc_now
from grady.runner import Job, build_jobs, git_sha, new_run_id, run_jobs, sha256_of

EXIT_OK = 0
EXIT_BELOW_THRESHOLD = 1
EXIT_CONFIG = 2
EXIT_REQUEST_ERRORS = 3
EXIT_INTERRUPTED = 130

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Run a prompt suite against OpenAI-compatible endpoints and grade the answers.",
)


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"grady {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_show_version, is_eager=True, help="Show the version and exit."
        ),
    ] = False,
) -> None:
    """Grady: grade model answers, and see quality next to speed."""


@app.command()
def run(
    suite_path: Annotated[
        Path, typer.Argument(metavar="SUITE", help="Suite file: .yaml, .yml or .jsonl.")
    ],
    models_path: Annotated[Path, typer.Option("--models", help="Models file (YAML).")],
    model_labels: Annotated[
        list[str] | None,
        typer.Option("--model", help="Model label to run. Repeatable. Default: every model."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(help="Results file. Default: results/<suite>-<timestamp>.jsonl."),
    ] = None,
    concurrency: Annotated[
        int, typer.Option(min=1, help="Max requests in flight across all models.")
    ] = 4,
    timeout: Annotated[float, typer.Option(min=0.1, help="Seconds per request.")] = 60.0,
    retries: Annotated[int, typer.Option(min=0, help="Retries after the first attempt.")] = 3,
    min_pass_rate: Annotated[
        float, typer.Option(min=0.0, max=1.0, help="Exit 1 when the pass rate is below this.")
    ] = 0.0,
) -> None:
    """Run SUITE against one or more models and write one JSONL record per case x model."""
    err = Console(stderr=True, soft_wrap=True)
    stdout = Console(soft_wrap=True)
    try:
        suite = load_suite(suite_path)
        models = select_models(load_models(models_path), model_labels or [])
        api_keys = resolve_api_keys(models, os.environ)
        run_id = new_run_id()
        out_path = out or Path("results") / f"{suite.suite}-{run_id.rsplit('-', 1)[0]}.jsonl"
        if out_path.exists():
            raise ConfigError(f"{out_path} already exists; pass a different --out")
    except ConfigError as exc:
        err.print(f"error: {exc}", markup=False, highlight=False)
        raise typer.Exit(EXIT_CONFIG) from None

    settings = RunSettings(
        concurrency=concurrency, timeout=timeout, retries=retries, min_pass_rate=min_pass_rate
    )
    jobs = build_jobs(suite, models)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_suffix(".meta.json")
    meta = RunMeta(
        grady_version=__version__,
        run_id=run_id,
        suite=suite.suite,
        suite_path=str(suite_path),
        suite_sha256=sha256_of(suite_path),
        models=[ModelInfo(label=m.label, base_url=m.base_url, model=m.model) for m in models],
        settings=settings,
        git_sha=git_sha(),
        started_at=utc_now(),
    )
    _write_meta(meta_path, meta)

    err.print(
        f"Running {len(jobs)} jobs ({len(suite.cases)} cases x {len(models)} models), "
        f"concurrency {concurrency} -> {out_path}",
        markup=False,
        highlight=False,
    )
    try:
        records = asyncio.run(_execute(jobs, suite, models, api_keys, settings, run_id, out_path))
    except KeyboardInterrupt:
        err.print(f"Interrupted. Completed results are in {out_path}", markup=False)
        raise typer.Exit(EXIT_INTERRUPTED) from None

    totals = Totals.from_records(records)
    meta.finished_at = utc_now()
    meta.totals = totals
    _write_meta(meta_path, meta)

    stdout.print(_summary_table(suite, run_id, models, records))
    _print_errors(err, records)
    stdout.print(f"Results:  {out_path}\nMetadata: {meta_path}", markup=False, highlight=False)

    code = exit_code_for(totals, min_pass_rate)
    if code == EXIT_REQUEST_ERRORS:
        err.print(
            f"{totals.errors} request(s) failed after retries; results are incomplete.",
            markup=False,
        )
    elif code == EXIT_BELOW_THRESHOLD:
        err.print(
            f"Pass rate {totals.pass_rate:.1%} is below --min-pass-rate {min_pass_rate:.1%}.",
            markup=False,
        )
    raise typer.Exit(code)


def exit_code_for(totals: Totals, min_pass_rate: float) -> int:
    """Request errors outrank the threshold: an incomplete run can't vouch for its pass rate."""
    if totals.errors:
        return EXIT_REQUEST_ERRORS
    if totals.pass_rate < min_pass_rate:
        return EXIT_BELOW_THRESHOLD
    return EXIT_OK


async def _execute(
    jobs: Sequence[Job],
    suite: Suite,
    models: Sequence[ModelSpec],
    api_keys: dict[str, str | None],
    settings: RunSettings,
    run_id: str,
    out_path: Path,
) -> list[ResultRecord]:
    async with ChatClient(timeout=settings.timeout, retries=settings.retries) as client:
        with (
            out_path.open("x", encoding="utf-8") as out,
            RunProgress(total=len(jobs)) as progress,
        ):
            return await run_jobs(
                jobs, suite, client, api_keys, run_id, out, progress, settings.concurrency
            )


def _write_meta(path: Path, meta: RunMeta) -> None:
    path.write_text(meta.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _summary_table(
    suite: Suite, run_id: str, models: Sequence[ModelSpec], records: Sequence[ResultRecord]
) -> Table:
    table = Table(
        title=f"{suite.suite} · run {run_id}",
        title_justify="left",
        caption="Latency and tok/s are medians, end to end (includes queueing and prompt).",
        caption_justify="left",
    )
    table.add_column("Model", no_wrap=True)
    for name in ("Cases", "Pass", "Fail", "Err", "Pass %", "Latency", "Tok/s"):
        table.add_column(name, justify="right", no_wrap=True)
    for model in models:
        rows = [r for r in records if r.model == model.label]
        totals = Totals.from_records(rows)
        table.add_row(
            model.label,
            str(totals.jobs),
            str(totals.passed),
            str(totals.failed),
            str(totals.errors),
            f"{totals.pass_rate:.0%}",
            _median([r.latency_s for r in rows], "{:.2f}s"),
            _median([r.tokens_per_s for r in rows], "{:.1f}"),
        )
    return table


def _median(values: Sequence[float | None], fmt: str) -> str:
    present = [v for v in values if v is not None]
    return fmt.format(statistics.median(present)) if present else "-"


def _print_errors(err: Console, records: Sequence[ResultRecord], limit: int = 3) -> None:
    """Show a few distinct request errors so a broken endpoint is diagnosable from the terminal."""
    seen: dict[tuple[str, str], int] = {}
    for record in records:
        if record.error is not None:
            key = (record.model, record.error)
            seen[key] = seen.get(key, 0) + 1
    for (model, error), count in list(seen.items())[:limit]:
        err.print(f"{model}: {count}x {error}", markup=False, highlight=False)
    if len(seen) > limit:
        err.print(f"... and {len(seen) - limit} more distinct errors", markup=False)
