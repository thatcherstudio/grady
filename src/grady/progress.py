"""The run's progress line.

``[ 37/120 ] 31% | 2.4 cases/s | pass 29 fail 8 | ETA 0:35``

In a terminal it redraws in place at most 4 times a second. When stderr is not a terminal
(CI logs, ``| tee``, background jobs) a live display would stay silent until the end, so it
prints a plain line at most every ``interval`` seconds instead, plus one at the start and one
at the end. Either way a long run is never silent.

Adapted from the maintainer's earlier project, civicbench (``civicbench/progress.py``).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from types import TracebackType
from typing import Literal

from rich.console import Console
from rich.live import Live
from rich.text import Text

Outcome = Literal["pass", "fail", "error"]


def format_duration(seconds: float) -> str:
    """``m:ss`` under an hour, ``h:mm:ss`` beyond."""
    total = max(0, round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_rate(per_second: float, unit: str = "cases") -> str:
    if per_second <= 0:
        return f"-- {unit}/s"
    if per_second >= 1:
        return f"{per_second:.1f} {unit}/s"
    return f"{per_second * 60:.1f} {unit}/min"


def format_line(
    done: int, total: int, passed: int, failed: int, errors: int, elapsed: float
) -> str:
    width = len(str(total))
    pct = 100.0 * done / total if total else 100.0
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = format_duration((total - done) / rate) if rate > 0 else "--:--"
    counts = f"pass {passed} fail {failed}"
    if errors:
        counts += f" err {errors}"
    return f"[{done:>{width}}/{total} ] {pct:.0f}% | {format_rate(rate)} | {counts} | ETA {eta}"


class RunProgress:
    """Counts outcomes and keeps the progress line on screen. Use as a context manager."""

    def __init__(
        self,
        total: int,
        console: Console | None = None,
        interval: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.total = total
        self.console = console or Console(stderr=True)
        self.interval = interval
        self.clock = clock
        self.passed = self.failed = self.errors = 0
        self._start = clock()
        self._last_print = float("-inf")
        self._live: Live | None = None

    @property
    def done(self) -> int:
        return self.passed + self.failed + self.errors

    def render(self) -> str:
        elapsed = self.clock() - self._start
        return format_line(self.done, self.total, self.passed, self.failed, self.errors, elapsed)

    def __rich__(self) -> Text:
        # Live calls this on every refresh, so the rate and ETA keep moving between results.
        return Text(self.render())

    def record(self, outcome: Outcome) -> None:
        if outcome == "pass":
            self.passed += 1
        elif outcome == "fail":
            self.failed += 1
        else:
            self.errors += 1
        if self._live is None:
            self._emit()

    def _emit(self, force: bool = False) -> None:
        now = self.clock()
        if not force and now - self._last_print < self.interval:
            return
        self._last_print = now
        self.console.print(self.render(), markup=False, highlight=False, soft_wrap=True)

    def __enter__(self) -> RunProgress:
        self._start = self.clock()
        if self.console.is_terminal:
            self._live = Live(self, console=self.console, refresh_per_second=4)
            self._live.start()
        else:
            self._emit(force=True)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._live is not None:
            self._live.stop()  # leaves the final line on screen
            self._live = None
        else:
            self._emit(force=True)
