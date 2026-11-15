from __future__ import annotations

import io

from rich.console import Console

from grady.progress import RunProgress, format_duration, format_line, format_rate


def test_line_matches_the_spec_example() -> None:
    # 37 done at 2.4 cases/s -> 83 left -> ~35 s
    assert format_line(37, 120, 29, 8, 0, elapsed=37 / 2.4) == (
        "[ 37/120 ] 31% | 2.4 cases/s | pass 29 fail 8 | ETA 0:35"
    )


def test_errors_are_shown_only_when_present() -> None:
    line = format_line(10, 20, 5, 3, 2, elapsed=5.0)
    assert "| pass 5 fail 3 err 2 |" in line


def test_start_of_run() -> None:
    assert format_line(0, 120, 0, 0, 0, elapsed=0.0) == (
        "[  0/120 ] 0% | -- cases/s | pass 0 fail 0 | ETA --:--"
    )


def test_slow_rates_switch_to_per_minute() -> None:
    assert format_rate(0.5) == "30.0 cases/min"
    assert format_rate(3.0) == "3.0 cases/s"


def test_durations() -> None:
    assert format_duration(35) == "0:35"
    assert format_duration(3723) == "1:02:03"


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_non_terminal_output_is_throttled() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=200)
    clock = FakeClock()
    with RunProgress(total=4, console=console, interval=10.0, clock=clock) as progress:
        progress.record("pass")  # within the interval: no line
        progress.record("fail")
        clock.now += 11
        progress.record("error")  # interval elapsed: one line
        progress.record("pass")
    lines = buf.getvalue().splitlines()
    assert lines == [
        "[0/4 ] 0% | -- cases/s | pass 0 fail 0 | ETA --:--",
        "[3/4 ] 75% | 16.4 cases/min | pass 1 fail 1 err 1 | ETA 0:04",
        "[4/4 ] 100% | 21.8 cases/min | pass 2 fail 1 err 1 | ETA 0:00",
    ]
