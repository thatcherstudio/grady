"""The labeling session, its files, and `grady label` end to end. Keys come from test input."""

from __future__ import annotations

import io
import itertools
import json
import random
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from grady.cli import app
from grady.config import ConfigError
from grady.labeling import (
    Candidate,
    Label,
    LabelSession,
    LabelStats,
    grade,
    grading_order,
    load_session,
    normalize_key,
    read_jsonl,
    save_labels,
)

SECRET = "secret-model-x"


def candidate(case: str, n: int, output: str = "An answer.") -> Candidate:
    return Candidate(
        id=f"{case}-{n}",
        case_id=case,
        prompt=f"Prompt for {case}",
        rubric=f"Rubric for {case}",
        output=output,
        source=SECRET if n == 1 else "other-model",
    )


def write_candidates(path: Path, items: list[Candidate]) -> Path:
    path.write_text("".join(c.model_dump_json() + "\n" for c in items), encoding="utf-8")
    return path


def read_labels(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def label_cli(cands: Path, out: Path, keys: str) -> tuple[int, str]:
    result = CliRunner().invoke(
        app, ["label", str(cands), "--out", str(out), "--seed", "1"], input=keys
    )
    return result.exit_code, result.output


# --- Grading order ----------------------------------------------------------------------------


def test_grading_order_keeps_answers_to_one_prompt_apart() -> None:
    items = [candidate(f"case{i}", n) for i in range(15) for n in (1, 2)]
    for seed in range(20):
        order = grading_order(items, random.Random(seed))
        assert sorted(c.id for c in order) == sorted(c.id for c in items)
        cases = [c.case_id for c in order]
        assert all(a != b for a, b in itertools.pairwise(cases))
        # every prompt is seen once before any is seen twice
        assert len(set(cases[:15])) == 15


def test_grading_order_is_shuffled_and_seeded() -> None:
    items = [candidate(f"case{i}", n) for i in range(10) for n in (1, 2)]
    a = [c.id for c in grading_order(items, random.Random(7))]
    b = [c.id for c in grading_order(items, random.Random(7))]
    c = [c.id for c in grading_order(items, random.Random(8))]
    assert a == b
    assert a != c


def test_grading_order_falls_back_when_adjacency_is_unavoidable() -> None:
    items = [candidate("only", n) for n in (1, 2, 3)]
    assert len(grading_order(items, random.Random(0))) == 3


# --- Session ----------------------------------------------------------------------------------


def session_for(tmp_path: Path, items: list[Candidate], seed: int = 0) -> LabelSession:
    cands = write_candidates(tmp_path / "candidates.jsonl", items)
    return load_session(cands, tmp_path / "labeled.jsonl", random.Random(seed))


def test_mark_records_flags_and_clears_them(tmp_path: Path) -> None:
    session = session_for(tmp_path, [candidate("a", 1), candidate("b", 1)])
    first = session.current
    assert first is not None
    session.unsure, session.note = True, "close call"
    label = session.mark("fail", "2026-11-15T00:00:00.000Z")
    assert (label.id, label.label, label.unsure, label.note) == (
        first.id,
        "fail",
        True,
        "close call",
    )
    assert (session.unsure, session.note) == (False, None)
    assert session.current is not None and session.current.id != first.id


def test_undo_brings_back_the_card_with_its_flags(tmp_path: Path) -> None:
    session = session_for(tmp_path, [candidate("a", 1), candidate("b", 1)])
    first = session.current
    assert first is not None
    session.note = "hmm"
    session.mark("pass", "t1")
    undone = session.undo()
    assert undone is not None and undone.id == first.id
    assert session.current == first
    assert session.note == "hmm"
    assert session.labels == {}
    assert session.undo() is None


def test_resume_skips_labeled_and_undo_reaches_back_across_sessions(tmp_path: Path) -> None:
    items = [candidate("a", 1), candidate("b", 1), candidate("c", 1)]
    session = session_for(tmp_path, items)
    session.mark("pass", "2026-11-15T10:00:00.000Z")
    second = session.mark("fail", "2026-11-15T10:00:05.000Z")
    save_labels(tmp_path / "labeled.jsonl", session.ordered_labels())

    resumed = session_for(tmp_path, items, seed=99)
    assert len(resumed.queue) == 1
    undone = resumed.undo()
    assert undone is not None and undone.id == second.id
    assert len(resumed.queue) == 2


def test_labels_are_saved_in_candidate_order(tmp_path: Path) -> None:
    items = [candidate(c, 1) for c in "abcde"]
    session = session_for(tmp_path, items, seed=3)
    while session.current is not None:
        session.mark("pass", "t")
    assert [lab.id for lab in session.ordered_labels()] == [c.id for c in items]


def test_a_label_for_a_changed_answer_is_an_error(tmp_path: Path) -> None:
    session = session_for(tmp_path, [candidate("a", 1, output="old answer")])
    session.mark("pass", "t")
    save_labels(tmp_path / "labeled.jsonl", session.ordered_labels())
    with pytest.raises(ConfigError, match=r"label for 'a-1' was given to a different answer"):
        session_for(tmp_path, [candidate("a", 1, output="new answer")])


def test_labels_without_a_candidate_are_kept(tmp_path: Path) -> None:
    session = session_for(tmp_path, [candidate("a", 1), candidate("gone", 1)])
    while session.current is not None:
        session.mark("fail", "t")
    save_labels(tmp_path / "labeled.jsonl", session.ordered_labels())

    resumed = session_for(tmp_path, [candidate("a", 1)])
    assert resumed.current is None
    assert [lab.id for lab in resumed.ordered_labels()] == ["a-1", "gone-1"]


def test_redo_puts_a_label_back_in_the_queue(tmp_path: Path) -> None:
    items = [candidate("a", 1), candidate("b", 1), candidate("c", 1)]
    session = session_for(tmp_path, items)
    while session.current is not None:
        session.mark("pass", "t")
    save_labels(tmp_path / "labeled.jsonl", session.ordered_labels())

    cands = tmp_path / "candidates.jsonl"
    redo = load_session(cands, tmp_path / "labeled.jsonl", random.Random(0), redo=["b-1"])
    assert [c.id for c in redo.queue] == ["b-1"]
    assert sorted(redo.labels) == ["a-1", "c-1"]
    assert "b-1" not in redo.history


def test_redo_of_an_unknown_id_is_an_error(tmp_path: Path) -> None:
    cands = write_candidates(tmp_path / "candidates.jsonl", [candidate("a", 1)])
    with pytest.raises(ConfigError, match=r"--redo 'zz-9': no answer with that id"):
        load_session(cands, tmp_path / "labeled.jsonl", random.Random(0), redo=["zz-9"])


def test_redo_of_an_unlabeled_answer_is_fine(tmp_path: Path) -> None:
    cands = write_candidates(tmp_path / "candidates.jsonl", [candidate("a", 1)])
    session = load_session(cands, tmp_path / "labeled.jsonl", random.Random(0), redo=["a-1"])
    assert [c.id for c in session.queue] == ["a-1"]


def test_save_is_atomic_and_leaves_no_temp_file(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "labeled.jsonl"
    save_labels(out, [])
    assert out.read_text(encoding="utf-8") == ""
    assert not (out.parent / "labeled.jsonl.tmp").exists()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not json\n", r"candidates.jsonl:1: invalid JSON"),
        ('{"id": "a"}\n', r"candidates.jsonl:1: .*Field required"),
        ("", r"no candidates found"),
    ],
)
def test_bad_candidate_files_name_the_line(tmp_path: Path, content: str, message: str) -> None:
    cands = tmp_path / "candidates.jsonl"
    cands.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_session(cands, tmp_path / "labeled.jsonl", random.Random(0))


def test_duplicate_ids_are_an_error(tmp_path: Path) -> None:
    cands = write_candidates(tmp_path / "c.jsonl", [candidate("a", 1), candidate("a", 1)])
    with pytest.raises(ConfigError, match=r"duplicate id 'a-1' \(lines 1 and 2\)"):
        read_jsonl(cands, Candidate)


# --- Stats ------------------------------------------------------------------------------------


def test_balance_warning_only_for_a_finished_lopsided_set() -> None:
    assert LabelStats(30, 30, 27, 3, 0).balance_warning() == (
        "Only 3 of 30 labels are fail. Agreement on a set this lopsided mostly measures the "
        "easy calls, so add answers that come out fail."
    )
    assert LabelStats(30, 30, 18, 12, 0).balance_warning() is None
    assert LabelStats(30, 10, 10, 0, 0).balance_warning() is None
    assert LabelStats(0, 0, 0, 0, 0).balance_warning() is None


# --- Keys and the loop ------------------------------------------------------------------------


def test_normalize_key() -> None:
    assert normalize_key("\x1b[C") == "\x1b[C"
    assert normalize_key("\x1b[C\x1b[C") == "\x1b[C"  # a fast double press grades one card
    assert normalize_key("P") == "p"
    assert normalize_key("pf") == "p"  # buffered keys can't grade an unseen card
    assert normalize_key("") == ""


def test_arrow_keys_grade(tmp_path: Path) -> None:
    session = session_for(tmp_path, [candidate("a", 1), candidate("b", 1)])
    keys = iter(["\x1b[C", "\x1b[D"])
    saves: list[int] = []
    outcome = grade(
        session,
        console=Console(file=io.StringIO()),
        read_key=lambda: next(keys),
        read_note=lambda _: "",
        save=lambda: saves.append(1),
    )
    assert outcome == "done"
    assert sorted(lab.label for lab in session.labels.values()) == ["fail", "pass"]
    assert len(saves) == 2


# --- CLI --------------------------------------------------------------------------------------


def test_cli_grades_every_card_blind(tmp_path: Path) -> None:
    items = [candidate(c, n, output=f"answer {c}{n}") for c in "abc" for n in (1, 2)]
    cands = write_candidates(tmp_path / "candidates.jsonl", items)
    out = tmp_path / "labeled.jsonl"

    code, output = label_cli(cands, out, "p?fnclose call\npfpp")

    assert code == 0, output
    assert "Rubric for a" in output and "answer c2" in output  # the cards were drawn...
    assert "Card 1 of 6" in output and "0 saved" in output
    assert not any(c.id in output for c in items)  # ids can hint at the model, so no ids
    assert SECRET not in output and "other-model" not in output  # ...without the source
    assert "Done. 6/6 labeled: 4 pass, 2 fail, 1 unsure." in output
    labels = read_labels(out)
    assert [lab["id"] for lab in labels] == [c.id for c in items]
    flagged = [lab for lab in labels if lab["unsure"]]
    noted = [lab for lab in labels if lab["note"]]
    assert len(flagged) == 1 and flagged[0]["label"] == "fail"
    assert len(noted) == 1 and noted[0]["note"] == "close call" and noted[0]["label"] == "pass"
    assert all(lab["source"] in (SECRET, "other-model") for lab in labels)


def test_cli_quit_then_resume(tmp_path: Path) -> None:
    items = [candidate(c, 1) for c in "abcd"]
    cands = write_candidates(tmp_path / "candidates.jsonl", items)
    out = tmp_path / "labeled.jsonl"

    code, output = label_cli(cands, out, "ppq")
    assert code == 0
    assert "Stopped. 2/4 labeled" in output
    assert len(read_labels(out)) == 2

    code, output = label_cli(cands, out, "ff")
    assert code == 0
    assert "Done. 4/4 labeled: 2 pass, 2 fail, 0 unsure." in output
    assert "Only" not in output  # 2 of 4 is balanced


def test_cli_undo_reaches_back_into_an_earlier_session(tmp_path: Path) -> None:
    cands = write_candidates(tmp_path / "candidates.jsonl", [candidate("a", 1), candidate("b", 1)])
    out = tmp_path / "labeled.jsonl"
    label_cli(cands, out, "pq")
    [first] = read_labels(out)

    code, output = label_cli(cands, out, "ufp")

    assert code == 0, output
    assert "Undid pass." in output
    by_id = {lab["id"]: lab["label"] for lab in read_labels(out)}
    assert by_id[first["id"]] == "fail"
    assert len(by_id) == 2


def test_cli_undo_key(tmp_path: Path) -> None:
    items = [candidate(c, 1) for c in "abc"]
    cands = write_candidates(tmp_path / "candidates.jsonl", items)
    out = tmp_path / "labeled.jsonl"
    code, output = label_cli(cands, out, "uppuq")
    assert code == 0
    assert "Nothing to undo." in output
    assert "Undid pass." in output
    assert "Stopped. 1/3 labeled" in output
    assert len(read_labels(out)) == 1


def test_cli_redo_grades_an_answer_again(tmp_path: Path) -> None:
    cands = write_candidates(tmp_path / "candidates.jsonl", [candidate("a", 1), candidate("b", 1)])
    out = tmp_path / "labeled.jsonl"
    label_cli(cands, out, "pp")

    result = CliRunner().invoke(
        app, ["label", str(cands), "--out", str(out), "--redo", "a-1"], input="f"
    )

    assert result.exit_code == 0, result.output
    assert "Done. 2/2 labeled: 1 pass, 1 fail" in result.output
    assert {lab["id"]: lab["label"] for lab in read_labels(out)} == {"a-1": "fail", "b-1": "pass"}


def test_cli_warns_on_a_lopsided_set(tmp_path: Path) -> None:
    cands = write_candidates(tmp_path / "candidates.jsonl", [candidate(c, 1) for c in "abcde"])
    code, output = label_cli(cands, tmp_path / "labeled.jsonl", "ppppp")
    assert code == 0
    assert "Only 0 of 5 labels are fail." in output


def test_cli_unknown_key_says_so(tmp_path: Path) -> None:
    cands = write_candidates(tmp_path / "candidates.jsonl", [candidate("a", 1)])
    code, output = label_cli(cands, tmp_path / "labeled.jsonl", "xq")
    assert code == 0
    assert "Unknown key." in output


def test_cli_end_of_input_stops_cleanly(tmp_path: Path) -> None:
    cands = write_candidates(tmp_path / "candidates.jsonl", [candidate("a", 1)])
    out = tmp_path / "labeled.jsonl"
    code, output = label_cli(cands, out, "")
    assert code == 0
    assert "Stopped. 0/1 labeled" in output
    assert not out.exists()


def test_cli_missing_candidates_is_a_config_error(tmp_path: Path) -> None:
    code, output = label_cli(tmp_path / "nope.jsonl", tmp_path / "labeled.jsonl", "")
    assert code == 2
    assert "nope.jsonl: file not found" in output


def test_label_round_trips(tmp_path: Path) -> None:
    label = Label.model_validate(
        {**candidate("a", 1).model_dump(), "label": "pass", "labeled_at": "t"}
    )
    save_labels(tmp_path / "l.jsonl", [label])
    assert read_jsonl(tmp_path / "l.jsonl", Label) == [label]
    assert label.candidate() == candidate("a", 1)
