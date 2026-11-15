# Build log

How Grady was built: what the coding agent did, what review caught, and what was changed by
hand. One entry per working session or PR, newest last. Keep entries honest and specific; this
is the source for the README's "How this was built" section.

## Entry template

```
## YYYY-MM-DD: <milestone or topic> (PR #n)

**Agent did:** what was generated, in a sentence or two.
**Review caught:** each problem found in review, with a link to the PR comment.
**Changed by hand:** what the maintainer wrote or rewrote directly.
**Decisions:** any spec change or open choice settled, and why.
```

---

## 2026-11-14: Kickoff

**Agent did:** Claude Code (Opus 5.5) turned the first-draft plan into `SPEC.md`, filling in the
suite and models file formats, score-entry grammar, CLI flags, exit codes, retry rules,
result-record schema and timing definitions. It scaffolded the repo (pyproject, MIT license,
pre-commit, CI) and opened one GitHub issue per milestone.

**Review caught:** _(maintainer to fill in after reviewing SPEC.md)_

**Changed by hand:** _(maintainer to fill in)_

**Decisions:**
- Concurrency, timeout and retry moved from M2 into M1, since they are runner and client core.
- CI (ruff, mypy, pytest, gitleaks) ships with the scaffold so every milestone PR is checked.
- Scorers are async from M1 so `llm_judge` in M2 doesn't force an interface change.
- Exit code 3 added for request errors, so an unreachable endpoint can never pass a CI gate.
- `tokens_per_s` is defined as end-to-end throughput (prompt processing included), not decode
  speed, because v1 does not stream.
- M1's local test model is `gemma3:4b` on Ollama. It doesn't emit `<think>` blocks, which
  would break `exact` and `contains`. How to handle reasoning models is deferred to M2.
- The progress line is adapted from the maintainer's earlier project (civicbench), which
  already handles the terminal vs. non-terminal split.

---

## 2026-11-14: M1, core run loop (issue #1)

**Agent did:** Claude Code (Opus 5.5) wrote `config.py`, `client.py`, `runner.py`,
`progress.py`, `records.py`, `cli.py`, the three string scorers, the examples, a fake
OpenAI-compatible server and 98 tests (about 1.7 s, no network beyond localhost). Verified
against gemma3:4b on a local Ollama: 8/8 pass warm, exit codes 1 and 3 reproduced by hand.

**Agent's own checks caught** (before maintainer review):
- The first real run failed `count-letters` because the model answered "Three" and the case
  used `exact: "3"`. That was a test-design bug: it measured format, not counting. The case now
  uses a regex that accepts both.
- tok/s ranged from 0.4 to 11 on that run. The first requests waited on model load and queued
  behind each other (concurrency 4), and answers were only 2 to 4 tokens. The metric is correct
  by its definition but easy to misread, so SPEC section 8 now spells out the caveat and the
  summary table says "end to end".
- Rich hard-wrapped error messages at 80 columns, splitting file paths mid-word. Fixed with
  `soft_wrap` and covered by an end-to-end assertion.
- The summary table truncated the model label to "ollama-gem…". The Model column is now
  `no_wrap`, with shorter headers.
- The fake server's `serve_forever` default poll of 0.5 s added half a second to every test
  teardown. The suite went from 6.2 s to 1.7 s.

**Review caught** (maintainer, PR #5):
- **Cold-start time was being scored as model latency.** The maintainer asked for a warm-up
  request so the model is loaded before timing starts. Added: one untimed `max_tokens: 1`
  request per model (`--no-warmup` to skip), its latency recorded in `.meta.json`. A failed
  warm-up warns and the run continues, so every job still reports honestly. Re-run from a
  cold Ollama: the 3.3 s load went to the warm-up, and job latencies dropped from 4–5 s to
  0.3–2.3 s, matching a warm run.
- Kept: rejecting unquoted YAML numbers in scorer arguments, to revisit if a real use case
  appears.
- Kept: any request error means exit 3, even above the pass threshold, because an incomplete
  run shouldn't vouch for its own pass rate.
- Accepted: the `count-letters` regex fix.

**Changed by hand:** _(maintainer to fill in)_

**Decisions:**
- Unquoted YAML numbers in scorer arguments (`contains: 1250.00`) are rejected with a
  "quote it" hint instead of being coerced, since coercion would silently search for "1250.0".
- `contains: ""` is a load error because it always matches.
- An existing `--out` file is never overwritten (exit 2).
- A scorer that raises becomes a failed score with the exception in `detail`, so the run
  continues and nothing passes silently.
- API keys are redacted from error text even when a server echoes them back.
- Typer 0.29 vendors Click, so there is no separate Click dependency. mypy uses the Pydantic
  plugin so field defaults type-check.

---

## 2026-11-15: Judge-check set and `grady label` (pre-M2)

**Agent did:** Claude Code (Opus 5.5) wrote `grady label`, a one-key-per-card terminal grader
(`labeling.py`); `python -m grady.judgeset`, which answers each rubric prompt with several
models; 15 rubric prompts in `judge-check/cases.yaml`; and 36 tests. It generated
`judge-check/candidates.jsonl`: 30 answers, one per prompt from `gemma3:4b` and one from
`gemma3:1b`, at temperature 0.7 with no system prompt. The agent did not read or grade the
answers; it checked only counts (30 rows, 15 prompts, none empty). It also drove the real
command through a pseudo-terminal with arrow-key bytes and Ctrl-C to check the raw-key path.

**Review caught** (maintainer, while grading on PR #6):
- **The header looked like a done count.** After grading 9 cards the maintainer had 8 labels
  saved: the header said "9/30" on the card being shown. It now reads "Card 9 of 30 · 8 saved".
  The same check found that a fast double press of an arrow arrives as one read
  (`\x1b[C\x1b[C`) and was rejected as an unknown key. Now the first arrow counts.
- **The set was too easy: 29 pass, 1 fail.** `gemma3:1b` passed all 15 prompts, so it wasn't
  the weak model it was meant to be. On a set like that, a judge that passes everything agrees
  97% of the time and scores 0 on kappa. `grady label`'s balance warning fired. Fix:
  10 prompts with strict, checkable rules (exact counts, "nothing else", a false premise),
  answered by both gemma models and `qwen2.5:0.5b`, plus `qwen2.5:0.5b` on the original 15.
  That adds 45 answers, 75 in all. `python -m grady.judgeset --add` appends them without
  touching the 30 already labeled.
- **Two labels contradicted their rubric.** The agent compared countable rubric rules (word
  and line counts) with the labels without reading the answers. All matched except both
  `subject-line` answers: 12–13 lines and about 100 words each, labeled pass against "One
  subject line only, under 10 words". The judge sees only the rubric, so the maintainer chose
  to re-grade both to the rubric as written (`grady label --redo ID`), not to loosen it.
  Ids came off the card at the same time, since numbers added by `--add` would hint at the
  newer model.

**Changed by hand:** _(maintainer to fill in)_

**Decisions:**
- **Moved before M2.** The labels have to exist before the judge prompt is written, or the
  prompt gets tuned toward the answers it already agrees with.
- **Two models of different strength** instead of one, so the set has fails as well as
  passes. A finished set that is less than 25% pass or less than 25% fail gets a warning.
- **Blind, spread apart, no skip.** The card never shows the source model, and ids are
  numbered in a shuffled order. Two answers to the same prompt are never adjacent, so neither
  is graded against the other. Unsure answers get flagged instead of skipped, because skipping
  them would inflate agreement.
- **Pass/fail labels, not 1–5,** to match what the scorer decides. Faster and more consistent
  for a human.
- **Labels are pinned to answers.** Regenerating the candidates makes any label that now points
  at a different answer a load error.
- **Terminal tool, not a web page.** SPEC rules out a web UI, and this keeps the tool in the
  repo with tests. Saves go through a temp file and a rename on every key, so a crash or
  Ctrl-C can't lose or truncate labels.
