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
