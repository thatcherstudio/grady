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
