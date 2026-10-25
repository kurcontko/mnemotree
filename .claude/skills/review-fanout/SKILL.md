---
name: review-fanout
description: "Reviews a diff with nine narrow lanes running in parallel — tests, lint, correctness, security, quality, test quality, performance, dependencies and deployment, simplification — then synthesises one ranked summary and a merge verdict. Use for a change big or risky enough to want every angle at once; for a quick look, use the built-in /code-review. Edits nothing."
---

# Fan-out review

Nine lanes review the same diff at once, each with its own context and its own
narrow question. You collect the reports and return one ranked summary.

Nine at once is more delegation than most work deserves. It is justified here
because the user asked for this review by name and the lanes are genuinely
independent — but any standing budget on concurrent subagents does not apply to
this skill. Everything else still holds: each lane gets a written brief, lane
reports are evidence rather than truth, and the user gets one integrated answer
rather than nine relayed ones.

## Scope

Resolve what to review in this order, and say which rule fired:

1. **The user named a scope** — a branch, commit SHA, PR number or URL, or file
   paths. Review that.
2. **On a feature branch** — `git diff main...HEAD` (three-dot, from the merge
   base). Substitute the repo's actual default branch.
3. **On the default branch with staged changes** — `git diff --staged`.
4. **On the default branch with nothing staged** — `git show HEAD`.

Three-dot diffs exclude staged and working-tree changes. If rule 2 fired and
the tree is dirty, say so and offer to include the working tree instead of
silently reviewing a diff the user has already moved past.

Resolve the ref and confirm the diff is non-empty **before spawning anything**.
A bad ref or an empty diff stops here, in front of the user, rather than inside
nine workers.

## Running the lanes

Launch all nine in a single message so they run concurrently.

Pick the most constrained worker available. A read-only agent that cannot edit
and has no `Agent` tool is ideal, because then a lane structurally cannot
rediscover this skill and fan out again — if a `verifier`-style agent is
installed, use it. Otherwise use a general-purpose subagent and rely on the
written rule below. **Never grant a lane the `Agent` tool.**

Every brief carries: the resolved scope and how to reproduce the diff, the
state the work starts from, the lane's question, and the reporting contract
below. Pass no model override — let each worker run on its own definition.

Two rules go in every brief:

- **Do not invoke this skill or spawn additional agents. Perform your lane
  directly.**
- **Every finding cites `file:line` in the diff or the surrounding code.** A
  claim about behaviour needs a citation in the source, not an inference from a
  name. A lane that cannot cite reports the concern as a coverage gap, not a
  finding.

If the repo has a `REVIEW.md`, pass it to every lane and honour it over the
defaults here — severity calibration, nit caps, skip paths, and repo-specific
checks. The lanes that judge convention (3, 5) read `AGENTS.md`.

### Lane 1 — Test runner

Run the tests relevant to the changed files. Report which tests ran, pass/fail
status, and the details of each failure. Never weaken, skip, or delete a test
to get a pass; a failing check is a finding.

### Lane 2 — Lint and static analysis

Run the project's linters and collect editor diagnostics for the changed files.
Report the tools used, warnings and errors, which are auto-fixable, and any type
errors or unresolved references.

### Lane 3 — Correctness

Read `AGENTS.md` first. Give up to five concrete improvements ranked by impact
and effort, formatted as
`[HIGH/MED/LOW impact, HIGH/MED/LOW effort] Title` with what, why, and how.
Only genuinely important issues — skip formatting, naming nits, and anything a
linter catches. Clean code gets fewer items, or none.

### Lane 4 — Security

Input validation and sanitisation; injection risks (SQL, command, XSS);
authentication and authorisation; secrets or credentials in code; error handling
that leaks sensitive information. Also missing error handling, swallowed errors,
and unhelpful error messages. Severity (Critical/High/Medium/Low) with
`file:line`.

### Lane 5 — Quality and style

Read `AGENTS.md` first. Complexity (long functions, deep nesting), dead code,
duplication. Then naming conventions, file placement, architectural patterns,
consistency with surrounding code, and the project's documented rules. Each issue gets a location, the problem, and a suggested fix.

### Lane 6 — Test quality

Coverage with an ROI lens: are the critical paths and the edge cases that matter
tested, and is coverage proportionate to risk? Quality: do the tests verify
behaviour rather than implementation, and will they break for the right reasons?
Test code quality: similar tests that should be parameterised, copy-pasted setup
that belongs in a fixture. Flakiness: timing dependencies, race conditions,
order-sensitive assertions, unawaited async. Anti-patterns: testing private
state, mocking so heavily nothing real is verified, coverage for its own sake.

### Lane 7 — Performance

N+1 queries and inefficient fetching, blocking operations in async contexts,
unnecessary recomputation or re-renders, memory leaks and unclosed resources,
missing pagination, expensive work in hot paths. Each concern gets its impact
and a fix.

### Lane 8 — Dependencies, breaking changes, deployment safety

New dependencies: justified, or already covered by an existing one? Maintained?
Bundle-size impact? Breaking changes: modified public interfaces, types or
exports, consumers that would break, whether a version bump is needed.
Deployment: migrations that could fail or lock, backward compatibility with
production data, ordering constraints, rollback safety, whether a feature flag
would help. Observability: if this fails in production, how would anyone know?

### Lane 9 — Simplification and atomicity

Fresh eyes on "could this be simpler?" — abstractions that do not pull their
weight, problems being solved that the project does not have, clever code that
costs clarity, premature helpers used once. Then atomicity: is this one logical
unit of work, are unrelated changes mixed in, could cleanup be split out, is it
sized for review? Each finding names the simpler alternative and the maintenance
cost saved.

## Synthesis

Lane reports are evidence, not truth. Before you promote a finding to Needs
attention, check its citation against the actual code. Drop findings whose
citation does not hold and say how many you dropped — a lane overstating an
impact costs the user a round trip.

Then: separate issues from suggestions, rank by severity across all lanes,
collapse clean lanes into one line, and give a verdict.

```
## Review summary — <resolved scope>

### Needs attention (N)
1. [Security] Title - file:line
   What is wrong, and what it costs.

### Suggestions (N)
1. [Quality] Title (HIGH impact, LOW effort) - file:line
   One or two lines.

### All clear
Tests (N passed), Lint (no issues), Performance, Dependencies

### Dropped
N findings whose citation did not hold.

### Verdict: Ready to merge | Needs attention | Needs work
One sentence on what to do next.
```

- **Ready to merge** — tests pass, nothing critical or high, suggestions optional.
- **Needs attention** — medium issues, or suggestions worth acting on first.
- **Needs work** — critical or high findings, or failing tests.

Findings are leads, not a backlog. Fixes create new surface and the judgement
calls are not deterministic between runs, so do not loop this until it comes
back clean. It will not.
