# promem-vn

A **$0-on-a-Mac** reproduction + dissection of
**"Remember When It Matters: Proactive Memory Agent for Long-Horizon Agents"**
(arXiv 2607.08716).

The paper runs a **separate memory agent** alongside an **unmodified action agent**:
it maintains a structured memory bank from the recent trajectory and, each step,
decides whether to **inject a memory-grounded reminder** or **stay silent** — to counter
*behavioral state decay* (task facts, prior attempts, and open subgoals getting buried
in / pushed out of the context window, so they stop influencing the next action).
Headline: +8.3 pp on Terminal-Bench 2.0, +6.8 pp on τ²-Bench.
Core claim: **selective intervention > passive bank exposure.**

## The trio

| Axis | Repo | Question |
|------|------|----------|
| WRITE | AutoMem (`automem-vn`) | what to store |
| READ  | NapMem (`napmem-vn`)   | what to fetch |
| **WHEN** | **ProMem (this repo)** | **proactively surface the right fact at the right moment** |

## What this repo tests (Phase 0, $0)

Instead of Terminal-Bench / τ²-Bench, we probe the central claim on a long-horizon
substrate we **already own** — `automem-vn`'s **Crafter (200-step episodes,
multi-subgoal)** — and add one axis the paper doesn't: **when does proactive injection
collapse into context-spam?**

Three arms on the **same unmodified action policy**:

1. `no-mem` — action agent alone.
2. `full-context` — passively dump the whole trajectory / memory bank each step
   (no gating). *The strong baseline the paper must beat.*
3. `active-injection` — the memory agent gates each step: inject vs stay silent.

**Findings we're after**
- **Q1** — does (3) beat (2), not just (1)? *(the honest bar for "memory actually helps")*
- **Q2** — at what injection frequency does (3) turn into distracting spam?
  *(reuses the reward-hacking / unnecessary-call probe from `napmem-vn`)*

## Reuse (nothing rebuilt)

- Crafter env + episode runner + action policy v0 (11.4%) ← `automem-vn`
- unnecessary-call / reward-hacking probe ← `napmem-vn`
- LLM calls ← 9router gateway

## Findings (Phase 0, honest)

Action agent = Claude via a self-hosted 9router gateway; substrate = automem-vn's
text-Crafter. Heuristic numbers are plumbing only, never a finding.

**1. Memory beats no-memory — surfacing the open subgoal helps (robust).**
A weak agent (Haiku, 60 steps) unlocks nothing from raw observations (0.0%) but
reaches `collect_wood` (4.55%) as soon as either memory arm surfaces the current
subgoal.

**2. Selective ≫ passive on cost, and the spam cliff is real (robust at the extremes).**
Sweeping the gate from gentle to aggressive (Sonnet, n=3 — see `results/q2_curve.svg`):

| gate | inject-rate | score | extra ctx tokens |
|------|------------|-------|------------------|
| gentle (20/12) | 0.08 | **10.6%** | **115** |
| … middle … | 0.12–0.48 | ~6–7.6% (flat, noisy) | 175–715 |
| saturating (2/1) | 0.98 | **4.55%** | 1470 |

- The gentlest gate scores highest — above passive full-context (9.09%) at **~25×
  less context** (115 vs 2839 tokens).
- The saturating gate **collapses to exactly the no-mem baseline (4.55%)** despite
  spending 1470 tokens: over-injection is context-spam, as useless as no memory.
- The middle of the curve is flat within noise (n=3) — only the endpoints are robust.

This reproduces the paper's *selective intervention > passive bank exposure* and
extends it: over-injection destroys the benefit — the same spamming failure mode
seen when a memory policy is trained in the sibling `napmem-vn`.

**3. Not established: sparse active > passive on *score* (n=2, within noise).**
The Q1 score gap (Sonnet) was +2.27pp but with stdev 9.6 ≫ gap — one lucky episode
carried it, and another had active < full. An 8-seed de-noise is the honest next
step (see `docs/NEXT.md`).

See `docs/NEXT.md` for the build plan, gateway invocation, and resume point.
