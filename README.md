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

> **Fidelity caveat (read first).** The paper's memory agent is itself an LLM (Opus 4.6)
> running a two-phase workflow — Phase 1 edits a structured bank (`status` / `knowledge` /
> `procedural`) via tool calls; Phase 2 an LLM decides inject-vs-silent. *Our* memory agent
> is **rule-based** (bank parsed from observations; gate = stall/loop heuristics). So the
> results below test a cheap **rule-based lower-bound** of ProMem, not the paper's LLM
> memory agent. Read the nulls — especially the strong-agent "break" — as *"a crude gate
> does not realize the gains,"* not *"ProMem's mechanism fails."* The paper reports gains
> even for a strong action agent (Opus +2.4pp Terminal-Bench, +2.5pp τ²-Bench) with its LLM
> memory agent; a faithful LLM memory agent here is the key next step (see `docs/NEXT.md`).

**1. Passive dumping is inert and expensive; only a floored weak agent gains from mere exposure.**
For a capable agent (Sonnet, 8 seeds) passive full-context does *not* beat no-mem
(7.39% vs 8.52% — within noise) yet costs ~2907 extra tokens/ep: dumping the whole
bank is a bad trade. (An earlier n=2 hint that full-context > no-mem was a
small-sample artifact.) Mere exposure helps in exactly one regime — a *floored*
weak agent: Haiku (60 steps) unlocks nothing from raw observations (0.0%) but
reaches `collect_wood` (4.55%) once either memory arm surfaces the current subgoal.

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
- Caveat: the baseline overlays (full 9.09, no-mem 4.55) are 2-seed estimates; the
  8-seed de-noise (finding 3) revises them to 7.39 / 8.52 and shows arm gaps are ≈1σ.
  The cliff *shape* (sparse best, saturating worst) is the robust part, not the exact deltas.

This reproduces the paper's *selective intervention > passive bank exposure* and
extends it: over-injection destroys the benefit — the same spamming failure mode
seen when a memory policy is trained in the sibling `napmem-vn`.

**3. Active injection is the only arm that lifts the mean — suggestive, not conclusive.**
8-seed de-noise (Sonnet, best gate): active **11.36%** vs full-context 7.39% vs
no-mem 8.52%, i.e. active − full = **+3.98pp** at ~25× less context (116 vs 2907
tokens). The edge *grew* with seeds (+2.27 at n=2 → +3.98 at n=8) and 6/8 seeds are
solid — but it is still ≈1σ (pooled stdev ~3.9), so suggestive, not established.
Tempering: an inject-audit finds wasted-rate **0.82** (only 18% of injects are
followed by progress within 5 steps) — active helps on net, but the gate is
imprecise, so the benefit is diffuse rather than per-inject.

**4. For strong agents, capability dominates and no memory benefit is detectable (honest partial-break).**
A 3-seed panel with stronger action agents reaches ~27% with *no* memory (Opus 4.8 and
GPT-5.6 alike) — 3× Sonnet, 6× Haiku. Memory does not lift the mean, and the sign of
active − full-context *flips* across models (Opus +6.06pp, GPT −6.06pp) inside enormous
noise (full-context stdev 13–21 at n=3). The paper's claim that memory helps *stronger*
agents does not reproduce here: capability dominates, and passive full-context can even
hurt (Opus 16.7% vs no-mem 27.3%). What *does* replicate across every model: active-injection
is the lowest-variance arm and ~35× cheaper than passive dumping (100 vs ~3500 tokens),
with inject wasted-rate ~0.70–0.82 throughout. (Caveat: n=3 at stdev ~15 cannot resolve a
~6pp effect — this is "not detected," not "proven zero.")

**5. With a *faithful* LLM memory agent AND a capable action agent, memory helps a lot — but "selective" buys cost, not score.**
Findings 1–4 used a rule-based memory agent. Re-running with the paper's actual design — an
LLM memory agent (Opus) curating a two-phase bank (`promem/llm_agent/`), plus a capable
action agent (Sonnet) — flips the picture. Fair regime, 3 seeds × 80 steps: no-mem 4.55%,
full-bank **15.15%**, active **12.12%** — both memory arms roughly *triple* no-mem (+7.6 to
+10.6pp). This confirms the paper's core direction: memory helps a capable long-horizon
agent. **Two factors are BOTH required** — a floored weak agent (Haiku: 0→3%, no lift) and a
rule-parsed bank on a capable agent (rule-gate Sonnet: ≈1σ) each show little; only
capable-action + LLM-curated-memory shows the big lift. *However*, we do **not** reproduce
"selective > passive" on Crafter: active did not beat exposing the full curated bank on score
(−3.03pp, within noise at n=3). What selective buys is **efficiency** — active captures most
of the lift at **233 tokens vs full-bank's 15,793 (~68× cheaper)**; inject wasted-rate stays
~0.83. So here "when to surface" is a cost lever, not an accuracy one. (n=3: the
memory>no-mem lift looks robust — no-mem stdev 0 — the selective-vs-passive gap is noise.)

**Bottom line.** With a faithful LLM memory agent and a capable action agent, memory roughly
triples score over no-mem on Crafter (+7–10pp) — confirming the paper's core claim, and
showing it needs **both** a capable action agent **and** a rich LLM-curated bank (a rule-parsed
bank *or* a floored weak agent shows little). What we do *not* reproduce is "selective >
passive": selective injection ≈ exposing the full bank on score, but wins decisively on cost
(~68× fewer tokens). So on this substrate, "when to surface" is an **efficiency** lever, not an
accuracy one — while over-injection (rule-gate sweep) still collapses to no-mem. Findings 1–4
were a rule-based lower-bound; finding 5 is the faithful result and supersedes the pessimistic
reading of 1 & 4.

See `docs/NEXT.md` for the build plan, gateway invocation, and resume point.
