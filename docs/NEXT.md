# ProMem-VN — NEXT (resume point)

## Paper
- **Title:** Remember When It Matters: Proactive Memory Agent for Long-Horizon Agents
- **arXiv:** 2607.08716
- **Failure it names:** *behavioral state decay* — task facts, prior attempts, open
  subgoals get buried in / pushed past the context window and stop influencing the
  next action.
- **Fix:** a separate **memory agent** beside an **unmodified action agent**; maintains
  a structured memory bank from the recent trajectory; each step decides **inject a
  memory-grounded reminder** vs **stay silent**. Plug-and-play with frontier agents.
- **Central claim to reproduce:** *selective intervention > passive bank exposure.*
- **Their numbers (from abstract — verify against full PDF):** +8.3 pp Terminal-Bench 2.0,
  +6.8 pp τ²-Bench. Trained Qwen3.5-27B on "SETA" (their training substrate — we swap
  for Crafter) via SFT + GRPO.
- **From full PDF only if Phase 0 shows signal:** exact memory-bank fields + exact gating
  policy (abstract does not specify them; Phase 0 uses the minimal design below).

## Thesis (Phase 0, $0)
Reproduce "selective > passive" on a long-horizon substrate we own (Crafter, 200 steps),
and add the axis the paper omits: the **injection-frequency → reward** curve — where
proactive reminding tips into context-spam. This is also the clean test for the meta
question "does agent memory actually help":
- **Q1** — does **active-injection beat full-context** (not just no-mem)?  *(honest bar)*
- **Q2** — at what inject rate does active-injection **collapse into spam** and hurt?

## Substrate (already built + verified in automem-vn)
- `automem-vn/automem/envs/crafter_text.py` — text Crafter wrapper.
- `automem-vn/automem/agent/runner.py::run_episode(..., max_steps=200)` — long-horizon.
- Action policy v0 (baseline 11.4% Crafter) — used **UNMODIFIED** across all 3 arms.

## 3 arms (same action policy; only the context differs)
1. **no-mem** — action agent alone.
2. **full-context** — passively prepend the whole trajectory / running bank every step,
   no gating. *The strong baseline the paper must beat.*
3. **active-injection** — MemoryAgent runs in parallel, gates each step: inject a
   grounded reminder or stay silent.

## MemoryAgent (Phase 0 minimal design — NOT yet paper-exact)
- **Bank fields:** `facts` (world/task facts observed), `attempts` (actions tried +
  outcome), `open_subgoals` (goals opened, not yet achieved).
- **Update:** parse each Crafter step (obs, action, achievement delta) into the three
  lists; flip a subgoal to done when its achievement unlocks.
- **Gating (inject vs silent):** inject when (a) an open subgoal has gone N steps with no
  progress, OR (b) the action agent is repeating a failed attempt already in `attempts`;
  suppress if a reminder fired within the last K steps (anti-spam). Start rule-based;
  optionally add a 9router LLM-judge gate as arm **3b**.

## Metric + success
- Primary: Crafter achievement count / reward per episode, averaged over seeds.
- **Q1 pass:** arm 3 > arm 2 beyond the seed noise band.
- **Q2 curve:** sweep gating aggressiveness (N, K, or judge threshold) → plot reward vs
  realized inject-rate; locate the spam cliff. Label each injection necessary/unnecessary
  with the reused `napmem-vn` probe.

## Reuse map ($0)
- Crafter env + runner + action policy ← automem-vn
- unnecessary-call / reward-hacking probe ← napmem-vn
- LLM calls ← 9router (GPT-5.5 / Claude)

## Explicitly NOT in Phase 0 (defer)
- Terminal-Bench 2.0 / τ²-Bench (their benches — expensive to stand up; only if Phase 0
  signal justifies it).
- GRPO/SFT training of the memory agent (they train Qwen3.5-27B; we start **training-free**
  with a rule/judge gate; train only if the frozen gate shows signal and we want the
  money-plot).
- Paper-exact bank/gating (pull from full PDF only after Phase 0 signal).

## STATUS — Phase 0 harness BUILT + verified ($0, 2026-07-11)
- `promem/` package: `memory/bank.py` (3-field bank + Crafter tech tree), `inject/gate.py`
  (stall/loop gate + cooldown), `agent/action_agent.py` (Heuristic + LLM, unmodified),
  `runner.py` (3-arm loop), `envs.py` (lazy bridge to automem Crafter).
- `experiments/run_3arm.py` — Q1 (3 arms) + `--sweep` Q2; heuristic default, `--policy llm`.
- **15/15 tests green** (hermetic stub env, no crafter/automem needed).
- **Real-Crafter plumbing verified**: full-context spends ~10× the extra context tokens of
  active-injection (~1045 vs ~105 /ep). Heuristic Q1 gap is plumbing, NOT a finding.

## Gateway invocation (working, 2026-07-11)
- Run tests + experiments with automem-vn's venv python; prefix `PYTHONPATH=.`.
- **9router gotchas:** self-hosted cert → set `AUTOMEM_INSECURE_SSL=1`. Model ids are
  prefix-routed: `cc/claude-*` (Claude) and `cx/gpt-*` (GPT). At build time the `cx/`
  (Codex/GPT) route returned 401 `token_invalidated` → **use `cc/`**. `gpt-5.5` alone 404s.
- First read command:
  `AUTOMEM_INSECURE_SSL=1 PYTHONPATH=. <automem venv>/python experiments/run_3arm.py \
   --policy llm --model cc/claude-haiku-4-5-20251001 --episodes 3 --steps 60`

## First read — Haiku, E3×60 (2026-07-11)
no-mem **0.00%** · full-context **4.55%** (~1670 ctx tok/ep) · active **4.55%** (~165 tok, 11 injects).
Honest read: (a) memory lifts weak Haiku off the 0 floor — surfacing the current subgoal
gets it to `collect_wood`; (b) **active = full-context on SCORE, but at ~10× less context**
(the real signal is efficiency, not score); (c) score **saturates at 1 achievement** — no
headroom to test Q1 on score (metaskill weak-backbone failure mode again). Result JSON:
`results/q1_3arm_llm.json`.

## Second read — Sonnet cc/claude-sonnet-4-6, E2×100 (has headroom now)
no-mem **4.55%** (collect_wood ×2, ±0) · full-context **9.09%** (+place_table ×2, ±0, ~2839 tok)
· active **11.36% (±9.6!)** (ep0: 4 unlocks incl make_wood_pickaxe / ep1: only collect_wood; ~270 tok, 18 injects).
Honest read: **robust** = memory>no-mem (full reliably +1 tier) + active ~10× cheaper ctx for
score ≥ + active's ceiling is deeper. **NOT established** = active>full on SCORE: stdev 9.6 >>
gap 2.27pp, one lucky ep carries the mean, other ep active<full → n=2 noise (metaskill lesson
again). Mechanism note: active intervenes → higher variance (higher ceiling, less reliable);
passive is deterministic. Result: `results/q1_3arm_sonnet.json`.

## Q2 spam sweep — Sonnet E3×100 (2026-07-13), `results/q2_sweep_sonnet.json`
inject_rate → score%:  0.08→**10.61** · 0.12→6.06 · 0.19→6.06 · 0.31→7.58 · 0.48→6.06 · 0.98→**4.55**.
Clean at the extremes: gentlest gate (0.08) is BEST — beats full-context (9.09) at 115 tok
(~25× cheaper than full's 2839); saturating gate (0.98) **collapses to exactly no-mem (4.55)**
despite 1470 tok = the **context-spam cliff**. Middle is flat/noisy (n=3, Sonnet var ~9.6) →
no precise dose-response, only the endpoints are robust. Finding: selective, SPARSE injection
is the value; over-injection is as bad as no memory (mirrors napmem spam reward-hacking).

## Status (2026-07-13)
1. **SHIPPED** — finding written into README, `results/q2_curve.svg` money-plot (stdlib,
   `scripts/plot_q2.py`), `git init` + commit `fa6749b` (28 files, no AI attribution).
   Local only — NOT pushed to GitHub yet (awaiting user ok).
2. **inject-audit BUILT** — `promem/analysis/inject_audit.py` (necessary vs wasted proxy) +
   6 tests; `run_3arm.py --dump-traces` writes `.traces.jsonl`. Quantification pending #3's traces.
3. **DONE** — 8-seed de-noise (Sonnet, best gate) → `results/q1_denoise_sonnet.json`.

## De-noise result (2026-07-13) — findings revised
Sonnet best gate, 8 seeds: no-mem **8.52% ±6.16** · full-context **7.39% ±3.38** ·
active **11.36% ±4.86**. Surprises: (a) full-context does NOT beat no-mem — the n=2 hint
was a 2-seed artifact (passive dump inert AND costs 2907 tok); (b) active is the only arm
lifting the mean, active−full **+3.98pp** (grew from +2.27 at n=2) but ≈1σ (pooled 3.9),
suggestive not conclusive; (c) inject-audit wasted_rate **0.82** (only 18% of injects
followed by progress ≤5 steps) → active helps on net but the gate is imprecise. README
findings #1 (revised: passive inert) & #3 (de-noise) updated.

## Strong-agent panel — DONE (2026-07-14)
opus-4-8: no-mem **27.3%** ±4.6 / full 16.7 ±13.1 / active 22.7 ±4.6 (active−full +6.06, wasted 0.75).
gpt-5.6-sol: no-mem 27.3 ±9.1 / full **31.8** ±20.8 / active 25.8 ±5.3 (active−full **−6.06**, wasted 0.70).
Honest: **capability dominates** — strong agents ~27% with no memory (3× Sonnet, 6× Haiku);
memory does NOT lift the mean and active−full **flips sign** across models inside stdev 13–21
noise → the paper's stronger-agent claim does **not reproduce** on this substrate (honest
partial-break). Robust across every model: active = lowest-variance arm + ~35× cheaper than
full-dump (100 vs ~3500 tok); over-injection → no-mem. README finding #4 + bottom line committed.

## Remaining
- **GitHub push** (commits so far) + profile pin — awaiting user ok (outward-facing).
- Optional deepening (only if wanted): ~18 seeds/strong-model to resolve the ~6pp (expensive,
  effect may be 0); GRPO-train the gate to cut wasted-rate 0.7–0.8; paper-exact bank/gating from
  full PDF; Terminal-Bench/τ²-Bench (their benches).
2. `--sweep` for the **Q2** inject-rate → score curve; label injects necessary/unnecessary
   with the reused napmem probe → locate the context-spam cliff.
3. Plot Q1 bars + Q2 curve; write the honest finding into README/paper.
4. Only if Phase 0 signals: pull paper-exact bank/gating from full PDF; consider
   Terminal-Bench/τ²-Bench + GRPO-trained gate.
