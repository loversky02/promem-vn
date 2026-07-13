"""Phase 0 experiment: no-mem vs full-context vs active-injection on Crafter.

Offline default uses the HeuristicActionAgent (no network, $0) so the harness is
CI-runnable — but heuristic numbers are PLUMBING ONLY, not a finding. Real Q1/Q2
signal needs `--policy llm` (routes the action agent through 9router).

  # plumbing check ($0):
  python experiments/run_3arm.py --episodes 4 --steps 120

  # real finding (9router):
  python experiments/run_3arm.py --policy llm --model gpt-5.5 --episodes 8

  # Q2 spam curve (sweep gate aggressiveness, active arm only):
  python experiments/run_3arm.py --sweep --policy llm --model gpt-5.5 --episodes 6
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from promem import MemoryBank, Gate, run_episode, ARMS
from promem.agent import HeuristicActionAgent, LLMActionAgent
from promem.agent.action_agent import with_retries
from promem.envs import make_crafter, task_vocab, default_describe


def build_agent(kind: str, vocab: list[str], seed: int, model: str | None):
    if kind == "heuristic":
        return HeuristicActionAgent(vocab, seed=seed)
    if kind == "llm":
        from automem.meta.driver import client_from_env
        return LLMActionAgent(with_retries(client_from_env(model)), vocab)
    raise ValueError(kind)


def _mean(xs):
    return statistics.mean(xs) if xs else 0.0


def _stdev(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def run_arms(args, describe_fn):
    """Q1: all three arms on identical world seeds."""
    per_arm: dict[str, list] = {a: [] for a in ARMS}
    traces: list = []
    for arm in ARMS:
        for e in range(args.episodes):
            seed = args.seed + e
            env = make_crafter(seed)
            agent = build_agent(args.policy, task_vocab(env), seed, args.model)
            res = run_episode(env, agent, MemoryBank(), Gate(args.stall_n, args.cooldown_k),
                              arm, max_steps=args.steps, describe_fn=describe_fn)
            per_arm[arm].append(res)
            if args.dump_traces:
                traces.append({"arm": arm, "seed": seed, "unlocked": res.unlocked,
                               "trace": [list(t) for t in res.trace]})
            print(f"[{arm:>16}] ep{e} seed{seed}: score={res.score*100:5.1f}%  "
                  f"injects={res.injects} rate={res.inject_rate:.2f} "
                  f"ctx_tok={res.ctx_tokens} unlocked={res.unlocked}")

    summary = {"mode": "q1", "policy": args.policy, "model": args.model,
               "episodes": args.episodes, "steps": args.steps,
               "gate": {"stall_n": args.stall_n, "cooldown_k": args.cooldown_k}, "arms": {}}
    for arm, rs in per_arm.items():
        scores = [r.score for r in rs]
        summary["arms"][arm] = {
            "mean_score_pct": 100 * _mean(scores), "stdev_pct": 100 * _stdev(scores),
            "mean_injects": _mean([r.injects for r in rs]),
            "mean_inject_rate": _mean([r.inject_rate for r in rs]),
            "mean_ctx_tokens": _mean([r.ctx_tokens for r in rs]),
            "records": [r.digest() for r in rs],
        }
    a, f = summary["arms"]["active-injection"], summary["arms"]["full-context"]
    summary["Q1_active_minus_fullctx_pct"] = a["mean_score_pct"] - f["mean_score_pct"]
    if args.dump_traces:
        summary["_traces"] = traces
    return summary


def run_sweep(args, describe_fn):
    """Q2: active arm only, vary gate aggressiveness -> inject_rate vs score."""
    grid = [(2, 1), (4, 2), (6, 3), (8, 5), (12, 8), (20, 12)]  # (stall_n, cooldown_k)
    points = []
    for stall_n, cooldown_k in grid:
        rs = []
        for e in range(args.episodes):
            seed = args.seed + e
            env = make_crafter(seed)
            agent = build_agent(args.policy, task_vocab(env), seed, args.model)
            rs.append(run_episode(env, agent, MemoryBank(), Gate(stall_n, cooldown_k),
                                  "active-injection", max_steps=args.steps, describe_fn=describe_fn))
        pt = {"stall_n": stall_n, "cooldown_k": cooldown_k,
              "mean_inject_rate": _mean([r.inject_rate for r in rs]),
              "mean_score_pct": 100 * _mean([r.score for r in rs]),
              "mean_ctx_tokens": _mean([r.ctx_tokens for r in rs])}
        points.append(pt)
        print(f"gate(stall={stall_n:2d},cool={cooldown_k:2d}): "
              f"inject_rate={pt['mean_inject_rate']:.2f}  score={pt['mean_score_pct']:5.1f}%")
    return {"mode": "q2_sweep", "policy": args.policy, "model": args.model,
            "episodes": args.episodes, "steps": args.steps, "points": points}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", choices=["heuristic", "llm"], default="heuristic")
    ap.add_argument("--model", default=None)
    ap.add_argument("--stall-n", dest="stall_n", type=int, default=8)
    ap.add_argument("--cooldown-k", dest="cooldown_k", type=int, default=5)
    ap.add_argument("--sweep", action="store_true", help="run the Q2 aggressiveness sweep")
    ap.add_argument("--dump-traces", dest="dump_traces", action="store_true",
                    help="write per-episode traces to a .traces.jsonl sidecar (for inject audit)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    describe_fn = default_describe()
    summary = run_sweep(args, describe_fn) if args.sweep else run_arms(args, describe_fn)

    out = Path(args.out) if args.out else Path(
        f"results/{'q2_sweep' if args.sweep else 'q1_3arm'}_{args.policy}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    traces = summary.pop("_traces", None)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if traces is not None:
        side = out.with_suffix(".traces.jsonl")
        side.write_text("\n".join(json.dumps(t) for t in traces), encoding="utf-8")
        print(f"-> {side} ({len(traces)} episode traces)")

    if not args.sweep:
        print(f"\nQ1: active-injection − full-context = "
              f"{summary['Q1_active_minus_fullctx_pct']:+.2f} pp  "
              f"({'heuristic plumbing, NOT a finding' if args.policy=='heuristic' else 'LLM'})")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
