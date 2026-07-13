"""Faithful ProMem: LLM two-phase memory agent, 3 arms on Crafter.

The memory agent is an LLM (paper uses Opus 4.6) running Phase-1 bank management +
Phase-2 inject/silent; the action agent is a (weaker) LLM. Arms: no-mem / full-bank
(expose the LLM-curated bank) / active (LLM decides inject vs silent).

  python experiments/run_llm_3arm.py --act-model cc/claude-haiku-4-5-20251001 \
      --mem-model cc/claude-opus-4-8 --episodes 2 --steps 60 --interval 6

Needs AUTOMEM_INSECURE_SSL=1 + a 9router endpoint. Memory-agent LLM calls dominate cost.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from promem.agent import LLMActionAgent
from promem.agent.action_agent import with_retries
from promem.envs import make_crafter, task_vocab, default_describe
from promem.llm_agent import LLMMemoryAgent
from promem.llm_agent.runner import run_episode_llm, LLM_ARMS


def _mean(xs):
    return statistics.mean(xs) if xs else 0.0


def _stdev(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-model", dest="act_model", default="cc/claude-haiku-4-5-20251001")
    ap.add_argument("--mem-model", dest="mem_model", default="cc/claude-opus-4-8")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--interval", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-traces", dest="dump_traces", action="store_true")
    ap.add_argument("--out", default="results/llm_3arm.json")
    args = ap.parse_args()

    from automem.meta.driver import client_from_env
    describe_fn = default_describe()
    act_backend = with_retries(client_from_env(args.act_model))
    mem_backend = with_retries(client_from_env(args.mem_model))

    per_arm = {a: [] for a in LLM_ARMS}
    mem_calls = {a: 0 for a in LLM_ARMS}
    traces = []
    for arm in LLM_ARMS:
        for e in range(args.episodes):
            seed = args.seed + e
            env = make_crafter(seed)
            action_agent = LLMActionAgent(act_backend, task_vocab(env))
            mem_agent = LLMMemoryAgent(mem_backend, interval=args.interval)
            res = run_episode_llm(env, action_agent, mem_agent, arm,
                                  max_steps=args.steps, describe_fn=describe_fn)
            per_arm[arm].append(res)
            mem_calls[arm] += mem_agent.phase1_calls + mem_agent.phase2_calls
            if args.dump_traces:
                traces.append({"arm": arm, "seed": seed, "unlocked": res.unlocked,
                               "trace": [list(t) for t in res.trace]})
            print(f"[{arm:>9}] ep{e} seed{seed}: score={res.score*100:5.1f}% "
                  f"injects={res.injects} ctx_tok={res.ctx_tokens} "
                  f"mem_calls(p1={mem_agent.phase1_calls},p2={mem_agent.phase2_calls}) "
                  f"unlocked={res.unlocked}")

    summary = {"act_model": args.act_model, "mem_model": args.mem_model,
               "episodes": args.episodes, "steps": args.steps, "interval": args.interval, "arms": {}}
    for arm, rs in per_arm.items():
        scores = [r.score for r in rs]
        summary["arms"][arm] = {
            "mean_score_pct": 100 * _mean(scores), "stdev_pct": 100 * _stdev(scores),
            "mean_injects": _mean([r.injects for r in rs]),
            "mean_ctx_tokens": _mean([r.ctx_tokens for r in rs]),
            "total_mem_calls": mem_calls[arm],
            "records": [r.digest() for r in rs],
        }
    a, f, n = (summary["arms"]["active"], summary["arms"]["full-bank"], summary["arms"]["no-mem"])
    summary["active_minus_fullbank_pct"] = a["mean_score_pct"] - f["mean_score_pct"]
    summary["active_minus_nomem_pct"] = a["mean_score_pct"] - n["mean_score_pct"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    traces_pending = traces if args.dump_traces else None
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if traces_pending is not None:
        side = out.with_suffix(".traces.jsonl")
        side.write_text("\n".join(json.dumps(t) for t in traces_pending), encoding="utf-8")

    print(f"\nfaithful ProMem: active - full-bank = {summary['active_minus_fullbank_pct']:+.2f} pp | "
          f"active - no-mem = {summary['active_minus_nomem_pct']:+.2f} pp  -> {out}")


if __name__ == "__main__":
    main()
