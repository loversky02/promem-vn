#!/usr/bin/env python3
"""Render the Q2 injection-rate vs score sweep to a dependency-free SVG money-plot.

Stdlib only (no matplotlib) to keep the repo MLX/Mac-light. Reads the sweep JSON
and overlays the Sonnet baselines from the Q1 run.

  python scripts/plot_q2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(rel: str, default=None):
    f = ROOT / rel
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else default


def main() -> None:
    sweep = _load("results/q2_sweep_sonnet.json")
    if not sweep:
        sys.exit("missing results/q2_sweep_sonnet.json — run run_3arm.py --sweep first")
    pts = sorted(sweep["points"], key=lambda p: p["mean_inject_rate"])
    arms = (_load("results/q1_3arm_sonnet.json", {}) or {}).get("arms", {})
    nomem = arms.get("no-mem", {}).get("mean_score_pct", 4.55)
    full = arms.get("full-context", {}).get("mean_score_pct", 9.09)

    W, H = 640, 380
    L, R, T, B = 64, 24, 44, 52
    pw, ph = W - L - R, H - T - B
    xmax, ymax = 1.0, 12.0

    def X(v: float) -> float:
        return L + pw * (v / xmax)

    def Y(v: float) -> float:
        return T + ph * (1 - v / ymax)

    e: list[str] = []
    e.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
             f'font-family="sans-serif" role="img" '
             f'aria-label="ProMem Q2: injection rate versus Crafter score with spam collapse">')
    e.append('<rect width="100%" height="100%" fill="white"/>')
    for g in range(0, 13, 2):
        y = Y(g)
        e.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="#e1e0d9"/>')
        e.append(f'<text x="{L-8}" y="{y+4:.1f}" font-size="11" fill="#898781" text-anchor="end">{g}</text>')
    for t in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        e.append(f'<text x="{X(t):.1f}" y="{H-B+18}" font-size="11" fill="#898781" '
                 f'text-anchor="middle">{t:.1f}</text>')
    e.append(f'<text x="{L+pw/2:.0f}" y="{H-8}" font-size="12" fill="#52514e" '
             f'text-anchor="middle">injection rate (reminders / step)</text>')
    e.append(f'<text x="16" y="{T+ph/2:.0f}" font-size="12" fill="#52514e" text-anchor="middle" '
             f'transform="rotate(-90 16 {T+ph/2:.0f})">Crafter score %</text>')
    for val, color, label in ((full, "#1baf7a", f"full-context {full:.2f}"),
                              (nomem, "#898781", f"no-mem {nomem:.2f}")):
        y = Y(val)
        e.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="{color}" '
                 f'stroke-width="2" stroke-dasharray="6 4"/>')
        e.append(f'<text x="{W-R}" y="{y-5:.1f}" font-size="11" fill="{color}" '
                 f'text-anchor="end">{label}</text>')
    poly = " ".join(f"{X(p['mean_inject_rate']):.1f},{Y(p['mean_score_pct']):.1f}" for p in pts)
    e.append(f'<polyline points="{poly}" fill="none" stroke="#2a78d6" stroke-width="2"/>')
    n = len(pts)
    for i, p in enumerate(pts):
        x, y = X(p["mean_inject_rate"]), Y(p["mean_score_pct"])
        c, r = ("#008300", 6) if i == 0 else ("#e34948", 6) if i == n - 1 else ("#2a78d6", 4)
        e.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{c}"/>')
    b, w = pts[0], pts[-1]
    e.append(f'<text x="{X(b["mean_inject_rate"])+8:.1f}" y="{Y(b["mean_score_pct"])-8:.1f}" '
             f'font-size="11" fill="#008300">best {b["mean_score_pct"]:.1f}% @ {b["mean_ctx_tokens"]:.0f} tok</text>')
    e.append(f'<text x="{X(w["mean_inject_rate"])-8:.1f}" y="{Y(w["mean_score_pct"])-8:.1f}" '
             f'font-size="11" fill="#e34948" text-anchor="end">spam collapse to no-mem</text>')
    e.append(f'<text x="{L}" y="24" font-size="13" fill="#0b0b0b">ProMem Q2: sparse selective '
             f'injection wins; over-injection collapses to no-mem</text>')
    e.append('</svg>')

    out = ROOT / "results/q2_curve.svg"
    out.write_text("\n".join(e), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
