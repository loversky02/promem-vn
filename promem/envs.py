"""Thin bridge to automem-vn's verified Crafter substrate.

All imports are LAZY so the offline test-suite (which injects a stub env and its
own describe_fn) never triggers `import automem` / `import crafter`. Only
`experiments/run_3arm.py` pays the real dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_automem_on_path() -> None:
    here = Path(__file__).resolve().parent.parent           # promem-vn/
    sibling = here.parent / "automem-vn"                     # Build with Paper/automem-vn
    if sibling.exists() and str(sibling) not in sys.path:
        sys.path.insert(0, str(sibling))


def default_describe():
    """Return automem's crafter_text.describe (env, info) -> str."""
    _ensure_automem_on_path()
    from automem.envs import crafter_text
    return crafter_text.describe


def make_crafter(seed: int):
    """Construct a real Crafter env (needs `crafter` installed in this venv)."""
    import crafter
    return crafter.Env(seed=seed)


def task_vocab(env) -> list[str]:
    return list(env.action_names)
