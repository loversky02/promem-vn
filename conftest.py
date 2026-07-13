"""Make the sibling automem-vn importable for real-Crafter experiments.

Offline tests do NOT import automem or crafter (they inject a stub env + a
simple describe_fn), so the whole suite stays $0 and network-free even without
automem-vn checked out. This only helps `experiments/run_3arm.py`.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# add sibling automem-vn (its package root) if present
_automem = _HERE.parent / "automem-vn"
if _automem.exists():
    sys.path.insert(0, str(_automem))
