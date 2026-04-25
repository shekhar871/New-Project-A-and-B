#!/usr/bin/env python3
"""
Reproducible baseline on `data/holdout_attacks.json` without GPU or API keys.

This is a **keyword** heuristic (not a trained model) so judges can run:
  uv run python inference.py
after clone and get a printout comparable to `scripts/eval_before_after.py`.

For OpenEnv episode rollouts with the full environment, use the HTTP client or
`DefenderBackend` in `grandfinals.adversarial_loop` with `HF_TOKEN` / `API_BASE_URL`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOLDOUT = ROOT / "data" / "holdout_attacks.json"

# Broad keyword set: blocks most obvious holds; counts as a crude "before" signal.
_RISK = (
    "ignore",
    "override",
    "system",
    "169.254",
    "127.0.0.1",
    "localhost",
    "[::",
    "inject",
    "jailbreak",
    "bypass",
    "exfil",
    "file://",
    "gopher://",
    "cluster.local",
    "unrestricted",
    "disable security",
    "translate map",
    "semantic anchor",
)


def main() -> int:
    if not HOLDOUT.is_file():
        print(f"[ERR] missing {HOLDOUT}", file=sys.stderr)
        return 1
    data = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    attacks = data.get("attacks", [])
    if not attacks:
        print("[ERR] holdout has no attacks[]", file=sys.stderr)
        return 1
    tps = 0
    for atk in attacks:
        text = (atk.get("text") or "").lower()
        blocked = any(k in text for k in _RISK)
        o = "TP" if blocked else "FN"
        print(f"[STEP] id={atk.get('id')} task={atk.get('task')} blocked_heuristic={blocked} outcome={o}")
        tps += int(o == "TP")
    score = tps / len(attacks)
    print(f"[END] holdout_block_rate={score:.1%} (heuristic; NOT a trained model)")
    print("      For LLM+API baseline, set HF_TOKEN and use scripts/eval_before_after.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
