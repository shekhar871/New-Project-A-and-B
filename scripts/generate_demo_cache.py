#!/usr/bin/env python3
"""
G14 (blueprint) — pre-cache demo runs for a stable live presentation (e.g. H-3 before judging).

V2: run this to snapshot last N trainer/adversarial events. Placeholder: writes
`results/demo_cache_hint.txt` and exits 0. Replace with your event-bus export.
"""

from __future__ import annotations

from pathlib import Path

OUT = Path("results/demo_cache_hint.txt")


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "Run the FastAPI server, exercise /adversarial/episode and /trainer/start, "
        "then capture SSE from /events into results/demo_last_run.jsonl (manual for v1).\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
