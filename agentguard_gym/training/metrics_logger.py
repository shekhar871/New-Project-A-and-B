from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


class JsonlMetricsLogger:
    def __init__(self, path: str = "runs/training_metrics.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, row: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

