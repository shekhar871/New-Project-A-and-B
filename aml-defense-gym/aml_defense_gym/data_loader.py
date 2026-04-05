from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import pandas as pd


def default_csv_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "synthetic_transactions.csv"


def iter_transaction_chunks(
    path: Optional[Path] = None,
    *,
    chunksize: int = 10_000,
) -> Iterator[pd.DataFrame]:
    """
    Chunk reader mirroring the blueprint (pd.read_csv(..., chunksize=10_000)).
    Point `path` at IBM HI-Large_Trans.csv for full-scale training.
    """
    src = path or default_csv_path()
    reader = pd.read_csv(src, chunksize=chunksize)
    for chunk in reader:
        yield chunk
