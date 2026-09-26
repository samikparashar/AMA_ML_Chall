"""TSV loading and submission output helpers."""

from pathlib import Path
from typing import Iterable

import pandas as pd

from .normalize import add_normalized_columns

RECORD_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def load_full_tsv(path: str | Path, normalize: bool = True) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = set(RECORD_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return add_normalized_columns(frame) if normalize else frame


def load_sample(path: str | Path, rows: int = 20) -> pd.DataFrame:
    return load_full_tsv(path).head(rows)


def write_id_lists(path: str | Path, ids: Iterable[str], mapping: dict[str, Iterable[str]], value_name: str) -> None:
    rows = [{"source1_entity_id": entity_id, value_name: ",".join(sorted(set(mapping.get(entity_id, ())))) } for entity_id in ids]
    pd.DataFrame(rows, columns=["source1_entity_id", value_name]).to_csv(path, sep="\t", index=False)
