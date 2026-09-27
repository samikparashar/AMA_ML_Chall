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


def build_candidate_map(queries: pd.DataFrame, corpus: pd.DataFrame, candidates: pd.DataFrame) -> dict[str, set[str]]:
    """Map each query's entity_id -> set of candidate entity_ids, from positional query_pos/corpus_pos.

    Vectorized via numpy fancy-indexing + groupby -- a plain Python loop calling
    .iloc[...] per row here was confirmed (2026-09-27, real SageMaker run) to take
    ~391s on 6M candidate rows; this version takes ~3s for the same input, identical
    output. This same pattern occurred independently in three places (evaluate.py's
    blocking_report, pipeline.py's stage2, matching.py's build_outputs) -- use this
    helper instead of reintroducing a fourth copy.
    """
    if not len(candidates):
        return {}
    query_ids = queries.entity_id.to_numpy()
    corpus_ids = corpus.entity_id.to_numpy()
    pairs = pd.DataFrame({
        "s1_id": query_ids[candidates.query_pos.to_numpy()],
        "candidate_id": corpus_ids[candidates.corpus_pos.to_numpy()],
    })
    return pairs.groupby("s1_id")["candidate_id"].apply(set).to_dict()


def write_id_lists(path: str | Path, ids: Iterable[str], mapping: dict[str, Iterable[str]], value_name: str) -> None:
    rows = [{"source1_entity_id": entity_id, value_name: ",".join(sorted(set(mapping.get(entity_id, ())))) } for entity_id in ids]
    pd.DataFrame(rows, columns=["source1_entity_id", value_name]).to_csv(path, sep="\t", index=False)
