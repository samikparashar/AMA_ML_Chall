"""Blocking recall and macro F-beta evaluation."""

import numpy as np
import pandas as pd


def load_ground_truth(path: str) -> dict[str, set[str]]:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {row.source1_entity_id: set(filter(None, row.matched_entity_ids.split(",")))
            for row in frame.itertuples()}


def blocking_report(queries: pd.DataFrame, corpus: pd.DataFrame, candidates: pd.DataFrame, truth: dict[str, set[str]], sample_misses: int = 5) -> dict:
    # vectorized: map positional query_pos/corpus_pos -> entity_id via numpy fancy indexing,
    # then let pandas' groupby do the set-building in C rather than looping .iloc per row
    # (that loop was O(len(candidates)) with real per-call .iloc overhead -- up to millions
    # of rows at scale; this is O(len(candidates)) numpy indexing + O(num_queries) grouping).
    query_ids = queries.entity_id.to_numpy()
    corpus_ids = corpus.entity_id.to_numpy()
    pairs = pd.DataFrame({
        "s1_id": query_ids[candidates.query_pos.to_numpy()],
        "candidate_id": corpus_ids[candidates.corpus_pos.to_numpy()],
    })
    candidate_sets = pairs.groupby("s1_id")["candidate_id"].apply(set).to_dict() if len(pairs) else {}
    total = found_count = 0
    misses = []
    for row in queries.itertuples():
        expected = truth.get(row.entity_id, set())
        actual = candidate_sets.get(row.entity_id, set())
        total += len(expected)
        found_count += len(expected & actual)
        if len(misses) < sample_misses:
            misses.extend(sorted(expected - actual))
    counts = np.array([len(candidate_sets.get(row.entity_id, set())) for row in queries.itertuples()])
    return {
        "recall": found_count / total if total else 0.0,
        "candidate_avg": float(counts.mean()) if len(counts) else 0.0,
        "candidate_median": float(np.median(counts)) if len(counts) else 0.0,
        "candidate_p90": float(np.percentile(counts, 90)) if len(counts) else 0.0,
        "candidate_p95": float(np.percentile(counts, 95)) if len(counts) else 0.0,
        "candidate_max": int(counts.max()) if len(counts) else 0,
        "zero_candidate_entities": int((counts == 0).sum()),
        "missed_match_ids": misses[:sample_misses],
    }


def f_beta_macro(frame: pd.DataFrame, truth: dict[str, set[str]], all_s1_ids=None, beta: float = 0.5) -> float:
    all_s1_ids = list(all_s1_ids) if all_s1_ids is not None else frame.s1_id.unique()
    groups = {s1_id: group for s1_id, group in frame.groupby("s1_id")}
    scores = []
    for s1_id in all_s1_ids:
        group = groups.get(s1_id)
        predicted = set(group.loc[group.probability >= 0.5, "match_id"]) if group is not None else set()
        expected = truth.get(s1_id, set())
        tp = len(predicted & expected)
        precision = tp / len(predicted) if predicted else (1.0 if not expected else 0.0)
        recall = tp / len(expected) if expected else (1.0 if not predicted else 0.0)
        denominator = beta * beta * precision + recall
        scores.append((1 + beta * beta) * precision * recall / denominator if denominator else 0.0)
    return float(np.mean(scores)) if scores else 0.0
