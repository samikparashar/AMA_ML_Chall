"""Candidate labeling, LightGBM training, threshold selection, and output."""

import numpy as np
import pandas as pd

from .data_io import write_id_lists
from .evaluate import f_beta_macro


def label_pairs(features: pd.DataFrame, truth: dict[str, set[str]]) -> pd.DataFrame:
    result = features.copy()
    result["label"] = [float(match_id in truth.get(s1_id, set()))
                       for s1_id, match_id in zip(result.s1_id, result.match_id)]
    return result


def train_classifier(features: pd.DataFrame, validation_fraction: float = 0.2, seed: int = 42):
    import lightgbm as lgb
    groups = pd.Series(features.s1_id.unique()).sample(frac=1, random_state=seed)
    split = int(len(groups) * (1 - validation_fraction))
    train_ids, valid_ids = set(groups.iloc[:split]), set(groups.iloc[split:])
    columns = [column for column in features.columns if column not in {"s1_id", "match_id", "label"}]
    train = features[features.s1_id.isin(train_ids)]
    valid = features[features.s1_id.isin(valid_ids)]
    model = lgb.LGBMClassifier(objective="binary", n_estimators=1000, learning_rate=0.05,
                               num_leaves=63, random_state=seed, n_jobs=-1)
    fit_kwargs = {"eval_set": [(valid[columns], valid.label)]} if len(valid) else {}
    callbacks = [lgb.early_stopping(50, verbose=False)] if len(valid) else []
    model.fit(train[columns], train.label, callbacks=callbacks, **fit_kwargs)
    valid.attrs["s1_ids"] = list(valid_ids)
    return model, columns, valid


def sweep_threshold(model, columns: list[str], valid: pd.DataFrame, truth: dict[str, set[str]]) -> tuple[float, float]:
    scored = valid.copy()
    scored["probability"] = model.predict_proba(scored[columns])[:, 1]
    best = (0.5, -1.0)
    for threshold in np.arange(0.01, 0.991, 0.01):
        candidate = scored.copy()
        candidate["probability"] = (candidate["probability"] >= threshold).astype(float)
        score = f_beta_macro(candidate, truth, all_s1_ids=valid.attrs.get("s1_ids"))
        if score > best[1]:
            best = (float(threshold), score)
    return best


def build_outputs(s1: pd.DataFrame, corpus: pd.DataFrame, candidates: pd.DataFrame, features: pd.DataFrame,
                  model, columns: list[str], threshold: float, output_dir: str) -> None:
    import os

    os.makedirs(output_dir, exist_ok=True)
    scored = features.copy()
    scored["probability"] = model.predict_proba(scored[columns])[:, 1]
    selected = scored[scored.probability >= threshold]
    candidate_map, match_map = {}, {}
    for block in candidates.itertuples(index=False):
        candidate_map.setdefault(s1.iloc[int(block.query_pos)].entity_id, set()).add(corpus.iloc[int(block.corpus_pos)].entity_id)
    for row in selected.itertuples(index=False):
        match_map.setdefault(row.s1_id, set()).add(row.match_id)
    ids = s1.entity_id.tolist()
    write_id_lists(f"{output_dir}/candidate_pairs.tsv", ids, candidate_map, "candidate_entity_ids")
    write_id_lists(f"{output_dir}/matching_results.tsv", ids, match_map, "matched_entity_ids")
