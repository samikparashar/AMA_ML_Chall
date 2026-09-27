"""Candidate labeling, LightGBM training, threshold selection, and output."""

import numpy as np
import pandas as pd

from .data_io import build_candidate_map, write_id_lists
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


def save_model(model, columns: list[str], threshold: float, model_dir: str) -> None:
    import json
    import os

    os.makedirs(model_dir, exist_ok=True)
    model.booster_.save_model(f"{model_dir}/model.txt")
    with open(f"{model_dir}/model_meta.json", "w") as f:
        json.dump({"columns": columns, "threshold": threshold}, f)


def load_model(model_dir: str):
    import json

    import lightgbm as lgb
    booster = lgb.Booster(model_file=f"{model_dir}/model.txt")
    with open(f"{model_dir}/model_meta.json") as f:
        meta = json.load(f)
    return booster, meta["columns"], meta["threshold"]


def build_outputs(s1: pd.DataFrame, corpus: pd.DataFrame, candidates: pd.DataFrame, features: pd.DataFrame,
                  model, columns: list[str], threshold: float, output_dir: str) -> None:
    import os

    os.makedirs(output_dir, exist_ok=True)
    scored = features.copy()
    if hasattr(model, "predict_proba"):
        scored["probability"] = model.predict_proba(scored[columns])[:, 1]
    else:
        # a native lgb.Booster (e.g. from load_model) -- binary objective predict() is P(positive) directly
        scored["probability"] = model.predict(scored[columns])
    selected = scored[scored.probability >= threshold]
    candidate_map = build_candidate_map(s1, corpus, candidates)
    match_map = {}
    for row in selected.itertuples(index=False):
        match_map.setdefault(row.s1_id, set()).add(row.match_id)
    ids = s1.entity_id.tolist()
    write_id_lists(f"{output_dir}/candidate_pairs.tsv", ids, candidate_map, "candidate_entity_ids")
    write_id_lists(f"{output_dir}/matching_results.tsv", ids, match_map, "matched_entity_ids")
