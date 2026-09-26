"""Pairwise features shared by training and inference."""

import re

import pandas as pd
from rapidfuzz.fuzz import partial_ratio, ratio, token_set_ratio, token_sort_ratio


def _digit_set(value: str) -> set[str]:
    return set(re.findall(r"\d+", value or ""))


def _pair(left: pd.Series, right: pd.Series, block: pd.Series) -> dict[str, float]:
    name_a, name_b = left.business_name_norm, right.business_name_norm
    addr_a, addr_b = left.business_address_norm, right.business_address_norm
    digits_a, digits_b = _digit_set(addr_a), _digit_set(addr_b)
    union = digits_a | digits_b
    return {
        "name_exact": float(name_a == name_b and bool(name_a)),
        "name_token_sort": token_sort_ratio(name_a, name_b) / 100,
        "name_token_set": token_set_ratio(name_a, name_b) / 100,
        "name_partial": partial_ratio(name_a, name_b) / 100,
        "name_ratio": ratio(name_a, name_b) / 100,
        "address_token_sort": token_sort_ratio(addr_a, addr_b) / 100,
        "address_partial": partial_ratio(addr_a, addr_b) / 100,
        "address_ratio": ratio(addr_a, addr_b) / 100,
        "name_length_delta": abs(len(name_a) - len(name_b)),
        "address_length_delta": abs(len(addr_a) - len(addr_b)),
        "country_exact": float(left.country == right.country),
        "city_exact": float(left.city_token == right.city_token and bool(left.city_token)),
        "phonetic_exact": float(left.phonetic_key == right.phonetic_key and bool(left.phonetic_key)),
        "digit_jaccard": len(digits_a & digits_b) / len(union) if union else 0.0,
        "tfidf_score": float(block.tfidf_score),
        "composite_match": float(block.composite_match),
        "exact_block_match": float(block.exact_match),
    }


def build_pair_features(queries: pd.DataFrame, corpus: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for block in candidates.itertuples(index=False):
        left = queries.iloc[int(block.query_pos)]
        right = corpus.iloc[int(block.corpus_pos)]
        values = _pair(left, right, pd.Series(block._asdict()))
        values.update({"s1_id": left.entity_id, "match_id": right.entity_id})
        rows.append(values)
    return pd.DataFrame(rows)
