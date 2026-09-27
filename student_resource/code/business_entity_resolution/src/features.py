"""Pairwise features shared by training and inference."""

import re

import pandas as pd
from rapidfuzz.fuzz import partial_ratio, ratio, token_set_ratio, token_sort_ratio


def _digit_set(value: str) -> set[str]:
    return set(re.findall(r"\d+", value or ""))


def build_pair_features(queries: pd.DataFrame, corpus: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    # vectorized field resolution (numpy fancy-indexing, not .iloc-per-row): the same
    # class of bottleneck found in evaluate.py/pipeline.py/matching.py (see
    # data_io.build_candidate_map) was also here, made worse by 8 rapidfuzz calls plus
    # ~8 pandas Series attribute accesses per row on top of it -- confirmed via a real
    # stuck SageMaker run (2026-09-27, 500k corpus / up to 6M candidate rows, stopped
    # after 34+ min with no result). rapidfuzz's own C-level comparison still needs a
    # per-pair Python call (no public elementwise-batched API for aligned pair lists,
    # as opposed to a full cross-product via process.cdist, which isn't what we want
    # here), so a Python loop remains -- but it now indexes plain numpy arrays instead
    # of pandas .iloc/Series, and results are collected into columns (fast DataFrame
    # construction) instead of a list of per-row dicts (slow).
    n = len(candidates)
    if n == 0:
        return pd.DataFrame()

    query_pos = candidates.query_pos.to_numpy()
    corpus_pos = candidates.corpus_pos.to_numpy()

    def field(frame: pd.DataFrame, name: str, pos):
        return frame[name].to_numpy()[pos]

    name_a, name_b = field(queries, "business_name_norm", query_pos), field(corpus, "business_name_norm", corpus_pos)
    addr_a, addr_b = field(queries, "business_address_norm", query_pos), field(corpus, "business_address_norm", corpus_pos)
    country_a, country_b = field(queries, "country", query_pos), field(corpus, "country", corpus_pos)
    city_a, city_b = field(queries, "city_token", query_pos), field(corpus, "city_token", corpus_pos)
    phonetic_a, phonetic_b = field(queries, "phonetic_key", query_pos), field(corpus, "phonetic_key", corpus_pos)
    s1_id = field(queries, "entity_id", query_pos)
    match_id = field(corpus, "entity_id", corpus_pos)

    cols = {key: [None] * n for key in (
        "name_exact", "name_token_sort", "name_token_set", "name_partial", "name_ratio",
        "address_token_sort", "address_partial", "address_ratio",
        "name_length_delta", "address_length_delta",
        "country_exact", "city_exact", "phonetic_exact", "digit_jaccard",
    )}
    for i in range(n):
        na, nb = name_a[i], name_b[i]
        aa, ab = addr_a[i], addr_b[i]
        cols["name_exact"][i] = float(na == nb and bool(na))
        cols["name_token_sort"][i] = token_sort_ratio(na, nb) / 100
        cols["name_token_set"][i] = token_set_ratio(na, nb) / 100
        cols["name_partial"][i] = partial_ratio(na, nb) / 100
        cols["name_ratio"][i] = ratio(na, nb) / 100
        cols["address_token_sort"][i] = token_sort_ratio(aa, ab) / 100
        cols["address_partial"][i] = partial_ratio(aa, ab) / 100
        cols["address_ratio"][i] = ratio(aa, ab) / 100
        cols["name_length_delta"][i] = abs(len(na) - len(nb))
        cols["address_length_delta"][i] = abs(len(aa) - len(ab))
        cols["country_exact"][i] = float(country_a[i] == country_b[i])
        cols["city_exact"][i] = float(city_a[i] == city_b[i] and bool(city_a[i]))
        cols["phonetic_exact"][i] = float(phonetic_a[i] == phonetic_b[i] and bool(phonetic_a[i]))
        digits_a, digits_b = _digit_set(aa), _digit_set(ab)
        union = digits_a | digits_b
        cols["digit_jaccard"][i] = len(digits_a & digits_b) / len(union) if union else 0.0

    result = pd.DataFrame(cols)
    result["tfidf_score"] = candidates["tfidf_score"].to_numpy(dtype=float)
    result["composite_match"] = candidates["composite_match"].to_numpy(dtype=float)
    result["exact_block_match"] = candidates["exact_match"].to_numpy(dtype=float)
    result["s1_id"] = s1_id
    result["match_id"] = match_id
    return result
