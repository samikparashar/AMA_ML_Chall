"""Complementary composite-key and sparse hashed n-gram blocking."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as l2_normalize


@dataclass
class BlockingConfig:
    batch_size: int = 10000
    max_candidates: int = 50
    tfidf_top_k: int = 50
    n_features: int = 2**18
    max_df_frac: float = 0.005
    similarity_chunk: int = 500
    composite_bonus: float = 0.5
    exact_bonus: float = 0.35


class BlockingIndex:
    def __init__(self, corpus: pd.DataFrame, config: BlockingConfig | None = None):
        self.config = config or BlockingConfig()
        self.corpus = corpus.reset_index(drop=True).copy()
        self.corpus["composite_key"] = self.corpus.apply(
            lambda row: f"{row.country}|{row.phonetic_key}|{row.city_token}".lower(), axis=1
        )
        self.vectorizer = HashingVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), n_features=self.config.n_features,
            alternate_sign=False, norm=None, lowercase=False,
        )
        text = self.corpus["business_name_norm"] + " " + self.corpus["business_address_norm"]
        raw = self.vectorizer.transform(text)
        self._fit_sparse_transform(raw)
        self.composite_groups = self.corpus.groupby("composite_key").indices
        self.exact_groups = self.corpus.groupby(
            ["country", "business_name_norm", "city_token"], dropna=False
        ).indices

    def _fit_sparse_transform(self, raw: sparse.csr_matrix) -> None:
        self.n_corpus = raw.shape[0]
        document_frequency = np.asarray((raw > 0).sum(axis=0)).ravel()
        keep = document_frequency <= self.config.max_df_frac * self.n_corpus
        idf = np.ones_like(document_frequency, dtype=np.float32)
        present = document_frequency > 0
        idf[present] = np.log((1.0 + self.n_corpus) / (1.0 + document_frequency[present])) + 1.0
        raw = raw.tocsc(copy=True)
        raw[:, ~keep] = 0
        raw = raw.tocsr()
        raw.data *= np.take(idf, raw.indices).astype(np.float32)
        self.matrix = l2_normalize(raw, norm="l2", axis=1, copy=False).tocsr()
        self.keep_mask, self.idf = keep, idf

    def transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        text = frame["business_name_norm"] + " " + frame["business_address_norm"]
        matrix = self.vectorizer.transform(text).tocsc()
        matrix[:, ~self.keep_mask] = 0
        matrix = matrix.tocsr()
        matrix.data *= np.take(self.idf, matrix.indices).astype(np.float32)
        return l2_normalize(matrix, norm="l2", axis=1, copy=False).tocsr()

    def _composite_candidates(self, query: pd.DataFrame) -> dict[int, dict[int, float]]:
        left = query[["country", "phonetic_key", "city_token"]].copy()
        left["composite_key"] = (left["country"] + "|" + left["phonetic_key"] + "|" + left["city_token"]).str.lower()
        left["query_pos"] = np.arange(len(left))
        right = self.corpus[["composite_key"]].reset_index(names="corpus_pos")
        merged = left.merge(right, on="composite_key", how="inner")
        merged = merged[merged["composite_key"] != merged["country"].str.lower() + "||"]
        return {int(pos): {int(row.corpus_pos): self.config.composite_bonus for row in group.itertuples()}
                for pos, group in merged.groupby("query_pos")}

    def _exact_candidates(self, query: pd.DataFrame) -> dict[int, dict[int, float]]:
        """Add exact normalized-name/city matches without broad country buckets."""
        result = {}
        for query_pos, row in query.reset_index(drop=True).iterrows():
            if not row.business_name_norm or not row.city_token:
                continue
            corpus_positions = self.exact_groups.get(
                (row.country, row.business_name_norm, row.city_token), ()
            )
            if len(corpus_positions) <= self.config.max_candidates * 2:
                result[query_pos] = {
                    int(corpus_pos): self.config.exact_bonus
                    for corpus_pos in corpus_positions
                }
        return result

    def candidates(self, query: pd.DataFrame) -> pd.DataFrame:
        composite = self._composite_candidates(query)
        exact = self._exact_candidates(query)
        query_matrix = self.transform(query)
        scores = {i: dict(exact.get(i, {})) for i in range(len(query))}
        for query_pos, values in composite.items():
            for corpus_pos, score in values.items():
                scores[query_pos][corpus_pos] = max(scores[query_pos].get(corpus_pos, 0.0), score)
        for start in range(0, len(query), self.config.similarity_chunk):
            stop = min(start + self.config.similarity_chunk, len(query))
            similarities = query_matrix[start:stop].dot(self.matrix.T).tocsr()
            for local, row in enumerate(similarities):
                if row.nnz == 0:
                    continue
                order = np.argsort(row.data)[-self.config.tfidf_top_k:]
                target = scores[start + local]
                for index in order:
                    corpus_pos = int(row.indices[index])
                    target[corpus_pos] = max(target.get(corpus_pos, 0.0), float(row.data[index]))
        rows = []
        for query_pos, candidates in scores.items():
            ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)[: self.config.max_candidates]
            for corpus_pos, score in ranked:
                rows.append({"query_pos": query_pos, "corpus_pos": corpus_pos, "tfidf_score": score,
                         "composite_match": corpus_pos in composite.get(query_pos, {}),
                         "exact_match": corpus_pos in exact.get(query_pos, {})})
            return pd.DataFrame(rows, columns=["query_pos", "corpus_pos", "tfidf_score", "composite_match", "exact_match"])


def build_corpus(source2: pd.DataFrame, source3: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([source2, source3], ignore_index=True)
