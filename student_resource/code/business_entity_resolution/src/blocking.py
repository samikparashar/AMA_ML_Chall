"""Composite-key, exact-match, and two-stage ANN + hybrid cosine/magnitude blocking.

Stage A (coarse recall) runs an HNSW approximate nearest-neighbor search over
small dense char n-gram vectors to prune the ~10M-row corpus down to a few
hundred candidates per query cheaply. Stage B (precise rerank) computes exact
cosine similarity on the full high-dimensional IDF-weighted sparse vectors for
just that shortlist, blended with a magnitude-similarity term (length-of-vector
ratio) so near-identical short/long name pairs like "Acme Inc" vs "Acme
Incorporated Holdings LLC" don't score deceptively high on cosine alone.
"""

from dataclasses import dataclass

import hnswlib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as l2_normalize


@dataclass
class BlockingConfig:
    batch_size: int = 10000
    max_candidates: int = 50
    n_features: int = 2**20
    max_df_frac: float = 0.005
    composite_bonus: float = 0.5
    exact_bonus: float = 0.35
    ann_dim: int = 256
    ann_candidates: int = 300
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 128
    hybrid_alpha: float = 0.7


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
        self.coarse_vectorizer = HashingVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), n_features=self.config.ann_dim,
            alternate_sign=True, norm=None, lowercase=False,
        )
        text = self.corpus["business_name_norm"] + " " + self.corpus["business_address_norm"]
        raw = self.vectorizer.transform(text)
        self._fit_sparse_transform(raw)
        self._build_ann_index(text)
        self.composite_groups = self.corpus.groupby("composite_key").indices
        self.exact_groups = self.corpus.groupby(
            ["country", "business_name_norm", "city_token"], dropna=False
        ).indices

    def _fit_sparse_transform(self, raw: sparse.csr_matrix) -> None:
        self.n_corpus = raw.shape[0]
        document_frequency = np.asarray((raw > 0).sum(axis=0)).ravel()
        keep = document_frequency <= self.config.max_df_frac * self.n_corpus
        idf = np.zeros_like(document_frequency, dtype=np.float32)
        present = document_frequency > 0
        idf[present & keep] = np.log((1.0 + self.n_corpus) / (1.0 + document_frequency[present & keep])) + 1.0
        raw = raw.tocsr(copy=False)
        raw.data *= np.take(idf, raw.indices).astype(np.float32)
        norms = np.sqrt(np.asarray(raw.multiply(raw).sum(axis=1)).ravel())
        self.matrix = l2_normalize(raw, norm="l2", axis=1, copy=False).tocsr()
        self.corpus_norms = np.maximum(norms, 1e-6).astype(np.float32)
        self.idf = idf

    def transform(self, frame: pd.DataFrame) -> tuple[sparse.csr_matrix, np.ndarray]:
        text = frame["business_name_norm"] + " " + frame["business_address_norm"]
        matrix = self.vectorizer.transform(text).tocsr(copy=False)
        matrix.data *= np.take(self.idf, matrix.indices).astype(np.float32)
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
        normalized = l2_normalize(matrix, norm="l2", axis=1, copy=False).tocsr()
        return normalized, np.maximum(norms, 1e-6).astype(np.float32)

    def _coarse_vectors(self, text: pd.Series) -> np.ndarray:
        dense = self.coarse_vectorizer.transform(text).toarray().astype(np.float32)
        return l2_normalize(dense, norm="l2", axis=1)

    def _build_ann_index(self, corpus_text: pd.Series) -> None:
        self.ann_index = hnswlib.Index(space="ip", dim=self.config.ann_dim)
        self.ann_index.init_index(
            max_elements=self.n_corpus,
            ef_construction=self.config.hnsw_ef_construction,
            M=self.config.hnsw_m,
        )
        chunk = 200_000
        for start in range(0, self.n_corpus, chunk):
            stop = min(start + chunk, self.n_corpus)
            vectors = self._coarse_vectors(corpus_text.iloc[start:stop])
            self.ann_index.add_items(vectors, np.arange(start, stop))
        self.ann_index.set_ef(self.config.hnsw_ef_search)

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
        query = query.reset_index(drop=True)
        composite = self._composite_candidates(query)
        exact = self._exact_candidates(query)
        fine_matrix, query_norms = self.transform(query)
        text = query["business_name_norm"] + " " + query["business_address_norm"]
        coarse_vectors = self._coarse_vectors(text)
        k = min(self.config.ann_candidates, self.n_corpus)
        labels, _ = self.ann_index.knn_query(coarse_vectors, k=k)

        rows = []
        for query_pos in range(len(query)):
            candidate_ids = set(int(c) for c in labels[query_pos])
            candidate_ids.update(composite.get(query_pos, {}))
            candidate_ids.update(exact.get(query_pos, {}))
            if not candidate_ids:
                continue
            candidate_array = np.fromiter(candidate_ids, dtype=np.int64)
            fine_sub = self.matrix[candidate_array]
            cosine = np.asarray(fine_matrix[query_pos].dot(fine_sub.T).todense()).ravel()
            candidate_norms = self.corpus_norms[candidate_array]
            magnitude = np.minimum(query_norms[query_pos], candidate_norms) / np.maximum(
                query_norms[query_pos], candidate_norms
            )
            hybrid = self.config.hybrid_alpha * cosine + (1 - self.config.hybrid_alpha) * magnitude
            order = np.argsort(hybrid)[::-1][: self.config.max_candidates]
            for idx in order:
                corpus_pos = int(candidate_array[idx])
                rows.append({
                    "query_pos": query_pos,
                    "corpus_pos": corpus_pos,
                    "tfidf_score": float(hybrid[idx]),
                    "composite_match": corpus_pos in composite.get(query_pos, {}),
                    "exact_match": corpus_pos in exact.get(query_pos, {}),
                })
        return pd.DataFrame(rows, columns=["query_pos", "corpus_pos", "tfidf_score", "composite_match", "exact_match"])


def build_corpus(source2: pd.DataFrame, source3: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([source2, source3], ignore_index=True)
