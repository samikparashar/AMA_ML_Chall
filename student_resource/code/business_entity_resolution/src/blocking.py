"""Composite-key, exact-match, and IVF (k-means) + GPU batched cosine/magnitude blocking.

Stage A (coarse recall) clusters the corpus's small dense char n-gram vectors
with k-means, assigns each query to its nearest few centroids ("probes"), and
does one batched matmul per cluster on the GPU (Metal/MPS on Apple Silicon) to
gather a candidate shortlist. This keeps the per-query search sub-linear
(only a handful of clusters are scanned per query, never the full ~10M-row
corpus) while still using the GPU for the matmuls that remain, by grouping
work per cluster (a few thousand iterations) instead of per query (millions).
Stage B (precise rerank) computes exact cosine similarity on the full
high-dimensional IDF-weighted sparse vectors for just that shortlist, blended
with a magnitude-similarity term (vector-length ratio) so near-identical
short/long name pairs like "Acme Inc" vs "Acme Incorporated Holdings LLC"
don't score deceptively high on cosine alone.
"""

from dataclasses import dataclass

import lightgbm  # noqa: F401 -- must load before torch: the two bundle conflicting native
                 # OpenMP runtimes and loading torch first causes lightgbm to segfault on macOS.
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as l2_normalize

from .transliterate import add_translit_columns


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    ivf_n_clusters: int = 2048
    ivf_n_probe: int = 4
    ivf_sample_size: int = 300_000
    hybrid_alpha: float = 0.7


class BlockingIndex:
    def __init__(self, corpus: pd.DataFrame, config: BlockingConfig | None = None):
        self.config = config or BlockingConfig()
        self.device = _device()
        self.corpus = corpus.reset_index(drop=True).copy()
        self.corpus = add_translit_columns(self.corpus, self.device)
        self.corpus["composite_key"] = self.corpus.apply(
            lambda row: f"{row.country}|{row.phonetic_key_translit}|{row.city_token_translit}".lower(), axis=1
        )
        self.vectorizer = HashingVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), n_features=self.config.n_features,
            alternate_sign=False, norm=None, lowercase=False,
        )
        self.coarse_vectorizer = HashingVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), n_features=self.config.ann_dim,
            alternate_sign=True, norm=None, lowercase=False,
        )
        text = self.corpus["business_name_translit"] + " " + self.corpus["business_address_translit"]
        raw = self.vectorizer.transform(text)
        self._fit_sparse_transform(raw)
        self._build_ivf_index(text)
        self.composite_groups = self.corpus.groupby("composite_key").indices
        self.exact_groups = self.corpus.groupby(
            ["country", "business_name_translit", "city_token_translit"], dropna=False
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
        frame = add_translit_columns(frame, self.device)
        text = frame["business_name_translit"] + " " + frame["business_address_translit"]
        matrix = self.vectorizer.transform(text).tocsr(copy=False)
        matrix.data *= np.take(self.idf, matrix.indices).astype(np.float32)
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
        normalized = l2_normalize(matrix, norm="l2", axis=1, copy=False).tocsr()
        return normalized, np.maximum(norms, 1e-6).astype(np.float32)

    def _coarse_vectors(self, text: pd.Series) -> np.ndarray:
        dense = self.coarse_vectorizer.transform(text).toarray().astype(np.float32)
        return l2_normalize(dense, norm="l2", axis=1)

    def _build_ivf_index(self, corpus_text: pd.Series) -> None:
        n_clusters = max(1, min(self.config.ivf_n_clusters, self.n_corpus // 10 or 1))
        sample_n = min(self.config.ivf_sample_size, self.n_corpus)
        rng = np.random.default_rng(0)
        sample_pos = rng.choice(self.n_corpus, size=sample_n, replace=False)
        sample_vectors = self._coarse_vectors(corpus_text.iloc[sample_pos])
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=10000, n_init=3, max_iter=100, random_state=0)
        kmeans.fit(sample_vectors)
        centroids = l2_normalize(kmeans.cluster_centers_.astype(np.float32), norm="l2", axis=1)
        self.centroids = torch.from_numpy(centroids).to(self.device)
        self.n_clusters = n_clusters

        chunk = 200_000
        cluster_id = np.empty(self.n_corpus, dtype=np.int32)
        coarse_all = np.empty((self.n_corpus, self.config.ann_dim), dtype=np.float32)
        for start in range(0, self.n_corpus, chunk):
            stop = min(start + chunk, self.n_corpus)
            vectors = self._coarse_vectors(corpus_text.iloc[start:stop])
            coarse_all[start:stop] = vectors
            v = torch.from_numpy(vectors).to(self.device)
            sims = v @ self.centroids.T
            cluster_id[start:stop] = torch.argmax(sims, dim=1).cpu().numpy().astype(np.int32)

        order = np.argsort(cluster_id, kind="stable")
        counts = np.bincount(cluster_id, minlength=n_clusters)
        offsets = np.zeros(n_clusters + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        self.cluster_order = order
        self.cluster_offsets = offsets
        self.corpus_coarse_sorted = coarse_all[order]
        self._cluster_gpu_cache: dict[int, torch.Tensor] = {}
        self._cluster_ids_gpu_cache: dict[int, torch.Tensor] = {}

    def _cluster_tensor(self, cluster_id: int) -> torch.Tensor:
        cached = self._cluster_gpu_cache.get(cluster_id)
        if cached is not None:
            return cached
        start, stop = int(self.cluster_offsets[cluster_id]), int(self.cluster_offsets[cluster_id + 1])
        tensor = torch.from_numpy(self.corpus_coarse_sorted[start:stop]).to(self.device)
        self._cluster_gpu_cache[cluster_id] = tensor
        return tensor

    def _cluster_row_ids(self, cluster_id: int) -> torch.Tensor:
        cached = self._cluster_ids_gpu_cache.get(cluster_id)
        if cached is not None:
            return cached
        start, stop = int(self.cluster_offsets[cluster_id]), int(self.cluster_offsets[cluster_id + 1])
        rows = self.cluster_order[start:stop].astype(np.int64)
        tensor = torch.from_numpy(rows).to(self.device)
        self._cluster_ids_gpu_cache[cluster_id] = tensor
        return tensor

    def _composite_candidates(self, query: pd.DataFrame) -> dict[int, dict[int, float]]:
        left = query[["country", "phonetic_key_translit", "city_token_translit"]].copy()
        left["composite_key"] = (
            left["country"] + "|" + left["phonetic_key_translit"] + "|" + left["city_token_translit"]
        ).str.lower()
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
            if not row.business_name_translit or not row.city_token_translit:
                continue
            corpus_positions = self.exact_groups.get(
                (row.country, row.business_name_translit, row.city_token_translit), ()
            )
            if len(corpus_positions) <= self.config.max_candidates * 2:
                result[query_pos] = {
                    int(corpus_pos): self.config.exact_bonus
                    for corpus_pos in corpus_positions
                }
        return result

    def _ivf_candidates(self, query: pd.DataFrame) -> tuple[list[list[int]], list[list[float]]]:
        text = query["business_name_translit"] + " " + query["business_address_translit"]
        coarse = self._coarse_vectors(text)
        q = torch.from_numpy(coarse).to(self.device)
        sims = q @ self.centroids.T
        n_probe = min(self.config.ivf_n_probe, self.n_clusters)
        _, probe_idx = torch.topk(sims, n_probe, dim=1)
        probe_idx_cpu = probe_idx.cpu().numpy()  # one small sync: n_query*n_probe ints, not per-cluster

        n_query = len(query)
        cluster_to_queries: dict[int, list[int]] = {}
        for qi in range(n_query):
            for c in probe_idx_cpu[qi]:
                cluster_to_queries.setdefault(int(c), []).append(qi)

        query_idx_chunks, corpus_id_chunks, score_chunks = [], [], []
        for cluster_id, qlist in cluster_to_queries.items():
            start, stop = int(self.cluster_offsets[cluster_id]), int(self.cluster_offsets[cluster_id + 1])
            if stop <= start:
                continue
            corpus_rows_t = self._cluster_row_ids(cluster_id)
            cluster_tensor = self._cluster_tensor(cluster_id)
            q_sub = q[qlist]
            sims_block = q_sub @ cluster_tensor.T
            k = min(self.config.ann_candidates, sims_block.shape[1])
            top_vals, top_idx = torch.topk(sims_block, k, dim=1)  # stays on device, no sync
            top_corpus_ids = corpus_rows_t[top_idx]
            qlist_t = torch.tensor(qlist, device=self.device).unsqueeze(1).expand(-1, k)
            query_idx_chunks.append(qlist_t.reshape(-1))
            corpus_id_chunks.append(top_corpus_ids.reshape(-1))
            score_chunks.append(top_vals.reshape(-1))

        accum_ids: list[list[int]] = [[] for _ in range(n_query)]
        accum_scores: list[list[float]] = [[] for _ in range(n_query)]
        if not query_idx_chunks:
            return accum_ids, accum_scores

        all_query_idx = torch.cat(query_idx_chunks).cpu().numpy()
        all_corpus_id = torch.cat(corpus_id_chunks).cpu().numpy()
        all_scores = torch.cat(score_chunks).cpu().numpy()

        order = np.argsort(all_query_idx, kind="stable")
        sorted_qidx = all_query_idx[order]
        sorted_cid = all_corpus_id[order]
        sorted_scores = all_scores[order]
        boundaries = np.searchsorted(sorted_qidx, np.arange(n_query + 1))
        for qi in range(n_query):
            s, e = boundaries[qi], boundaries[qi + 1]
            accum_ids[qi] = sorted_cid[s:e].tolist()
            accum_scores[qi] = sorted_scores[s:e].tolist()
        return accum_ids, accum_scores

    def candidates(self, query: pd.DataFrame) -> pd.DataFrame:
        query = query.reset_index(drop=True)
        query = add_translit_columns(query, self.device)
        composite = self._composite_candidates(query)
        exact = self._exact_candidates(query)
        fine_matrix, query_norms = self.transform(query)
        accum_ids, _ = self._ivf_candidates(query)

        rows = []
        for query_pos in range(len(query)):
            candidate_ids = set(accum_ids[query_pos])
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
