"""Composite-key, exact-match, and FAISS IVF blocking for CUDA/GPU hardware
(e.g. an AWS SageMaker GPU instance).

Unlike the torch/MPS backend in blocking.py (built for this Mac's Metal GPU,
which has no compatible graph-ANN library), FAISS's IndexIVFFlat is a single
native (C++/CUDA) call for the whole coarse-search stage per batch -- no
per-cluster Python loop, no per-iteration dispatch overhead. index.search()
does the sub-linear cluster-probe search entirely inside FAISS itself.

Device selection is automatic: faiss.get_num_gpus() decides whether the IVF
index lives on GPU or CPU, so the same code runs (slower) on a CPU-only
machine and (fast) on a CUDA GPU without any changes. This module has not
been runtime-tested on this Mac -- it has no CUDA GPU, and faiss-gpu will not
even install without one. It must be verified on the actual SageMaker GPU
instance before being trusted for a real run.

Stage B (precise rerank) is identical to blocking.py: exact cosine on the
full high-dimensional IDF-weighted sparse vectors for the shortlist, blended
with a magnitude-similarity term.
"""

from dataclasses import dataclass

import lightgbm  # noqa: F401 -- see blocking.py: must load before any GPU/BLAS-heavy
                 # library that could bundle a conflicting OpenMP runtime.
import faiss
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as l2_normalize

from .transliterate import add_translit_columns


@dataclass
class FaissBlockingConfig:
    batch_size: int = 10000
    max_candidates: int = 50
    n_features: int = 2**20
    max_df_frac: float = 0.005
    composite_bonus: float = 0.5
    exact_bonus: float = 0.35
    ann_dim: int = 256
    ann_candidates: int = 300
    ivf_n_clusters: int = 4096
    ivf_n_probe: int = 16
    ivf_sample_size: int = 300_000
    hybrid_alpha: float = 0.7
    use_gpu: bool | None = None  # None = auto-detect via faiss.get_num_gpus()


class FaissBlockingIndex:
    def __init__(self, corpus: pd.DataFrame, config: FaissBlockingConfig | None = None):
        self.config = config or FaissBlockingConfig()
        self.corpus = corpus.reset_index(drop=True).copy()
        self.corpus = add_translit_columns(self.corpus)
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
        self._build_faiss_index(text)
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
        frame = add_translit_columns(frame)
        text = frame["business_name_translit"] + " " + frame["business_address_translit"]
        matrix = self.vectorizer.transform(text).tocsr(copy=False)
        matrix.data *= np.take(self.idf, matrix.indices).astype(np.float32)
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
        normalized = l2_normalize(matrix, norm="l2", axis=1, copy=False).tocsr()
        return normalized, np.maximum(norms, 1e-6).astype(np.float32)

    def _coarse_vectors(self, text: pd.Series) -> np.ndarray:
        dense = self.coarse_vectorizer.transform(text).toarray().astype(np.float32)
        return np.ascontiguousarray(l2_normalize(dense, norm="l2", axis=1))

    def _build_faiss_index(self, corpus_text: pd.Series) -> None:
        n_clusters = max(1, min(self.config.ivf_n_clusters, self.n_corpus // 10 or 1))
        quantizer = faiss.IndexFlatIP(self.config.ann_dim)
        index = faiss.IndexIVFFlat(quantizer, self.config.ann_dim, n_clusters, faiss.METRIC_INNER_PRODUCT)

        n_gpus = faiss.get_num_gpus()
        use_gpu = self.config.use_gpu if self.config.use_gpu is not None else n_gpus > 0
        self.using_gpu = use_gpu
        if use_gpu:
            self.gpu_resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, index)

        sample_n = min(self.config.ivf_sample_size, self.n_corpus)
        rng = np.random.default_rng(0)
        sample_pos = rng.choice(self.n_corpus, size=sample_n, replace=False)
        sample_vectors = self._coarse_vectors(corpus_text.iloc[sample_pos])
        index.train(sample_vectors)

        chunk = 500_000
        for start in range(0, self.n_corpus, chunk):
            stop = min(start + chunk, self.n_corpus)
            vectors = self._coarse_vectors(corpus_text.iloc[start:stop])
            index.add(vectors)

        index.nprobe = min(self.config.ivf_n_probe, n_clusters)
        self.index = index
        self.n_clusters = n_clusters

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

    def candidates(self, query: pd.DataFrame) -> pd.DataFrame:
        query = query.reset_index(drop=True)
        query = add_translit_columns(query)
        composite = self._composite_candidates(query)
        exact = self._exact_candidates(query)
        fine_matrix, query_norms = self.transform(query)
        text = query["business_name_translit"] + " " + query["business_address_translit"]
        coarse = self._coarse_vectors(text)
        k = min(self.config.ann_candidates, self.n_corpus)
        _, labels = self.index.search(coarse, k)  # single native call for the whole batch

        rows = []
        for query_pos in range(len(query)):
            candidate_ids = set(int(c) for c in labels[query_pos] if c >= 0)
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
