# Business Entity Resolution

## Pipeline

1. **Normalize** (`src/normalize.py`) — NFKC fold, Latin accent-folding (NFKD decompose + strip combining marks in the Latin diacritic block, so `"Thómas"` and `"Thomas"` normalize identically — does not touch the separately-preserved Devanagari-through-Malayalam script range), legal-suffix / address-abbreviation canonicalization, city-token extraction, phonetic key (metaphone for ASCII names, a manual vowel-collapse fallback otherwise).
2. **Transliterate** (`src/transliterate.py`) — adds parallel `*_translit` columns (GPU-batched, MPS locally / CUDA on SageMaker) that map Devanagari/Tamil/Gujarati/Kannada/Bengali/Telugu/Malayalam/Oriya/Gurmukhi script to a Latin approximation, so a cross-script true match (e.g. `"Surya Corp"` vs. `"सूर्य कॉर्प"`) shares real n-gram overlap instead of zero. The original `_norm` fields are left untouched; blocking uses the `_translit` columns.
3. **Block / generate candidates** — two backends, same public interface (`candidates(query) -> DataFrame`):
   - `src/blocking.py` (`--backend torch`, default): local Mac development, Metal/MPS-accelerated. Two stages: a coarse k-means/IVF shortlist over a small dense vector, then a precise cosine + magnitude-similarity rerank over the full hashed high-dimensional vector.
   - `src/blocking_faiss.py` (`--backend faiss`): SageMaker GPU only. Same two-stage design, using FAISS's native `IndexIVFFlat` (auto-detects and uses CUDA when available). This is the backend used for any real-scale/SageMaker run.
   - Both also do a composite-key exact match and an exact-name/city match ahead of the ANN search.
4. **Feature engineering** (`src/features.py`) — rapidfuzz name/address similarity ratios, exact-match flags, digit-set Jaccard, plus the blocking-stage scores, all fed to the classifier as features.
5. **Train + threshold** (`src/matching.py`) — LightGBM binary classifier on an S1-group-aware 80/20 split (no group leaked across train/valid), then a linear threshold sweep (0.01 steps) optimizing macro F₀.₅ directly on the validation split.
6. **Output** — `matching_results.tsv` (scored) and `candidate_pairs.tsv` (pre-classifier candidate set), validated with `student_resource/utils/validate_submission.py`.

## Model / candidate persistence

- `save_model()` / `load_model()` (`src/matching.py`) persist the trained LightGBM model (native `booster_.save_model()` format) plus a small JSON sidecar recording the feature-column order and the selected threshold. Both `stage3` and `stage4` save a model into `--model-dir` (default `model/`); `stage4` additionally saves the pre-final validation-split model into `{model-dir}/validation/`.
- **Candidate/clustering caching is automatic, not opt-in.** `_run_block()` (`src/pipeline.py`) — the function every stage uses to build the blocking index and generate candidates — checks for a cache at `{model-dir}/candidates_cache/{split}_{backend}/` before doing any work, and saves to it afterward. The cache is invalidated automatically if any blocking hyperparameter differs from what produced it (compared via a `config.json` written alongside the cached data), so changing `--max-candidates`/`--ivf-n-probe`/etc. is always safe — it just triggers a real recompute rather than silently reusing stale candidates. This means: build the blocking index/candidates once for a given config, then re-run `stage3` as many times as needed (different classifier tuning) without repeating the expensive clustering/blocking step, as long as `--model-dir` and the blocking config stay the same.

## Running

From `student_resource/`:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r code/business_entity_resolution/requirements.txt
# On a SageMaker/CUDA GPU instance, also: pip install -r code/business_entity_resolution/requirements-sagemaker.txt

PYTHONPATH=code/business_entity_resolution python -m src.pipeline stage1 --source1 dataset/train/train_source1.tsv --rows 20
PYTHONPATH=code/business_entity_resolution python -m src.pipeline stage2 --root . --split train --ground-truth dataset/train/train_ground_truth.tsv
PYTHONPATH=code/business_entity_resolution python -m src.pipeline stage3 --root . --ground-truth dataset/train/train_ground_truth.tsv
PYTHONPATH=code/business_entity_resolution python -m src.pipeline stage4 --root . --ground-truth dataset/train/train_ground_truth.tsv --output output
python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

Stage 1 prints the required manual inspection columns. Stage 2 reports blocking recall and candidate distribution when ground truth is supplied. Stage 3 uses an S1-group-aware split for threshold selection and saves a model but writes no other files. Stage 4 retrains on 100% of train data, writes complete test coverage (including empty rows for entities with no candidates), and saves the final model.

Add `--backend faiss` on a SageMaker/CUDA GPU instance for the production blocking path; the default `--backend torch` (Metal/MPS) is for local Mac development only.

### Key CLI flags and current defaults

All hyperparameters are exposed on every stage's CLI (`configs/default.yaml` mirrors these but is **not** automatically loaded — CLI defaults are what actually takes effect unless you pass a flag explicitly):

| Flag | Default | Notes |
|---|---|---|
| `--backend` | `torch` | `torch` (local MPS) or `faiss` (SageMaker CUDA) |
| `--model-dir` | `model` | Where the trained model and the candidates cache are written/read |
| `--output` | `output` | Where the two competition TSVs are written (kept separate from `--model-dir`) |
| `--batch-size` | `10000` | S1 query batch size for blocking |
| `--max-candidates` | `300` | Max candidates kept per query after Stage-B rerank |
| `--ivf-n-probe` | `64` | IVF clusters probed per query (recall/speed tradeoff) |
| `--ivf-n-clusters` | `2048` | Clamped internally to `min(configured, corpus_size // 10)` for small corpora |
| `--ann-dim` | `256` | Dense vector dimension for the coarse Stage-A search |
| `--ann-candidates` | `300` | Shortlist size handed from Stage A to Stage B |
| `--n-features` | `2^20` | Hashed sparse vector dimensionality for Stage-B cosine rerank |
| `--hybrid-alpha` | `0.7` | Weight on cosine vs. magnitude-similarity in the Stage-B hybrid score |

`max_candidates=300`/`ivf_n_probe=64` were last tuned against `max_candidates=150`/`ivf_n_probe=8` on a real (though medium-scale, not full-scale) validation run — recall improved 71.04% → 84.69% → 85.46% across `50/8` → `150/64` → `300/64`, with `150→300` showing clearly diminishing returns for meaningfully more cost. **Whether `150/64` is actually the better tradeoff at full scale is still an open, unresolved question** — see `handoffs/handoff_5.md`.

## SageMaker

`sagemaker_entry.py` (this directory) is a thin wrapper that lets a SageMaker training job invoke this same CLI — it reads the job's hyperparameters (JSON via `SM_HPS`), reconstructs the equivalent `sys.argv` (including the positional `stage` argument, which SageMaker's hyperparameter mechanism can't express directly), and auto-wires `--model-dir` to `SM_MODEL_DIR` when running on SageMaker so the trained model and candidates cache land in the job's `model.tar.gz` output automatically. No changes to `pipeline.py`'s own CLI semantics. See `handoffs/handoff_2.md` onward for the AWS setup this depends on (IAM role, S3 bucket, GPU quota) — those are operational/account-specific and not part of this reproducible package.
