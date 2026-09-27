# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary
*Provide a brief 2-3 sentence overview of your approach and key innovations.*

---

## 2. Methodology

### 2.1 Problem Analysis
*Key insights discovered during EDA — noise patterns, address variations, missing fields, etc.*

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier.
**Core Innovation:** A two-stage blocking pipeline (coarse ANN/IVF shortlist, then a precise cosine + magnitude-similarity rerank) with a dedicated GPU-batched transliteration stage so cross-script true matches (e.g. Latin ↔ Devanagari/Tamil/Gujarati/Kannada/Bengali/Telugu/Malayalam/Oriya/Gurmukhi business names) share real n-gram overlap instead of zero. Two interchangeable backends implement the identical blocking interface: a local Metal/MPS (torch) backend for development, and a CUDA (FAISS) backend for production-scale/SageMaker runs.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** composite-key exact match, exact name/city match, hashed character n-gram (char_wb) TF-IDF cosine similarity, IVF-clustered approximate nearest-neighbor search for scale.
- **Candidate pairs generated:** [pending full-scale run — see `handoffs/` for medium-scale validation numbers]
- **How true matches were not lost:** a dedicated Latin accent-folding fix (NFKD decompose + strip combining diacritic marks, scoped to not interfere with Indic script handling) and the transliteration stage above were added specifically after real validation runs surfaced true matches that raw character-hashing alone could not find. Blocking recall on a medium-scale (non-final) validation run reached 85.46% after tuning `max_candidates`/`ivf_n_probe`; full-scale recall has not yet been measured.

---

## 4. Matching Model

**Features used:**
- Name features: exact match, token-sort/token-set/partial/plain rapidfuzz ratios, phonetic-key exact match, name length delta.
- Address features: token-sort/partial/plain rapidfuzz ratios, digit-set Jaccard (numeric tokens in the address, e.g. PIN/ZIP codes and building numbers), address length delta.
- Other: country exact match, city-token exact match, and the blocking stage's own scores (hashed TF-IDF cosine, composite-key match flag, exact-block-match flag) fed back in as classifier features.

**Model type:** LightGBM binary classifier (gradient-boosted decision trees), MIT-licensed, well under the 8B-parameter constraint.
**Threshold selection method:** linear sweep (0.01 steps) directly optimizing macro F₀.₅ on an S1-group-aware held-out validation split (no S1 group leaked across train/validation).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [pending full-scale run — a medium-scale (20k-query, 500k-corpus, not the full ~2.2M/~10.3M dataset) validation run reached 0.9021; this is not the final submission number and should not be reported as such]
- **Common false positives (wrong merges):** [pending full-scale error analysis]
- **Common false negatives (missed matches):** two real patterns identified during medium-scale validation: (1) genuinely noisy source data (e.g. one true match had a completely empty address field; another had an apparent digit-entry typo in the street number) — not fixable via blocking/model changes; (2) domain-name-style business name variants (e.g. `"primemoney.com"` vs `"Prime Money"`) not yet specifically addressed.

---

## 6. Conclusion
*Summarize your approach, key achievements, and lessons learned in 2-3 sentences.*

---

## Appendix

### A. Code Artefacts
*Your complete, runnable code ships in the submission zip under
`code/business_entity_resolution/` (all source in `src/`, with a `README.md` and
`requirements.txt`). Summarise its structure and the entry point(s) to reproduce
`output/matching_results.tsv` and `output/candidate_pairs.tsv` here.*

### B. Additional Results
*Include any additional charts, graphs, or detailed results.*

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
