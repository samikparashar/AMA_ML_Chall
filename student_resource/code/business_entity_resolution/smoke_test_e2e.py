"""End-to-end correctness smoke test: block -> features -> train -> threshold sweep.

Runs on a small subset purely to catch runtime errors and confirm the full
stage3-equivalent flow executes cleanly. Not meant to produce a meaningful
F_0.5 score (the corpus subset won't contain most true matches).
"""
import sys
import time

import pandas as pd

sys.path.insert(0, "/Users/Maruti/AMA_ML_Chall/student_resource/code/business_entity_resolution")
from src.blocking import BlockingConfig, BlockingIndex, build_corpus
from src.evaluate import load_ground_truth, blocking_report
from src.features import build_pair_features
from src.matching import label_pairs, train_classifier, sweep_threshold, build_outputs
from src.normalize import add_normalized_columns

root = "/Users/Maruti/AMA_ML_Chall/student_resource/dataset/train"
CORPUS_ROWS = 100_000
QUERY_ROWS = 3_000


def load_subset(path, n):
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, nrows=n)
    return add_normalized_columns(frame)


t0 = time.time()
s2 = load_subset(f"{root}/train_source2.tsv", CORPUS_ROWS // 2)
s3 = load_subset(f"{root}/train_source3.tsv", CORPUS_ROWS // 2)
s1 = load_subset(f"{root}/train_source1.tsv", QUERY_ROWS)
corpus = build_corpus(s2, s3)
truth = load_ground_truth(f"{root}/train_ground_truth.tsv")
print(f"[1/7] loaded {len(corpus)} corpus rows, {len(s1)} query rows in {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
config = BlockingConfig()
index = BlockingIndex(corpus, config)
print(f"[2/7] index build: {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
candidates = index.candidates(s1.reset_index(drop=True))
print(f"[3/7] blocking: {time.time()-t0:.1f}s, {len(candidates)} candidate rows", flush=True)

report = blocking_report(s1, corpus, candidates, truth)
print(f"[3b/7] blocking report (subset corpus, expect low recall - most true matches excluded): {report}", flush=True)

t0 = time.time()
features = build_pair_features(s1, corpus, candidates)
print(f"[4/7] features: {time.time()-t0:.1f}s, shape {features.shape}, columns {list(features.columns)}", flush=True)

t0 = time.time()
labeled = label_pairs(features, truth)
print(f"[5/7] labeled: {time.time()-t0:.1f}s, positive rate {labeled.label.mean():.4f}", flush=True)

t0 = time.time()
model, columns, valid = train_classifier(labeled)
print(f"[6/7] trained: {time.time()-t0:.1f}s, valid set size {len(valid)}", flush=True)

t0 = time.time()
threshold, score = sweep_threshold(model, columns, valid, truth)
print(f"[7/7] threshold sweep: {time.time()-t0:.1f}s -> best threshold={threshold:.2f}, macro F_0.5={score:.4f}", flush=True)

t0 = time.time()
build_outputs(s1, corpus, candidates, features, model, columns, threshold, "/tmp/smoke_test_output")
print(f"[extra] build_outputs: {time.time()-t0:.1f}s, wrote to /tmp/smoke_test_output", flush=True)

print("SMOKE TEST PASSED: full pipeline ran with no errors", flush=True)
