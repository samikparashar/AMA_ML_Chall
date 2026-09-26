# Business Entity Resolution

The pipeline keeps raw fields intact and adds normalized fields, then builds one full S2/S3 sparse index reused across S1 batches. Blocking is the union of a vectorized composite-key merge and structurally pruned hashed character n-gram retrieval. Candidate pairs are scored with rapidfuzz features and a LightGBM binary classifier.

From `student_resource/`:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r code/business_entity_resolution/requirements.txt
PYTHONPATH=code/business_entity_resolution python -m src.pipeline stage1 --source1 dataset/train/train_source1.tsv --rows 20
PYTHONPATH=code/business_entity_resolution python -m src.pipeline stage2 --root . --split train --ground-truth dataset/train/train_ground_truth.tsv
PYTHONPATH=code/business_entity_resolution python -m src.pipeline stage3 --root . --ground-truth dataset/train/train_ground_truth.tsv
PYTHONPATH=code/business_entity_resolution python -m src.pipeline stage4 --root . --ground-truth dataset/train/train_ground_truth.tsv --output output
python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

Stage 1 prints the required manual inspection columns. Stage 2 reports blocking recall and candidate distribution when ground truth is supplied. Stage 3 uses an S1-group-aware split for threshold selection. Stage 4 writes complete test coverage, including empty rows for entities with no candidates.
