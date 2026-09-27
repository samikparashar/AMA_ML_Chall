"""CLI entry point for normalization, blocking, evaluation, and matching."""

import argparse
from pathlib import Path

import pandas as pd

from .data_io import build_candidate_map, load_block_cache, load_full_tsv, save_block_cache, write_id_lists
from .evaluate import blocking_report, load_ground_truth
from .features import build_pair_features
from .matching import build_outputs, label_pairs, save_model, sweep_threshold, train_classifier


def _paths(root: Path, split: str):
    folder = root / "dataset" / split
    prefix = "train" if split == "train" else "test"
    return [folder / f"{prefix}_source{i}.tsv" for i in (1, 2, 3)]


def stage1(args):
    frame = load_full_tsv(args.source1).head(args.rows)
    print(frame[["business_name", "business_name_norm", "business_address", "business_address_norm", "city_token", "phonetic_key"]].to_string(index=False))


def _backend(name: str):
    if name == "faiss":
        from .blocking_faiss import FaissBlockingConfig, FaissBlockingIndex, build_corpus
        return FaissBlockingConfig, FaissBlockingIndex, build_corpus
    from .blocking import BlockingConfig, BlockingIndex, build_corpus  # imports lightgbm before torch internally
    return BlockingConfig, BlockingIndex, build_corpus


def _run_block(args, split: str):
    config_cls, index_cls, build_corpus = _backend(args.backend)
    values = {key: getattr(args, key) for key in config_cls.__annotations__ if hasattr(args, key)}
    cache_dir = Path(args.model_dir) / "candidates_cache" / f"{split}_{args.backend}"
    cached = load_block_cache(cache_dir, values)
    if cached is not None:
        print(f"loaded cached blocking result from {cache_dir} (skipped re-clustering/re-blocking)")
        return cached
    s1_path, s2_path, s3_path = _paths(Path(args.root), split)
    s1, s2, s3 = (load_full_tsv(path) for path in (s1_path, s2_path, s3_path))
    config = config_cls(**values)
    corpus = build_corpus(s2, s3)
    index = index_cls(corpus, config)
    pieces = []
    for start in range(0, len(s1), config.batch_size):
        query = s1.iloc[start:start + config.batch_size]
        piece = index.candidates(query)
        piece["query_pos"] += start
        pieces.append(piece)
        print(f"blocked {min(start + config.batch_size, len(s1))}/{len(s1)}")
    candidates = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    save_block_cache(cache_dir, s1, corpus, candidates, values)
    return s1, corpus, candidates


def stage2(args):
    s1, corpus, candidates = _run_block(args, args.split)
    if args.ground_truth:
        print(blocking_report(s1, corpus, candidates, load_ground_truth(args.ground_truth)))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    mapping = build_candidate_map(s1, corpus, candidates)
    write_id_lists(output / "candidate_pairs.tsv", s1.entity_id, mapping, "candidate_entity_ids")


def stage3(args):
    s1, corpus, candidates = _run_block(args, "train")
    truth = load_ground_truth(args.ground_truth)
    features = label_pairs(build_pair_features(s1, corpus, candidates), truth)
    model, columns, valid = train_classifier(features)
    threshold, score = sweep_threshold(model, columns, valid, truth)
    print(f"best threshold={threshold:.2f}, validation macro F_0.5={score:.4f}")
    save_model(model, columns, threshold, args.model_dir)


def stage4(args):
    train_s1, train_corpus, train_candidates = _run_block(args, "train")
    truth = load_ground_truth(args.ground_truth)
    train_features = label_pairs(build_pair_features(train_s1, train_corpus, train_candidates), truth)
    validation_model, columns, valid = train_classifier(train_features)
    threshold, score = sweep_threshold(validation_model, columns, valid, truth)
    save_model(validation_model, columns, threshold, f"{args.model_dir}/validation")
    model, columns, _ = train_classifier(train_features, validation_fraction=0.0)
    print(f"selected threshold={threshold:.2f}, validation macro F_0.5={score:.4f}")
    save_model(model, columns, threshold, args.model_dir)
    del train_features, train_candidates, train_corpus, train_s1, validation_model, valid
    s1, corpus, candidates = _run_block(args, "test")
    test_features = build_pair_features(s1, corpus, candidates)
    build_outputs(s1, corpus, candidates, test_features, model, columns, threshold, args.output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["stage1", "stage2", "stage3", "stage4"])
    parser.add_argument("--backend", default="torch", choices=["torch", "faiss"],
                        help="torch: MPS-accelerated IVF, for this Mac. faiss: FAISS IVF, "
                             "auto-detects and uses CUDA GPU when available (SageMaker).")
    parser.add_argument("--root", default=".")
    parser.add_argument("--source1")
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--ground-truth")
    parser.add_argument("--output", default="output")
    parser.add_argument("--model-dir", dest="model_dir", default="model")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=10000)
    parser.add_argument("--max-candidates", dest="max_candidates", type=int, default=300)
    parser.add_argument("--n-features", dest="n_features", type=int, default=2**20)
    parser.add_argument("--max-df-frac", dest="max_df_frac", type=float, default=0.005)
    parser.add_argument("--ann-dim", dest="ann_dim", type=int, default=256)
    parser.add_argument("--ann-candidates", dest="ann_candidates", type=int, default=300)
    parser.add_argument("--ivf-n-clusters", dest="ivf_n_clusters", type=int, default=2048)
    parser.add_argument("--ivf-n-probe", dest="ivf_n_probe", type=int, default=64)
    parser.add_argument("--ivf-sample-size", dest="ivf_sample_size", type=int, default=300_000)
    parser.add_argument("--hybrid-alpha", dest="hybrid_alpha", type=float, default=0.7)
    parser.add_argument("--threshold", type=float, default=0.68)
    args = parser.parse_args()
    if args.stage == "stage1":
        if not args.source1:
            parser.error("stage1 requires --source1")
        stage1(args)
    elif args.stage == "stage2":
        stage2(args)
    elif args.stage == "stage3":
        stage3(args)
    else:
        stage4(args)


if __name__ == "__main__":
    main()
