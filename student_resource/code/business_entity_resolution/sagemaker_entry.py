"""SageMaker training-job entry point.

Wraps the existing `src.pipeline` CLI unchanged. SageMaker's generic Estimator
passes hyperparameters as `--key value` flags (via the SM_HPS env var, JSON-encoded)
and has no concept of a positional argument, but `pipeline.py stage2/3/4` takes
`stage` positionally -- so this script reads SM_HPS itself, pulls out `stage`,
and reconstructs sys.argv before calling the real `main()`. No changes made to
pipeline.py or any src/ module.
"""
import json
import os
import sys
from urllib.parse import urlparse

import lightgbm  # noqa: F401  -- must import before torch, see handoff_2 section 2 (OpenMP segfault)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _s3_parts(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip("/")


def _download_cache(s3_uri: str, local_dir: str) -> None:
    """Pull a previously-built blocking cache down from a fixed S3 location, if one
    exists there yet. Every stage2/3/4 job passes the *same* --cache-s3 URI, so the
    cache only ever needs to be built once (by whichever job runs first with a given
    blocking config) -- later jobs restore it here before the pipeline runs, and
    pipeline.py's load_block_cache() then skips reclustering/reblocking as long as
    the blocking hyperparameters this run match exactly what produced the cache
    (load_block_cache compares config.json, see data_io.py).
    """
    import boto3

    bucket, prefix = _s3_parts(s3_uri)
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    found = False
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix):].lstrip("/")
            if not rel:
                continue
            dest = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            s3.download_file(bucket, key, dest)
            found = True
    if found:
        print(f"[sagemaker_entry] restored blocking cache from {s3_uri} to {local_dir}", flush=True)


def _upload_cache(local_dir: str, s3_uri: str) -> None:
    """Push this job's (possibly freshly-built) blocking cache back up to the same
    fixed S3 location, so the next job -- regardless of stage -- finds it there.
    Runs after the pipeline completes; if the job crashes/times out first, nothing
    is uploaded here (model_dir still gets tarred to model.tar.gz by SageMaker
    itself as a fallback, but that's per-job, not the shared cache location).
    """
    import boto3

    if not os.path.isdir(local_dir):
        return
    bucket, prefix = _s3_parts(s3_uri)
    s3 = boto3.client("s3")
    for root, _, files in os.walk(local_dir):
        for name in files:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, local_dir)
            key = f"{prefix.rstrip('/')}/{rel}"
            s3.upload_file(path, bucket, key)
    print(f"[sagemaker_entry] uploaded blocking cache from {local_dir} to {s3_uri}", flush=True)


def main():
    hps = json.loads(os.environ.get("SM_HPS", "{}"))
    stage = hps.pop("stage")
    cache_s3 = hps.pop("cache_s3", None)

    argv = [stage]
    for key, value in hps.items():
        argv.append(f"--{key.replace('_', '-')}")
        argv.append(str(value))

    # SageMaker's dedicated model-artifact dir -- gets auto-tarred to model.tar.gz in
    # the job's S3 output_path. Only stage4 (matching.py's save_model) reads --model-dir.
    model_dir = hps.get("model_dir") or os.environ.get("SM_MODEL_DIR")
    if "model_dir" not in hps and "SM_MODEL_DIR" in os.environ:
        argv += ["--model-dir", os.environ["SM_MODEL_DIR"]]

    cache_dir = os.path.join(model_dir, "candidates_cache") if model_dir else None
    if cache_dir and cache_s3:
        _download_cache(cache_s3, cache_dir)

    sys.argv = ["pipeline.py"] + argv
    print(f"[sagemaker_entry] invoking: {sys.argv}", flush=True)

    from src.pipeline import main as pipeline_main
    pipeline_main()

    if cache_dir and cache_s3:
        _upload_cache(cache_dir, cache_s3)


if __name__ == "__main__":
    main()
