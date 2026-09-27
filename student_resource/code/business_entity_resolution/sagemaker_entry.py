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

import lightgbm  # noqa: F401  -- must import before torch, see handoff_2 section 2 (OpenMP segfault)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    hps = json.loads(os.environ.get("SM_HPS", "{}"))
    stage = hps.pop("stage")

    argv = [stage]
    for key, value in hps.items():
        argv.append(f"--{key.replace('_', '-')}")
        argv.append(str(value))

    sys.argv = ["pipeline.py"] + argv
    print(f"[sagemaker_entry] invoking: {sys.argv}", flush=True)

    from src.pipeline import main as pipeline_main
    pipeline_main()


if __name__ == "__main__":
    main()
