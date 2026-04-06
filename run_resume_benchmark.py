"""Recommended interview/demo entrypoint for the smallest EEG benchmark run.

This wrapper keeps the CLI intentionally small and always delegates to
`src/benchmark_v1.py` with the resume-friendly baseline suite:

- models: dummy, ridge, random_forest
- feature variants: eo_ec_concat, eo_ec_concat_plus_regions,
  eo_ec_concat_plus_diff_plus_regions

That gives new users one obvious command to run while preserving the same
benchmark outputs, metadata, and split-handling behavior as `src/benchmark_v1.py`.

Example:
    python run_resume_benchmark.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

RESUME_MODELS = ["dummy", "ridge", "random_forest"]
RESUME_FEATURE_VARIANTS = [
    "eo_ec_concat",
    "eo_ec_concat_plus_regions",
    "eo_ec_concat_plus_diff_plus_regions",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the recommended interview-friendly benchmark suite with the default "
            "baseline models and public EO/EC feature variants."
        )
    )
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--experiment-name", type=str)
    parser.add_argument("--n-splits", type=int)
    parser.add_argument("--random-state", type=int)
    parser.add_argument("--force-regenerate", action="store_true")
    return parser.parse_args(argv)


def build_benchmark_argv(args: argparse.Namespace) -> List[str]:
    forwarded_args: List[str] = [
        "--models",
        *RESUME_MODELS,
        "--feature-variants",
        *RESUME_FEATURE_VARIANTS,
    ]

    if args.split_manifest is not None:
        forwarded_args.extend(["--split-manifest", str(args.split_manifest)])
    if args.results_root is not None:
        forwarded_args.extend(["--results-root", str(args.results_root)])
    if args.experiment_name is not None:
        forwarded_args.extend(["--experiment-name", args.experiment_name])
    if args.n_splits is not None:
        forwarded_args.extend(["--n-splits", str(args.n_splits)])
    if args.random_state is not None:
        forwarded_args.extend(["--random-state", str(args.random_state)])
    if args.force_regenerate:
        forwarded_args.append("--force-regenerate")

    return forwarded_args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    from src import benchmark_v1

    print(
        "[INFO] Resume benchmark entrypoint: running the recommended benchmark-first "
        "suite from src/benchmark_v1.py."
    )
    benchmark_v1.main(build_benchmark_argv(args))


if __name__ == "__main__":
    main()
