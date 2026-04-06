"""Run reproducible EEG regression benchmarks over subject-level feature variants.

Design philosophy: keep the default run small, reproducible, and easy to explain
in interviews, while preserving broader model and feature ablations as opt-in
extensions.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from src.feature_variants import ALL_FEATURE_VARIANTS, FEATURE_VARIANTS, build_feature_variant
    from src.split_manifest import (
        fold_counts,
        generate_split_manifest,
        load_split_manifest,
        save_split_manifest,
        validate_split_manifest,
    )
    from src.train_lemon_multitarget import (
        ATTENTION_TARGET_COLUMN,
        EXECUTIVE_FUNCTION_TARGET_COLUMN,
        INTELLIGENCE_TARGET_COLUMN,
        PROJECT_ROOT,
        TARGET_ALIASES,
        WORKING_MEMORY_TARGET_COLUMN,
        build_dataset,
        find_subject_file_pairs,
        load_metadata_and_targets,
        warn,
    )
except ImportError:
    from feature_variants import ALL_FEATURE_VARIANTS, FEATURE_VARIANTS, build_feature_variant
    from split_manifest import (
        fold_counts,
        generate_split_manifest,
        load_split_manifest,
        save_split_manifest,
        validate_split_manifest,
    )
    from train_lemon_multitarget import (
        ATTENTION_TARGET_COLUMN,
        EXECUTIVE_FUNCTION_TARGET_COLUMN,
        INTELLIGENCE_TARGET_COLUMN,
        PROJECT_ROOT,
        TARGET_ALIASES,
        WORKING_MEMORY_TARGET_COLUMN,
        build_dataset,
        find_subject_file_pairs,
        load_metadata_and_targets,
        warn,
    )

DEFAULT_SPLIT_PATH = PROJECT_ROOT / "processed" / "splits" / "benchmark_v1_splits.csv"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "benchmark_v1"
DEFAULT_N_SPLITS = 5
DEFAULT_RANDOM_STATE = 42
DEFAULT_EXPERIMENT_NAME = "benchmark_v1"
DEFAULT_LOGRATIO_EPS = 1e-6
ALL_MODEL_NAMES = ["dummy", "ridge", "elasticnet", "random_forest", "hist_gb"]
DEFAULT_MODEL_NAMES = ["dummy", "ridge", "random_forest"]
DEFAULT_SUITE_EXPLANATION = (
    "dummy as the floor, ridge as the linear baseline, and random_forest as the "
    "stronger nonlinear tabular baseline across the three public EO/EC variants"
)
FOLD_RESULTS_COLUMNS = [
    "experiment_name",
    "run_id",
    "fold",
    "model_name",
    "feature_variant",
    "target",
    "r2",
    "rmse",
    "mae",
    "pearson_r",
    "n_train",
    "n_test",
]
SUMMARY_RESULTS_COLUMNS = [
    "experiment_name",
    "run_id",
    "model_name",
    "feature_variant",
    "target",
    "r2_mean",
    "r2_std",
    "rmse_mean",
    "rmse_std",
    "mae_mean",
    "mae_std",
    "pearson_r_mean",
    "pearson_r_std",
    "n_folds",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark v1: fixed-fold EEG regression benchmarking with an interview-friendly "
            "baseline suite by default and optional exploratory extensions. The default "
            "suite uses dummy, ridge, and random_forest across the three public EO/EC variants."
        )
    )
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--experiment-name", type=str, default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--allow-target-fallback", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODEL_NAMES),
        choices=ALL_MODEL_NAMES,
        help=(
            "One or more models to run. Defaults to the interview-friendly baseline suite: "
            "dummy, ridge, and random_forest. Extra models such as elasticnet and hist_gb "
            "are optional exploratory extensions."
        ),
    )
    parser.add_argument(
        "--feature-variants",
        nargs="+",
        default=list(FEATURE_VARIANTS),
        choices=ALL_FEATURE_VARIANTS,
        help=(
            "One or more feature variants to run. Defaults to the interview-friendly public "
            "suite: eo_ec_concat, eo_ec_concat_plus_regions, and "
            "eo_ec_concat_plus_diff_plus_regions. Additional variants remain available as "
            "optional exploratory extensions."
        ),
    )
    parser.add_argument(
        "--logratio-eps",
        type=float,
        default=DEFAULT_LOGRATIO_EPS,
        help=f"Epsilon used in log-ratio features log(EO+eps)-log(EC+eps) (default: {DEFAULT_LOGRATIO_EPS}).",
    )
    return parser.parse_args(argv)


def expected_target_columns() -> Dict[str, str]:
    expected = {
        "working_memory": WORKING_MEMORY_TARGET_COLUMN,
        "attention": ATTENTION_TARGET_COLUMN,
        "executive_function": EXECUTIVE_FUNCTION_TARGET_COLUMN,
        "intelligence": INTELLIGENCE_TARGET_COLUMN,
    }
    missing_expected = [name for name, col in expected.items() if not col]
    if missing_expected:
        raise RuntimeError(f"Missing configured target columns for: {missing_expected}")
    return {name: str(col) for name, col in expected.items()}


def build_subject_level_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, str]]:
    metadata_targets, selected_target_columns = load_metadata_and_targets()
    if metadata_targets.empty:
        raise RuntimeError("Metadata/target merge failed. Cannot prepare benchmark dataset.")

    subject_file_pairs = find_subject_file_pairs()
    if not subject_file_pairs:
        raise RuntimeError("No EO+EC subject pairs available. Cannot prepare benchmark dataset.")

    X, y_raw, features_df = build_dataset(metadata_targets, subject_file_pairs)
    if X.empty or y_raw.empty or features_df.empty or "ID" not in features_df.columns:
        raise RuntimeError("Failed to build benchmark dataset with features, targets, and IDs.")

    subject_ids = features_df["ID"].astype(str).str.strip().tolist()
    if not subject_ids:
        raise RuntimeError("Dataset contains no subject IDs.")
    if len(set(subject_ids)) != len(subject_ids):
        raise RuntimeError("Dataset subject IDs are not unique; expected one row per subject.")
    if not selected_target_columns:
        raise RuntimeError("Resolved target column mapping is empty.")

    return X.reset_index(drop=True), y_raw.reset_index(drop=True), subject_ids, {str(k): str(v) for k, v in selected_target_columns.items()}


def enforce_target_lock(resolved_target_columns: Dict[str, str], allow_target_fallback: bool) -> Dict[str, object]:
    expected = expected_target_columns()
    missing_aliases = sorted(set(expected.keys()) - set(resolved_target_columns.keys()))
    if missing_aliases:
        raise RuntimeError(f"Resolved target mapping missing aliases: {missing_aliases}")

    drift = {
        alias: {"expected": expected[alias], "resolved": resolved_target_columns.get(alias, "")}
        for alias in expected
        if resolved_target_columns.get(alias) != expected[alias]
    }
    if drift and not allow_target_fallback:
        raise RuntimeError(
            "Target-definition drift detected. "
            f"Differences: {drift}. Re-run with --allow-target-fallback only if intentional."
        )

    return {
        "expected_target_columns": expected,
        "resolved_target_columns": {alias: resolved_target_columns[alias] for alias in expected.keys()},
        "target_lock_passed": len(drift) == 0,
        "target_drift": drift,
        "allow_target_fallback": bool(allow_target_fallback),
    }


def init_run_outputs(results_root: Path, experiment_name: str) -> Tuple[Path, str, str]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{experiment_name}_{timestamp}"
    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_id, timestamp


def write_csv_skeleton(path: Path, columns: list[str]) -> None:
    pd.DataFrame(columns=columns).to_csv(path, index=False)


def write_fold_summary(path: Path, counts: pd.DataFrame, run_id: str, experiment_name: str) -> None:
    summary = counts.copy()
    summary.insert(0, "experiment_name", experiment_name)
    summary.insert(1, "run_id", run_id)
    summary.to_csv(path, index=False)


def write_metadata(path: Path, metadata: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def build_model(model_name: str, random_state: int):
    if model_name == "dummy":
        return DummyRegressor(strategy="mean"), {"wrapped_multioutput": False}
    if model_name == "ridge":
        estimator = Pipeline([("scaler", StandardScaler()), ("regressor", Ridge(alpha=1.0, random_state=random_state))])
        return estimator, {"wrapped_multioutput": False}
    if model_name == "elasticnet":
        base = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=random_state)),
            ]
        )
        return MultiOutputRegressor(base), {"wrapped_multioutput": True}
    if model_name == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=500,
            random_state=random_state,
            n_jobs=-1,
            max_features="sqrt",
        )
        return estimator, {"wrapped_multioutput": False}
    if model_name == "hist_gb":
        base = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.0,
            early_stopping=False,
            random_state=random_state,
        )
        return MultiOutputRegressor(base), {
            "wrapped_multioutput": True,
            "hyperparameters": {
                "loss": "squared_error",
                "learning_rate": 0.05,
                "max_iter": 300,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 20,
                "l2_regularization": 0.0,
                "early_stopping": False,
            },
            "dependency": "scikit-learn",
        }
    raise ValueError(f"Unsupported model: {model_name}")


def zscore_targets_train_only(y_train: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    # Fit target scaling on the train fold only so test-fold target statistics
    # never leak into training or make cross-validation look better than it is.
    means = y_train.mean(axis=0)
    stds = y_train.std(axis=0, ddof=0)
    safe_stds = stds.copy()
    for col in y_train.columns:
        value = float(safe_stds[col])
        if not np.isfinite(value) or value == 0.0:
            safe_stds[col] = 1.0

    y_train_scaled = (y_train - means) / safe_stds
    means_dict = {str(col): float(means[col]) for col in y_train.columns}
    stds_dict = {str(col): float(safe_stds[col]) for col in y_train.columns}
    return y_train_scaled, means_dict, stds_dict


def inverse_scale_predictions(y_pred_scaled: np.ndarray, target_order: List[str], means: Dict[str, float], stds: Dict[str, float]) -> np.ndarray:
    y_pred_raw = np.asarray(y_pred_scaled, dtype=float).copy()
    for idx, alias in enumerate(target_order):
        y_pred_raw[:, idx] = (y_pred_raw[:, idx] * float(stds[alias])) + float(means[alias])
    return y_pred_raw


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or y_pred.size < 2:
        return float("nan")
    if np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return float("nan")

    corr = np.corrcoef(y_true, y_pred)[0, 1]
    if not np.isfinite(corr):
        return float("nan")
    return float(corr)


def evaluate_models(
    X: pd.DataFrame,
    y_raw: pd.DataFrame,
    subject_ids: List[str],
    manifest: pd.DataFrame,
    selected_models: List[str],
    selected_feature_variants: List[str],
    random_state: int,
    logratio_eps: float,
    experiment_name: str,
    run_id: str,
) -> pd.DataFrame:
    X_by_id = X.copy()
    X_by_id.index = subject_ids

    y_by_id = y_raw.copy()
    y_by_id.index = subject_ids

    target_order = list(TARGET_ALIASES)
    rows: List[Dict[str, object]] = []

    for feature_variant in selected_feature_variants:
        print(f"[INFO] Running feature variant: {feature_variant}")
        X_variant = build_feature_variant(X_by_id, variant=feature_variant, eps=logratio_eps)

        for fold in sorted(manifest["fold"].unique().tolist()):
            fold_df = manifest.loc[manifest["fold"] == fold]
            train_ids = fold_df.loc[fold_df["split"] == "train", "id"].astype(str).tolist()
            test_ids = fold_df.loc[fold_df["split"] == "test", "id"].astype(str).tolist()

            if not train_ids or not test_ids:
                raise RuntimeError(f"Fold {fold} has empty train or test partition.")

            X_train = X_variant.loc[train_ids]
            X_test = X_variant.loc[test_ids]
            y_train_raw = y_by_id.loc[train_ids][target_order]
            y_test_raw = y_by_id.loc[test_ids][target_order]

            y_train_scaled, means, stds = zscore_targets_train_only(y_train_raw)

            for model_name in selected_models:
                estimator, _ = build_model(model_name=model_name, random_state=random_state)
                estimator.fit(X_train, y_train_scaled)
                y_pred_scaled = np.asarray(estimator.predict(X_test), dtype=float)
                if y_pred_scaled.ndim == 1:
                    y_pred_scaled = y_pred_scaled.reshape(-1, 1)

                y_pred_raw = inverse_scale_predictions(
                    y_pred_scaled=y_pred_scaled,
                    target_order=target_order,
                    means=means,
                    stds=stds,
                )

                for idx, target in enumerate(target_order):
                    y_true_target = y_test_raw[target].to_numpy(dtype=float)
                    y_pred_target = y_pred_raw[:, idx]
                    rows.append(
                        {
                            "experiment_name": experiment_name,
                            "run_id": run_id,
                            "fold": int(fold),
                            "model_name": model_name,
                            "feature_variant": feature_variant,
                            "target": target,
                            "r2": float(r2_score(y_true_target, y_pred_target)),
                            "rmse": float(np.sqrt(mean_squared_error(y_true_target, y_pred_target))),
                            "mae": float(mean_absolute_error(y_true_target, y_pred_target)),
                            "pearson_r": safe_pearson(y_true_target, y_pred_target),
                            "n_train": int(len(train_ids)),
                            "n_test": int(len(test_ids)),
                        }
                    )

    if not rows:
        return pd.DataFrame(columns=FOLD_RESULTS_COLUMNS)

    out = pd.DataFrame(rows)
    return out[FOLD_RESULTS_COLUMNS]


def summarize_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    if fold_results.empty:
        return pd.DataFrame(columns=SUMMARY_RESULTS_COLUMNS)

    grouped = fold_results.groupby(
        ["experiment_name", "run_id", "model_name", "feature_variant", "target"],
        as_index=False,
    )
    summary = grouped.agg(
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        pearson_r_mean=("pearson_r", "mean"),
        pearson_r_std=("pearson_r", "std"),
        n_folds=("fold", "nunique"),
    )
    return summary[SUMMARY_RESULTS_COLUMNS]


def print_human_readable_summary(summary_results: pd.DataFrame) -> None:
    """Print a compact end-of-run summary that is easy to read in the terminal."""
    if summary_results.empty:
        print("\n[INFO] Human-readable summary: no benchmark rows were produced.")
        return

    overall_best = summary_results.sort_values(by="r2_mean", ascending=False).iloc[0]
    best_per_target = (
        summary_results.sort_values(by=["target", "r2_mean"], ascending=[True, False])
        .groupby("target", as_index=False)
        .first()
    )

    print("\n[INFO] Human-readable summary:")
    print(
        "- Best single target/model/variant row: "
        f"{overall_best['target']} with {overall_best['model_name']} + "
        f"{overall_best['feature_variant']} (mean CV R2={overall_best['r2_mean']:.3f})"
    )
    print("- Best row per target:")
    for _, row in best_per_target.iterrows():
        print(
            f"  - {row['target']}: {row['model_name']} + {row['feature_variant']} "
            f"| mean CV R2={row['r2_mean']:.3f}, mean Pearson r={row['pearson_r_mean']:.3f}"
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    X, y_raw, subject_ids, resolved_target_columns = build_subject_level_dataset()
    target_lock_info = enforce_target_lock(
        resolved_target_columns=resolved_target_columns,
        allow_target_fallback=args.allow_target_fallback,
    )

    should_generate = args.force_regenerate or not args.split_manifest.exists()
    if should_generate:
        manifest = generate_split_manifest(subject_ids=subject_ids, n_splits=args.n_splits, random_state=args.random_state)
        save_split_manifest(manifest, args.split_manifest)
        print(f"[INFO] Generated split manifest: {args.split_manifest}")
    else:
        print(f"[INFO] Using existing split manifest: {args.split_manifest}")

    manifest = load_split_manifest(args.split_manifest)
    try:
        validate_split_manifest(manifest, dataset_subject_ids=subject_ids)
    except Exception as exc:
        raise RuntimeError(f"Split manifest validation failed: {exc}") from exc

    unique_subjects = len(set(subject_ids))
    print(f"[INFO] Dataset subjects: {unique_subjects}")

    counts = fold_counts(manifest)
    if counts.empty:
        warn("No fold counts available after loading manifest.")
        return

    print("\nResolved target columns:")
    for alias, source_col in target_lock_info["resolved_target_columns"].items():
        print(f"- {alias}: {source_col}")

    print("\nPer-fold train/test counts:")
    for _, row in counts.iterrows():
        print(f"- Fold {int(row['fold'])}: train={int(row['train'])}, test={int(row['test'])}")

    run_dir, run_id, timestamp = init_run_outputs(args.results_root, args.experiment_name)
    fold_results_path = run_dir / "fold_results.csv"
    summary_results_path = run_dir / "summary_results.csv"
    split_summary_path = run_dir / "split_summary.csv"
    metadata_path = run_dir / "metadata.json"

    write_csv_skeleton(fold_results_path, columns=FOLD_RESULTS_COLUMNS)
    write_csv_skeleton(summary_results_path, columns=SUMMARY_RESULTS_COLUMNS)
    write_fold_summary(split_summary_path, counts=counts, run_id=run_id, experiment_name=args.experiment_name)

    selected_models = list(dict.fromkeys(args.models))
    selected_feature_variants = list(dict.fromkeys(args.feature_variants))

    if selected_models == DEFAULT_MODEL_NAMES and selected_feature_variants == FEATURE_VARIANTS:
        print(
            "\n[INFO] Running the interview-friendly default suite: "
            f"{DEFAULT_SUITE_EXPLANATION}."
        )
    else:
        print(
            "\n[INFO] Running a custom benchmark selection. Optional models and advanced "
            "feature variants are treated as exploratory extensions on top of the baseline suite."
        )

    print(f"[INFO] Models selected: {selected_models}")
    print(f"[INFO] Feature variants selected: {selected_feature_variants}")

    fold_results = evaluate_models(
        X=X,
        y_raw=y_raw,
        subject_ids=subject_ids,
        manifest=manifest,
        selected_models=selected_models,
        selected_feature_variants=selected_feature_variants,
        random_state=args.random_state,
        logratio_eps=args.logratio_eps,
        experiment_name=args.experiment_name,
        run_id=run_id,
    )
    if not fold_results.empty:
        fold_results.to_csv(fold_results_path, mode="a", header=False, index=False)

    summary_results = summarize_results(fold_results)
    summary_results.to_csv(summary_results_path, index=False)

    model_registry_info = {}
    for name in selected_models:
        _, info = build_model(model_name=name, random_state=args.random_state)
        model_registry_info[name] = info

    metadata = {
        "timestamp_utc": timestamp,
        "experiment_name": args.experiment_name,
        "run_id": run_id,
        "split_manifest_path": str(args.split_manifest.resolve()),
        "results_dir": str(run_dir.resolve()),
        "random_state": int(args.random_state),
        "requested_n_splits": int(args.n_splits),
        "effective_n_splits": int(counts["fold"].nunique()),
        "dataset_subject_count": int(unique_subjects),
        "target_definition": target_lock_info,
        "models_requested": selected_models,
        "feature_variants_requested": selected_feature_variants,
        "feature_variant_logratio_eps": float(args.logratio_eps),
        "model_registry": model_registry_info,
        "target_scaling_for_cv": {
            "method": "zscore_train_fold_only",
            "prediction_inverse_transform": "to_raw_target_space",
            "metrics_space": "raw",
        },
        "notes": {"report_style_prediction": "exploratory only"},
    }
    write_metadata(metadata_path, metadata)

    print("\n[INFO] Wrote benchmark outputs:")
    print(f"- metadata: {metadata_path}")
    print(f"- fold results: {fold_results_path}")
    print(f"- summary results: {summary_results_path}")
    print(f"- split summary: {split_summary_path}")
    print_human_readable_summary(summary_results)


if __name__ == "__main__":
    main()
