"""Create, validate, and summarize subject-level benchmark split manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Set

import pandas as pd
from sklearn.model_selection import KFold

REQUIRED_COLUMNS = ["id", "fold", "split"]
VALID_SPLITS = {"train", "test"}


def generate_split_manifest(subject_ids: Sequence[str], n_splits: int, random_state: int) -> pd.DataFrame:
    unique_ids = sorted({str(subject_id).strip() for subject_id in subject_ids if str(subject_id).strip()})
    if len(unique_ids) < 2:
        raise ValueError("Need at least 2 unique subject IDs to create KFold splits.")

    effective_splits = min(n_splits, len(unique_ids))
    if effective_splits < 2:
        raise ValueError("Effective n_splits must be >= 2.")

    kf = KFold(n_splits=effective_splits, shuffle=True, random_state=random_state)

    rows: List[Dict[str, object]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(unique_ids), start=1):
        train_ids = {unique_ids[i] for i in train_idx}
        test_ids = {unique_ids[i] for i in test_idx}

        for subject_id in unique_ids:
            split = "test" if subject_id in test_ids else "train"
            rows.append({"id": subject_id, "fold": fold_idx, "split": split})

    manifest = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    return manifest.sort_values(by=["fold", "id"]).reset_index(drop=True)


def save_split_manifest(manifest: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)


def load_split_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Split manifest not found: {path}")

    manifest = pd.read_csv(path)
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in manifest.columns]
    if missing_cols:
        raise ValueError(f"Split manifest missing required columns: {missing_cols}")

    manifest = manifest[REQUIRED_COLUMNS].copy()
    manifest["id"] = manifest["id"].astype(str).str.strip()
    manifest["fold"] = pd.to_numeric(manifest["fold"], errors="raise").astype(int)
    manifest["split"] = manifest["split"].astype(str).str.strip().str.lower()

    invalid_split_values = sorted(set(manifest["split"]) - VALID_SPLITS)
    if invalid_split_values:
        raise ValueError(f"Split manifest contains invalid split values: {invalid_split_values}")

    return manifest


def validate_split_manifest(manifest: pd.DataFrame, dataset_subject_ids: Sequence[str]) -> None:
    data_ids: Set[str] = {str(subject_id).strip() for subject_id in dataset_subject_ids if str(subject_id).strip()}
    manifest_ids: Set[str] = set(manifest["id"].tolist())

    if not data_ids:
        raise ValueError("Dataset subject IDs are empty.")

    if data_ids != manifest_ids:
        only_in_data = sorted(data_ids - manifest_ids)
        only_in_manifest = sorted(manifest_ids - data_ids)
        raise ValueError(
            "Dataset IDs and manifest IDs do not match. "
            f"Only in dataset: {len(only_in_data)}; only in manifest: {len(only_in_manifest)}"
        )

    duplicates = manifest.duplicated(subset=["fold", "id"]).any()
    if duplicates:
        raise ValueError("Split manifest contains duplicate (fold, id) rows.")

    folds = sorted(manifest["fold"].unique().tolist())
    for fold in folds:
        fold_df = manifest.loc[manifest["fold"] == fold]
        if len(fold_df) != len(data_ids):
            raise ValueError(f"Fold {fold} has {len(fold_df)} rows, expected {len(data_ids)}.")

        split_values = set(fold_df["split"].tolist())
        if split_values != VALID_SPLITS:
            raise ValueError(f"Fold {fold} must contain both train and test assignments.")


def fold_counts(manifest: pd.DataFrame) -> pd.DataFrame:
    counts = (
        manifest.groupby(["fold", "split"], as_index=False)
        .size()
        .pivot(index="fold", columns="split", values="size")
        .fillna(0)
        .astype(int)
        .reset_index()
    )

    if "train" not in counts.columns:
        counts["train"] = 0
    if "test" not in counts.columns:
        counts["test"] = 0

    return counts[["fold", "train", "test"]].sort_values(by="fold").reset_index(drop=True)
