from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import mne
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor


if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EEG_DIR = DATA_DIR / "eeg" / "EEG_Preprocessed_BIDS_ID"

METADATA_CSV = (
    DATA_DIR
    / "phenotype"
    / "Behavioural_Data_MPILMBB_LEMON"
    / "META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv"
)

COGNITIVE_DIR = (
    DATA_DIR / "phenotype" / "Behavioural_Data_MPILMBB_LEMON" / "Cognitive_Test_Battery_LEMON"
)

TARGET_FILES = {
    "working_memory": COGNITIVE_DIR / "TAP_Working_Memory" / "TAP-Working Memory.csv",
    "attention": COGNITIVE_DIR / "TAP_Alertness" / "TAP-Alertness.csv",
    "executive_function": COGNITIVE_DIR / "TAP_Incompatibility" / "TAP-Incompatibility.csv",
    "intelligence": COGNITIVE_DIR / "WST" / "WST.csv",
}

TARGET_COLUMNS = ["working_memory", "attention", "executive_function", "intelligence"]

MODEL_PATH = PROJECT_ROOT / "models" / "rf_model.pkl"
FEATURES_CSV_PATH = PROJECT_ROOT / "processed" / "features.csv"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "processed" / "feature_columns.json"

RANDOM_STATE = 42
CV_SPLITS = 3

BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "low_alpha": (8.0, 10.0),
    "alpha": (8.0, 12.0),
    "high_alpha": (10.0, 12.0),
    "low_beta": (12.0, 20.0),
    "beta": (12.0, 30.0),
    "high_beta": (20.0, 30.0),
}


def log(message: str) -> None:
    print(f"[INFO] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def extract_subject_id_from_filename(file_path: Path) -> Optional[str]:
    match = re.search(r"(sub-\d+)_EC\.set$", file_path.name)
    if not match:
        return None
    return match.group(1)


def parse_age_range(value: object) -> float:
    if pd.isna(value):
        return float("nan")

    text = str(value).strip()
    range_match = re.match(r"^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)$", text)
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        return (low + high) / 2.0

    as_num = pd.to_numeric(text, errors="coerce")
    if pd.isna(as_num):
        return float("nan")
    return float(as_num)


def map_gender(value: object) -> float:
    if pd.isna(value):
        return float("nan")

    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return float("nan")

    if int(numeric) == 1:
        return 0.0
    if int(numeric) == 2:
        return 1.0
    return float("nan")


def sanitize_channel_name(channel_name: str) -> str:
    cleaned = channel_name.strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in cleaned).strip("_")


def load_metadata() -> pd.DataFrame:
    if not METADATA_CSV.exists():
        warn(f"Metadata CSV not found: {METADATA_CSV}")
        return pd.DataFrame()

    try:
        metadata = pd.read_csv(METADATA_CSV)
    except Exception as exc:
        warn(f"Failed to read metadata CSV: {exc}")
        return pd.DataFrame()

    required_cols = ["ID", "Age", "Gender_ 1=female_2=male"]
    missing = [c for c in required_cols if c not in metadata.columns]
    if missing:
        warn(f"Metadata missing required columns: {missing}")
        return pd.DataFrame()

    metadata = metadata[["ID", "Age", "Gender_ 1=female_2=male"]].copy()
    metadata["ID"] = metadata["ID"].astype(str).str.strip()
    metadata["age"] = metadata["Age"].map(parse_age_range)
    metadata["gender"] = metadata["Gender_ 1=female_2=male"].map(map_gender)

    metadata = metadata.dropna(subset=["ID", "age", "gender"]).copy()
    metadata = metadata[["ID", "age", "gender"]]
    metadata = metadata.drop_duplicates(subset=["ID"], keep="first")

    log(f"Loaded metadata rows: {len(metadata)}")
    return metadata


def select_first_numeric_target(df: pd.DataFrame, dataset_name: str) -> Optional[pd.Series]:
    for col in df.columns:
        if col == "ID":
            continue
        numeric_col = pd.to_numeric(df[col], errors="coerce")
        if numeric_col.notna().any():
            return numeric_col

    warn(f"No numeric target column found in {dataset_name}")
    return None


def load_target_file(file_path: Path, target_name: str) -> pd.DataFrame:
    if not file_path.exists():
        warn(f"Target file not found for {target_name}: {file_path}")
        return pd.DataFrame(columns=["ID", target_name])

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        warn(f"Failed to read target file {file_path}: {exc}")
        return pd.DataFrame(columns=["ID", target_name])

    if "ID" not in df.columns:
        warn(f"Target file missing ID column: {file_path}")
        return pd.DataFrame(columns=["ID", target_name])

    series = select_first_numeric_target(df, target_name)
    if series is None:
        return pd.DataFrame(columns=["ID", target_name])

    out = pd.DataFrame({"ID": df["ID"].astype(str).str.strip(), target_name: series})
    out = out.dropna(subset=["ID", target_name]).copy()
    out = out.drop_duplicates(subset=["ID"], keep="first")
    log(f"Loaded target '{target_name}' rows: {len(out)}")
    return out


def build_merged_targets_and_metadata() -> pd.DataFrame:
    metadata = load_metadata()
    if metadata.empty:
        return pd.DataFrame()

    merged = metadata.copy()
    for target_name, target_path in TARGET_FILES.items():
        target_df = load_target_file(target_path, target_name)
        if target_df.empty:
            warn(f"Target dataframe empty for {target_name}")
            return pd.DataFrame()
        merged = merged.merge(target_df, on="ID", how="inner")
        log(f"Rows after merging {target_name}: {len(merged)}")

    merged = merged.dropna(subset=["age", "gender", *TARGET_COLUMNS]).copy()
    merged = merged.drop_duplicates(subset=["ID"], keep="first")
    log(f"Final merged rows (metadata + all targets): {len(merged)}")
    return merged


def find_eeg_files() -> List[Path]:
    if not EEG_DIR.exists():
        warn(f"EEG directory not found: {EEG_DIR}")
        return []

    files = sorted(EEG_DIR.glob("*_EC.set"))
    log(f"Found *_EC.set EEG files: {len(files)}")
    return files


def load_and_preprocess_eeg(file_path: Path) -> Optional[mne.io.BaseRaw]:
    try:
        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose="ERROR")
    except Exception as exc:
        warn(f"Failed to load EEG file {file_path.name}: {exc}")
        return None

    try:
        raw.pick_types(eeg=True, meg=False, eog=False, ecg=False, stim=False, exclude="bads")
        if len(raw.ch_names) == 0:
            warn(f"No EEG channels after picking EEG: {file_path.name}")
            return None
        raw.set_eeg_reference(ref_channels="average", projection=False, verbose="ERROR")
    except Exception as exc:
        warn(f"Failed EEG preprocessing for {file_path.name}: {exc}")
        return None

    return raw


def compute_bandpower_features(raw: mne.io.BaseRaw) -> Dict[str, float]:
    eeg_data = raw.get_data(picks="eeg")
    if eeg_data.size == 0:
        return {}

    sfreq = float(raw.info["sfreq"])
    n_times = eeg_data.shape[1]
    if n_times < 8:
        return {}

    n_fft = min(2048, n_times)

    psd, freqs = mne.time_frequency.psd_array_welch(
        eeg_data,
        sfreq=sfreq,
        fmin=1.0,
        fmax=30.0,
        n_fft=n_fft,
        n_overlap=0,
        verbose="ERROR",
    )

    features: Dict[str, float] = {}
    for ch_idx, ch_name in enumerate(raw.ch_names):
        clean_ch = sanitize_channel_name(ch_name)
        for band_name, (fmin, fmax) in BANDS.items():
            mask = (freqs >= fmin) & (freqs <= fmax)
            if np.count_nonzero(mask) < 2:
                power = 0.0
            else:
                power = float(np.trapz(psd[ch_idx, mask], freqs[mask]))
            features[f"{clean_ch}_{band_name}"] = power

    return features


def build_dataset(merged_table: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eeg_files = find_eeg_files()
    if not eeg_files:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    merged_map = merged_table.set_index("ID")

    row_dicts: List[Dict[str, float]] = []
    row_targets: List[Dict[str, float]] = []
    participant_ids: List[str] = []

    for file_path in eeg_files:
        subject_id = extract_subject_id_from_filename(file_path)
        if subject_id is None:
            warn(f"Could not parse subject ID from filename: {file_path.name}")
            continue

        if subject_id not in merged_map.index:
            warn(f"Skipping {subject_id}: metadata/targets missing")
            continue

        raw = load_and_preprocess_eeg(file_path)
        if raw is None:
            warn(f"Skipping {subject_id}: EEG load/preprocessing failed")
            continue

        features = compute_bandpower_features(raw)
        if not features:
            warn(f"Skipping {subject_id}: feature extraction failed")
            continue

        subject_row = merged_map.loc[subject_id]
        features["age"] = float(subject_row["age"])
        features["gender"] = float(subject_row["gender"])

        targets = {
            "working_memory": float(subject_row["working_memory"]),
            "attention": float(subject_row["attention"]),
            "executive_function": float(subject_row["executive_function"]),
            "intelligence": float(subject_row["intelligence"]),
        }

        row_dicts.append(features)
        row_targets.append(targets)
        participant_ids.append(subject_id)
        log(f"Included subject: {subject_id}")

    if not row_dicts:
        warn("No valid subject samples created")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    X = pd.DataFrame(row_dicts)
    y = pd.DataFrame(row_targets, columns=TARGET_COLUMNS)

    band_cols = sorted([c for c in X.columns if c not in {"age", "gender"}])
    feature_cols = band_cols + ["age", "gender"]
    X = X.reindex(columns=feature_cols).fillna(0.0)

    combined = pd.concat(
        [
            pd.Series(participant_ids, name="ID"),
            X.reset_index(drop=True),
            y.reset_index(drop=True),
        ],
        axis=1,
    )

    log(f"Final dataset shapes: X={X.shape}, y={y.shape}")
    return X, y, combined


def build_model() -> MultiOutputRegressor:
    base = RandomForestRegressor(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features="sqrt",
    )
    return MultiOutputRegressor(base)


def evaluate_model(model: MultiOutputRegressor, X: pd.DataFrame, y: pd.DataFrame) -> None:
    n_samples = len(X)
    if n_samples < 2:
        warn("Not enough samples for evaluation")
        return

    if n_samples < CV_SPLITS:
        warn(f"Not enough samples for KFold n_splits={CV_SPLITS}; skipping CV")
        return

    kf = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        fold_model = build_model()
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_test)

        y_true_all.append(y_test.to_numpy())
        y_pred_all.append(np.asarray(y_pred))

        log(f"Completed fold {fold_idx}/{CV_SPLITS}")

    y_true = np.vstack(y_true_all)
    y_pred = np.vstack(y_pred_all)

    print("\nPer-target CV metrics:")
    for i, target in enumerate(TARGET_COLUMNS):
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        rmse = float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
        print(f"- {target}: R2={r2:.4f}, RMSE={rmse:.4f}")


def save_outputs(model: MultiOutputRegressor, X: pd.DataFrame, combined_df: pd.DataFrame) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_COLUMNS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    combined_df.to_csv(FEATURES_CSV_PATH, index=False)

    with FEATURE_COLUMNS_PATH.open("w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, indent=2)

    log(f"Saved model: {MODEL_PATH}")
    log(f"Saved features CSV: {FEATURES_CSV_PATH}")
    log(f"Saved feature schema: {FEATURE_COLUMNS_PATH}")


def main() -> None:
    log("Building merged metadata + targets")
    merged = build_merged_targets_and_metadata()
    if merged.empty:
        warn("Merged metadata/targets table is empty. Exiting.")
        return

    log("Building EEG feature dataset")
    X, y, combined_df = build_dataset(merged)
    if X.empty or y.empty:
        warn("Feature dataset is empty. Exiting.")
        return

    log("Training MultiOutput RandomForest model")
    model = build_model()
    model.fit(X, y)

    log("Running KFold evaluation")
    evaluate_model(model, X, y)

    log("Saving outputs")
    save_outputs(model, X, combined_df)


if __name__ == "__main__":
    main()
