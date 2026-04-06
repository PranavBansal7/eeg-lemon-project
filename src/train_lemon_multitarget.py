"""Build the base EEG dataset and save the main training artifacts.

This script has four main stages:

1. load metadata and target tables
2. load and lightly preprocess EO/EC EEG recordings
3. compute Welch PSD bandpower features for each subject
4. train, evaluate, and save the baseline model artifacts used elsewhere
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


def resolve_project_root() -> Path:
    # Prefer the repo containing `data/`, but keep a small fallback for common
    # workspace layouts so the script is easier to run in different environments.
    default_root = Path(__file__).resolve().parent.parent
    if (default_root / "data").exists():
        return default_root

    alt_root = Path("/workspaces") / default_root.name
    if (alt_root / "data").exists():
        return alt_root

    return default_root


PROJECT_ROOT = resolve_project_root()
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

WORKING_MEMORY_CSV = COGNITIVE_DIR / "TAP_Working_Memory" / "TAP-Working Memory.csv"
ATTENTION_CSV = COGNITIVE_DIR / "TAP_Alertness" / "TAP-Alertness.csv"
EXECUTIVE_FUNCTION_CSV = COGNITIVE_DIR / "TAP_Incompatibility" / "TAP-Incompatibility.csv"
INTELLIGENCE_CSV = COGNITIVE_DIR / "WST" / "WST.csv"

# Easy-to-edit target column choices.
# If a chosen column is missing/non-numeric, the script falls back to the first numeric column and logs a warning.
WORKING_MEMORY_TARGET_COLUMN: Optional[str] = "TAP_WM_1"
ATTENTION_TARGET_COLUMN: Optional[str] = "TAP_A_1"
EXECUTIVE_FUNCTION_TARGET_COLUMN: Optional[str] = "TAP_I_1"
INTELLIGENCE_TARGET_COLUMN: Optional[str] = "WST_1"

STANDARDIZE_TARGETS = True
RANDOM_STATE = 42
CV_SPLITS = 5

TARGET_ALIASES = ["working_memory", "attention", "executive_function", "intelligence"]

MODEL_PATH = PROJECT_ROOT / "models" / "rf_model.pkl"
FEATURES_CSV_PATH = PROJECT_ROOT / "processed" / "features.csv"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "processed" / "feature_columns.json"
TARGET_SCALER_PATH = PROJECT_ROOT / "processed" / "target_scaler.json"

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


def sanitize_channel_name(channel_name: str) -> str:
    cleaned = channel_name.strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in cleaned).strip("_")


def extract_subject_id_from_filename(file_path: Path) -> Optional[str]:
    match = re.search(r"(sub-\d+)_E[OC]\.set$", file_path.name)
    if not match:
        return None
    return match.group(1)


def parse_age_value(value: object) -> float:
    if pd.isna(value):
        return float("nan")

    text = str(value).strip()
    range_match = re.match(r"^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)$", text)
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        return (low + high) / 2.0

    numeric = pd.to_numeric(text, errors="coerce")
    if pd.isna(numeric):
        return float("nan")
    return float(numeric)


def map_gender_value(value: object) -> float:
    if pd.isna(value):
        return float("nan")

    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return float("nan")

    numeric_int = int(numeric)
    if numeric_int == 1:
        return 0.0
    if numeric_int == 2:
        return 1.0
    return float("nan")


def load_metadata() -> pd.DataFrame:
    if not METADATA_CSV.exists():
        warn(f"Metadata CSV not found: {METADATA_CSV}")
        return pd.DataFrame()

    try:
        metadata = pd.read_csv(METADATA_CSV)
    except Exception as exc:
        warn(f"Failed to read metadata CSV: {exc}")
        return pd.DataFrame()

    required_columns = ["ID", "Age", "Gender_ 1=female_2=male"]
    missing = [col for col in required_columns if col not in metadata.columns]
    if missing:
        warn(f"Metadata CSV missing required columns: {missing}")
        return pd.DataFrame()

    metadata = metadata[["ID", "Age", "Gender_ 1=female_2=male"]].copy()
    metadata["ID"] = metadata["ID"].astype(str).str.strip()
    metadata["age"] = metadata["Age"].map(parse_age_value)
    metadata["gender"] = metadata["Gender_ 1=female_2=male"].map(map_gender_value)

    metadata = metadata.dropna(subset=["ID", "age", "gender"]).copy()
    metadata = metadata[["ID", "age", "gender"]].drop_duplicates(subset=["ID"], keep="first")

    log(f"Loaded metadata rows: {len(metadata)}")
    return metadata


def find_id_column(df: pd.DataFrame) -> Optional[str]:
    if "ID" in df.columns:
        return "ID"

    for col in df.columns:
        if str(col).strip().lower() == "id":
            return col

    return None


def find_first_numeric_column(df: pd.DataFrame, exclude: Optional[Sequence[str]] = None) -> Optional[str]:
    exclude_set = {item.lower() for item in (exclude or [])}

    for col in df.columns:
        if str(col).strip().lower() in exclude_set:
            continue

        numeric_col = pd.to_numeric(df[col], errors="coerce")
        if numeric_col.notna().any():
            return col

    return None


def choose_target_column(
    df: pd.DataFrame,
    dataset_alias: str,
    csv_path: Path,
    preferred_column: Optional[str],
    id_column: str,
) -> Optional[str]:
    if preferred_column is not None:
        if preferred_column not in df.columns:
            warn(
                f"Preferred target column '{preferred_column}' not found for {dataset_alias} in {csv_path.name}. "
                "Falling back to first numeric column."
            )
        else:
            numeric_preferred = pd.to_numeric(df[preferred_column], errors="coerce")
            if numeric_preferred.notna().any():
                log(f"Using configured target column for {dataset_alias}: {preferred_column}")
                return preferred_column

            warn(
                f"Configured target column '{preferred_column}' for {dataset_alias} is non-numeric/empty in "
                f"{csv_path.name}. Falling back to first numeric column."
            )

    fallback = find_first_numeric_column(df, exclude=[id_column])
    if fallback is None:
        warn(f"No numeric target column found for {dataset_alias} in {csv_path.name}")
        return None

    warn(
        f"No exact target column configured/usable for {dataset_alias}. "
        f"Selected first numeric column: '{fallback}' from {csv_path.name}"
    )
    return fallback


def load_target_table(
    csv_path: Path,
    dataset_alias: str,
    preferred_column: Optional[str],
) -> Tuple[pd.DataFrame, Optional[str]]:
    if not csv_path.exists():
        warn(f"Target CSV not found for {dataset_alias}: {csv_path}")
        return pd.DataFrame(columns=["ID", dataset_alias]), None

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        warn(f"Failed reading {csv_path}: {exc}")
        return pd.DataFrame(columns=["ID", dataset_alias]), None

    id_column = find_id_column(df)
    if id_column is None:
        warn(f"No ID column found in target file: {csv_path}")
        return pd.DataFrame(columns=["ID", dataset_alias]), None

    selected_column = choose_target_column(
        df=df,
        dataset_alias=dataset_alias,
        csv_path=csv_path,
        preferred_column=preferred_column,
        id_column=id_column,
    )
    if selected_column is None:
        return pd.DataFrame(columns=["ID", dataset_alias]), None

    values = pd.to_numeric(df[selected_column], errors="coerce")
    out = pd.DataFrame(
        {
            "ID": df[id_column].astype(str).str.strip(),
            dataset_alias: values,
        }
    )
    out = out.dropna(subset=["ID", dataset_alias]).drop_duplicates(subset=["ID"], keep="first")

    log(f"Loaded target '{dataset_alias}' rows: {len(out)} (source column: {selected_column})")
    return out, selected_column


def load_metadata_and_targets() -> Tuple[pd.DataFrame, Dict[str, str]]:
    # Keep metadata and target loading explicit so it is easy to explain which
    # tables feed the final regression dataset.
    metadata = load_metadata()
    if metadata.empty:
        return pd.DataFrame(), {}

    target_specs = [
        ("working_memory", WORKING_MEMORY_CSV, WORKING_MEMORY_TARGET_COLUMN),
        ("attention", ATTENTION_CSV, ATTENTION_TARGET_COLUMN),
        ("executive_function", EXECUTIVE_FUNCTION_CSV, EXECUTIVE_FUNCTION_TARGET_COLUMN),
        ("intelligence", INTELLIGENCE_CSV, INTELLIGENCE_TARGET_COLUMN),
    ]

    merged = metadata.copy()
    selected_target_columns: Dict[str, str] = {}

    for alias, csv_path, preferred_col in target_specs:
        # Merge one target at a time so we can log exactly where rows drop.
        target_df, selected_col = load_target_table(csv_path, alias, preferred_col)
        if target_df.empty or selected_col is None:
            warn(f"Target table for {alias} is empty. Stopping dataset assembly.")
            return pd.DataFrame(), {}

        selected_target_columns[alias] = selected_col
        merged = merged.merge(target_df, on="ID", how="inner")
        log(f"Rows after merging target '{alias}': {len(merged)}")

    merged = merged.dropna(subset=["ID", "age", "gender", *TARGET_ALIASES])
    merged = merged.drop_duplicates(subset=["ID"], keep="first")

    log("Selected target columns:")
    for alias in TARGET_ALIASES:
        log(f"  {alias} -> {selected_target_columns.get(alias, 'unknown')}")

    log(f"Final merged metadata+targets rows: {len(merged)}")
    return merged, selected_target_columns


def find_subject_file_pairs() -> Dict[str, Dict[str, Path]]:
    if not EEG_DIR.exists():
        warn(f"EEG directory not found: {EEG_DIR}")
        return {}

    eo_files: Dict[str, Path] = {}
    ec_files: Dict[str, Path] = {}

    for path in sorted(EEG_DIR.glob("*_EO.set")):
        subject_id = extract_subject_id_from_filename(path)
        if subject_id is not None:
            eo_files[subject_id] = path

    for path in sorted(EEG_DIR.glob("*_EC.set")):
        subject_id = extract_subject_id_from_filename(path)
        if subject_id is not None:
            ec_files[subject_id] = path

    # The benchmark only makes sense when we have both resting-state conditions
    # for the same subject, because EO/EC contrast is part of the feature story.
    shared_subjects = sorted(set(eo_files.keys()) & set(ec_files.keys()))
    only_eo = sorted(set(eo_files.keys()) - set(ec_files.keys()))
    only_ec = sorted(set(ec_files.keys()) - set(eo_files.keys()))

    log(f"Found EO files: {len(eo_files)}")
    log(f"Found EC files: {len(ec_files)}")
    log(f"Subjects with both EO and EC: {len(shared_subjects)}")

    if only_eo:
        warn(f"Subjects with EO but missing EC: {len(only_eo)}")
    if only_ec:
        warn(f"Subjects with EC but missing EO: {len(only_ec)}")

    return {subject_id: {"eo": eo_files[subject_id], "ec": ec_files[subject_id]} for subject_id in shared_subjects}


def load_and_preprocess_eeg(file_path: Path) -> Optional[mne.io.BaseRaw]:
    try:
        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose="ERROR")
    except Exception as exc:
        warn(f"Failed to load EEG file {file_path.name}: {exc}")
        return None

    try:
        processed = raw.copy()
        # Keep preprocessing intentionally light and reproducible: EEG channels
        # only, use marked bad channels if available, then apply average reference.
        processed.pick(picks="eeg", exclude="bads")
        if len(processed.ch_names) == 0:
            warn(f"No EEG channels available after picking EEG: {file_path.name}")
            return None

        processed.set_eeg_reference(ref_channels="average", projection=False, verbose="ERROR")
        return processed
    except Exception as exc:
        warn(f"Preprocessing failed for {file_path.name}: {exc}")
        return None


def compute_bandpower_features(raw: mne.io.BaseRaw, state_suffix: str) -> Dict[str, float]:
    eeg_data = raw.get_data(picks="eeg")
    if eeg_data.size == 0:
        return {}

    sfreq = float(raw.info["sfreq"])
    n_times = eeg_data.shape[1]
    if n_times < 8:
        return {}

    n_fft = min(2048, n_times)

    # Welch PSD is an FFT-based spectral estimation method that averages
    # spectra across windows to make the frequency estimate more stable.
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
                bandpower = 0.0
            else:
                # Bandpower is created by integrating PSD values inside a named
                # frequency range such as alpha or beta.
                bandpower = float(np.trapz(psd[ch_idx, mask], freqs[mask]))

            feature_name = f"{clean_ch}_{band_name}_{state_suffix}"
            features[feature_name] = bandpower

    return features


def build_dataset(
    metadata_targets: pd.DataFrame,
    subject_file_pairs: Dict[str, Dict[str, Path]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if metadata_targets.empty:
        warn("Metadata/target table is empty.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if not subject_file_pairs:
        warn("No subjects with both EO and EC files were found.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    metadata_by_id = metadata_targets.set_index("ID")

    feature_rows: List[Dict[str, float]] = []
    target_rows: List[Dict[str, float]] = []
    participant_ids: List[str] = []

    for subject_id, pair in subject_file_pairs.items():
        if subject_id not in metadata_by_id.index:
            warn(f"Skipping {subject_id}: missing metadata/target row")
            continue

        eo_path = pair["eo"]
        ec_path = pair["ec"]

        raw_eo = load_and_preprocess_eeg(eo_path)
        if raw_eo is None:
            warn(f"Skipping {subject_id}: EO load/preprocess failed")
            continue

        raw_ec = load_and_preprocess_eeg(ec_path)
        if raw_ec is None:
            warn(f"Skipping {subject_id}: EC load/preprocess failed")
            continue

        eo_features = compute_bandpower_features(raw_eo, state_suffix="eo")
        ec_features = compute_bandpower_features(raw_ec, state_suffix="ec")

        if not eo_features or not ec_features:
            warn(f"Skipping {subject_id}: empty EO/EC features")
            continue

        row = metadata_by_id.loc[subject_id]

        # Each subject becomes one tabular ML row: EO features, EC features,
        # simple metadata, and the four target values.
        combined_features: Dict[str, float] = {}
        combined_features.update(eo_features)
        combined_features.update(ec_features)
        combined_features["age"] = float(row["age"])
        combined_features["gender"] = float(row["gender"])

        target_row = {alias: float(row[alias]) for alias in TARGET_ALIASES}

        feature_rows.append(combined_features)
        target_rows.append(target_row)
        participant_ids.append(subject_id)

        log(f"Included subject: {subject_id}")

    if not feature_rows:
        warn("No valid training rows were created.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    X = pd.DataFrame(feature_rows)
    y_raw = pd.DataFrame(target_rows, columns=TARGET_ALIASES)

    eeg_feature_columns = sorted([col for col in X.columns if col not in {"age", "gender"}])
    ordered_columns = eeg_feature_columns + ["age", "gender"]
    X = X.reindex(columns=ordered_columns).fillna(0.0)

    features_df = pd.concat(
        [
            pd.Series(participant_ids, name="ID"),
            X.reset_index(drop=True),
            y_raw.reset_index(drop=True),
        ],
        axis=1,
    )

    log(f"Final dataset shapes -> X: {X.shape}, y: {y_raw.shape}")
    return X, y_raw, features_df


def standardize_targets(y_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    # Save the mean/std per target so training can happen in a normalized space
    # while later reporting can still return to raw target units.
    means = y_raw.mean(axis=0)
    stds = y_raw.std(axis=0, ddof=0)

    safe_stds = stds.copy()
    for target in y_raw.columns:
        std_value = float(safe_stds[target])
        if not np.isfinite(std_value) or std_value == 0.0:
            warn(f"Target '{target}' has zero/non-finite std. Using std=1.0 for scaling.")
            safe_stds[target] = 1.0

    y_z = (y_raw - means) / safe_stds

    means_dict = {str(col): float(means[col]) for col in y_raw.columns}
    stds_dict = {str(col): float(safe_stds[col]) for col in y_raw.columns}
    return y_z, means_dict, stds_dict


def build_model() -> MultiOutputRegressor:
    base = RandomForestRegressor(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features="sqrt",
    )
    return MultiOutputRegressor(base)


def evaluate_model(
    X: pd.DataFrame,
    y_eval: pd.DataFrame,
    selected_target_columns: Dict[str, str],
    standardized: bool,
) -> None:
    if X.empty or y_eval.empty:
        warn("Evaluation skipped: empty dataset.")
        return

    n_samples = len(X)
    n_splits = min(CV_SPLITS, n_samples)

    if n_splits < 2:
        warn("Evaluation skipped: need at least 2 samples for KFold.")
        return

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    y_true_parts: List[np.ndarray] = []
    y_pred_parts: List[np.ndarray] = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_eval.iloc[train_idx], y_eval.iloc[test_idx]

        model = build_model()
        model.fit(X_train, y_train)
        y_pred = np.asarray(model.predict(X_test))

        y_true_parts.append(y_test.to_numpy())
        y_pred_parts.append(y_pred)

        log(f"Completed CV fold {fold_idx}/{n_splits}")

    y_true = np.vstack(y_true_parts)
    y_pred = np.vstack(y_pred_parts)

    print("\nTarget mapping:")
    for alias in TARGET_ALIASES:
        source_col = selected_target_columns.get(alias, "unknown")
        print(f"- {alias}: source column '{source_col}'")

    space_label = "z-score" if standardized else "raw"
    print(f"\nCross-validation metrics ({space_label} target space):")

    for idx, alias in enumerate(TARGET_ALIASES):
        source_col = selected_target_columns.get(alias, "unknown")
        r2_value = r2_score(y_true[:, idx], y_pred[:, idx])
        rmse_value = float(np.sqrt(mean_squared_error(y_true[:, idx], y_pred[:, idx])))
        print(
            f"- {alias} [{source_col}] -> R2={r2_value:.4f}, RMSE={rmse_value:.4f}"
        )


def save_outputs(
    model: MultiOutputRegressor,
    X: pd.DataFrame,
    features_df: pd.DataFrame,
    scaler_payload: Dict[str, object],
) -> None:
    # Save the full set of artifacts needed by the optional demo/inference path:
    # trained model, feature table, feature schema, and target-scaling metadata.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_COLUMNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TARGET_SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    features_df.to_csv(FEATURES_CSV_PATH, index=False)

    with FEATURE_COLUMNS_PATH.open("w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, indent=2)

    with TARGET_SCALER_PATH.open("w", encoding="utf-8") as f:
        json.dump(scaler_payload, f, indent=2)

    log(f"Saved model: {MODEL_PATH}")
    log(f"Saved features CSV: {FEATURES_CSV_PATH}")
    log(f"Saved feature schema: {FEATURE_COLUMNS_PATH}")
    log(f"Saved target scaler: {TARGET_SCALER_PATH}")


def main() -> None:
    log("Loading metadata and target tables")
    metadata_targets, selected_target_columns = load_metadata_and_targets()
    if metadata_targets.empty:
        warn("Metadata/target merge failed. Exiting.")
        return

    log("Finding subjects with both EO and EC files")
    subject_file_pairs = find_subject_file_pairs()
    if not subject_file_pairs:
        warn("No EO+EC subject pairs available. Exiting.")
        return

    log("Building EO+EC feature dataset")
    X, y_raw, features_df = build_dataset(metadata_targets, subject_file_pairs)
    if X.empty or y_raw.empty:
        warn("Dataset assembly failed. Exiting.")
        return

    if STANDARDIZE_TARGETS:
        log("Standardizing targets to z-scores")
        # Standardizing targets can make the multi-target training problem more
        # numerically stable while still letting us save the raw-space statistics.
        y_train, target_means, target_stds = standardize_targets(y_raw)

        for alias in TARGET_ALIASES:
            features_df[f"{alias}_z"] = y_train[alias].to_numpy()
    else:
        log("Using raw targets (no standardization)")
        y_train = y_raw.copy()
        target_means = {alias: float(y_raw[alias].mean()) for alias in TARGET_ALIASES}
        std_series = y_raw.std(axis=0, ddof=0)
        target_stds = {
            alias: float(std_series[alias]) if np.isfinite(std_series[alias]) and float(std_series[alias]) != 0.0 else 1.0
            for alias in TARGET_ALIASES
        }

    log("Running cross-validation")
    evaluate_model(
        X=X,
        y_eval=y_train,
        selected_target_columns=selected_target_columns,
        standardized=STANDARDIZE_TARGETS,
    )

    log("Training final model on all available data")
    model = build_model()
    model.fit(X, y_train)

    scaler_payload: Dict[str, object] = {
        "standardized": bool(STANDARDIZE_TARGETS),
        "target_order": list(TARGET_ALIASES),
        "source_columns": {alias: selected_target_columns.get(alias, "") for alias in TARGET_ALIASES},
        "means": {alias: float(target_means[alias]) for alias in TARGET_ALIASES},
        "stds": {alias: float(target_stds[alias]) for alias in TARGET_ALIASES},
    }

    log("Saving outputs")
    save_outputs(
        model=model,
        X=X,
        features_df=features_df,
        scaler_payload=scaler_payload,
    )


if __name__ == "__main__":
    main()
