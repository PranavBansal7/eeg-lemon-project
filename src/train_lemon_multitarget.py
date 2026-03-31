from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from mne.io import BaseRaw
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.multioutput import MultiOutputRegressor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Easy-to-edit metadata settings
PHENOTYPE_CSV_PATH = DATA_DIR / "phenotype.csv"
PARTICIPANT_ID_COLUMN = "participant_id"
AGE_COLUMN = "age"
GENDER_COLUMN = "gender"

# Easy-to-edit target definitions
TARGET_COLUMNS = [
    "attention",
    "fluid_intelligence",
    "working_memory",
    "executive_function",
]

# Output locations
MODEL_PATH = PROJECT_ROOT / "models" / "rf_model.pkl"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "processed" / "feature_columns.json"
FEATURES_CSV_PATH = PROJECT_ROOT / "processed" / "features.csv"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "processed" / "feature_importance.png"

RANDOM_STATE = 42
CV_SPLITS = 5

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


def sanitize_channel_name(channel_name: str) -> str:
    cleaned = channel_name.strip().lower()
    normalized = [char if char.isalnum() else "_" for char in cleaned]
    return "".join(normalized).strip("_")


def _to_numeric(value: object) -> float:
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        return float("nan")
    return float(numeric_value)


def _map_gender(value: object) -> float:
    if pd.isna(value):
        return float("nan")

    if isinstance(value, (int, float, np.integer, np.floating)):
        as_float = float(value)
        if as_float in (0.0, 1.0):
            return as_float
        if as_float == 2.0:
            return 0.0
        return float("nan")

    normalized = str(value).strip().lower()
    mapping = {
        "m": 1.0,
        "male": 1.0,
        "man": 1.0,
        "f": 0.0,
        "female": 0.0,
        "woman": 0.0,
        "0": 0.0,
        "1": 1.0,
        "2": 0.0,
    }
    return mapping.get(normalized, float("nan"))


def _build_model() -> MultiOutputRegressor:
    base_model = RandomForestRegressor(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features="sqrt",
    )
    return MultiOutputRegressor(base_model)


def _find_participant_file(participant_id: str, data_dir: Path = DATA_DIR) -> Optional[Path]:
    pid = str(participant_id).strip().lower()
    if not pid:
        return None

    fif_files = sorted(data_dir.rglob("*.fif")) + sorted(data_dir.rglob("*.fif.gz"))
    if not fif_files:
        return None

    for file_path in fif_files:
        file_name = file_path.name.lower()
        file_stem = file_path.stem.lower()
        if pid == file_stem or pid in file_name:
            return file_path

    return None


def load_participant_file(participant_id: str, data_dir: Path = DATA_DIR) -> Optional[BaseRaw]:
    file_path = _find_participant_file(participant_id=participant_id, data_dir=data_dir)
    if file_path is None:
        print(f"No .fif EEG file found for participant '{participant_id}'.")
        return None

    try:
        return mne.io.read_raw_eeglab(file_path, preload=True, verbose="ERROR")
    except Exception as exc:
        print(f"Failed to load EEG file for participant '{participant_id}' from {file_path}: {exc}")
        return None


def preprocess_raw(raw: BaseRaw) -> Optional[BaseRaw]:
    try:
        processed = raw.copy()
        processed.pick_types(eeg=True, meg=False, eog=False, ecg=False, stim=False, exclude="bads")
        if not processed.ch_names:
            print("No EEG channels available after EEG-only channel selection.")
            return None

        processed.set_eeg_reference(ref_channels="average", projection=False, verbose="ERROR")
        return processed
    except Exception as exc:
        print(f"Preprocessing failed: {exc}")
        return None


def compute_bandpower_features(raw: BaseRaw) -> Dict[str, float]:
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
    for channel_index, channel_name in enumerate(raw.ch_names):
        clean_channel = sanitize_channel_name(channel_name)
        for band_name, (fmin, fmax) in BANDS.items():
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            if np.count_nonzero(freq_mask) < 2:
                band_power = 0.0
            else:
                band_power = float(np.trapz(psd[channel_index, freq_mask], freqs[freq_mask]))
            feature_key = f"{clean_channel}_{band_name}"
            features[feature_key] = band_power

    return features


def load_metadata(phenotype_path: Path = PHENOTYPE_CSV_PATH) -> pd.DataFrame:
    if not phenotype_path.exists():
        print(f"Phenotype CSV not found at {phenotype_path}")
        print("Place phenotype.csv in data/ and ensure it includes participant_id, age, gender, and targets.")
        return pd.DataFrame()

    try:
        metadata = pd.read_csv(phenotype_path)
    except Exception as exc:
        print(f"Could not read phenotype CSV at {phenotype_path}: {exc}")
        return pd.DataFrame()

    required_columns = [PARTICIPANT_ID_COLUMN, AGE_COLUMN, GENDER_COLUMN, *TARGET_COLUMNS]
    missing_columns = [column for column in required_columns if column not in metadata.columns]
    if missing_columns:
        print("Phenotype CSV is missing required columns:")
        print(", ".join(missing_columns))
        print("Edit constants near the top of this file if your CSV uses different column names.")
        return pd.DataFrame()

    metadata = metadata.copy()
    metadata[PARTICIPANT_ID_COLUMN] = metadata[PARTICIPANT_ID_COLUMN].astype(str).str.strip()
    metadata = metadata[metadata[PARTICIPANT_ID_COLUMN] != ""]

    metadata[AGE_COLUMN] = metadata[AGE_COLUMN].map(_to_numeric)
    metadata[GENDER_COLUMN] = metadata[GENDER_COLUMN].map(_map_gender)
    for target in TARGET_COLUMNS:
        metadata[target] = metadata[target].map(_to_numeric)

    metadata = metadata.dropna(subset=[AGE_COLUMN, GENDER_COLUMN, *TARGET_COLUMNS])

    if metadata.empty:
        print("No valid metadata rows remain after numeric conversion and NaN filtering.")

    return metadata


def build_dataset(
    data_dir: Path = DATA_DIR,
    metadata: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    fif_files = sorted(data_dir.rglob("*.fif")) + sorted(data_dir.rglob("*.fif.gz"))
    if not fif_files:
        print(f"No .fif files found under {data_dir}")
        print("Add LEMON EEG .fif files into data/ (or subfolders) and rerun.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if metadata is None:
        metadata = load_metadata(PHENOTYPE_CSV_PATH)

    if metadata.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    feature_rows = []
    target_rows = []
    participant_ids = []

    for _, row in metadata.iterrows():
        participant_id = str(row[PARTICIPANT_ID_COLUMN]).strip()
        raw = load_participant_file(participant_id, data_dir=data_dir)
        if raw is None:
            continue

        processed = preprocess_raw(raw)
        if processed is None:
            continue

        feature_dict = compute_bandpower_features(processed)
        if not feature_dict:
            continue

        feature_dict["age"] = float(row[AGE_COLUMN])
        feature_dict["gender"] = float(row[GENDER_COLUMN])

        feature_rows.append(feature_dict)
        target_rows.append({target: float(row[target]) for target in TARGET_COLUMNS})
        participant_ids.append(participant_id)

    if not feature_rows:
        print("No valid participant samples were created. Check participant IDs and EEG file names.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    X = pd.DataFrame(feature_rows)
    y = pd.DataFrame(target_rows, columns=TARGET_COLUMNS)

    band_columns = sorted(column for column in X.columns if column not in {"age", "gender"})
    feature_columns = band_columns + ["age", "gender"]
    X = X.reindex(columns=feature_columns).fillna(0.0)

    extracted_features = pd.concat(
        [
            pd.Series(participant_ids, name=PARTICIPANT_ID_COLUMN),
            X.reset_index(drop=True),
            y.reset_index(drop=True),
        ],
        axis=1,
    )

    return X, y, extracted_features


def train_model(X: pd.DataFrame, y: pd.DataFrame) -> MultiOutputRegressor:
    model = _build_model()
    model.fit(X, y)
    return model


def evaluate_model(model: MultiOutputRegressor, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    if X.empty or y.empty:
        print("Dataset is empty. Evaluation skipped.")
        return pd.DataFrame(columns=["target", "r2", "rmse"])

    n_splits = min(CV_SPLITS, len(X))
    if n_splits < 2:
        print("Not enough samples for cross-validation. Need at least 2 samples.")
        return pd.DataFrame(columns=["target", "r2", "rmse"])

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    y_pred = cross_val_predict(clone(model), X, y, cv=cv, n_jobs=-1)

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    metrics_rows = []
    for target_index, target_name in enumerate(TARGET_COLUMNS):
        target_true = y.iloc[:, target_index].to_numpy()
        target_pred = y_pred[:, target_index]

        target_r2 = r2_score(target_true, target_pred)
        target_rmse = float(np.sqrt(mean_squared_error(target_true, target_pred)))

        metrics_rows.append(
            {
                "target": target_name,
                "r2": target_r2,
                "rmse": target_rmse,
            }
        )

    metrics_df = pd.DataFrame(metrics_rows, columns=["target", "r2", "rmse"])
    print("\nCross-validation metrics:")
    print(metrics_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return metrics_df


def plot_feature_importance(
    model: MultiOutputRegressor,
    feature_columns: Sequence[str],
    output_path: Path = FEATURE_IMPORTANCE_PATH,
    top_n: int = 30,
) -> None:
    if not hasattr(model, "estimators_") or not model.estimators_:
        print("Feature importance plot skipped because model is not fitted.")
        return

    importance_list = []
    for estimator in model.estimators_:
        if hasattr(estimator, "feature_importances_"):
            importance_list.append(estimator.feature_importances_)

    if not importance_list:
        print("Feature importance plot skipped because importances are unavailable.")
        return

    mean_importance = np.mean(np.vstack(importance_list), axis=0)
    importance_df = pd.DataFrame(
        {
            "feature": list(feature_columns),
            "importance": mean_importance,
        }
    ).sort_values("importance", ascending=False)

    top_n = min(top_n, len(importance_df))
    if top_n <= 0:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, max(6, int(top_n * 0.35))))
    sns.barplot(data=importance_df.head(top_n), x="importance", y="feature", hue="feature", palette="crest", dodge=False)
    plt.title("Top Feature Importances (Mean Across Targets)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_outputs(
    model: MultiOutputRegressor,
    feature_columns: Sequence[str],
    extracted_features: pd.DataFrame,
) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_COLUMNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    with FEATURE_COLUMNS_PATH.open("w", encoding="utf-8") as file_handle:
        json.dump(list(feature_columns), file_handle, indent=2)

    extracted_features.to_csv(FEATURES_CSV_PATH, index=False)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved feature columns to {FEATURE_COLUMNS_PATH}")
    print(f"Saved extracted features to {FEATURES_CSV_PATH}")


def main() -> None:
    if not DATA_DIR.exists():
        print(f"Missing data directory: {DATA_DIR}")
        print("Create data/ in the project root and add LEMON .fif EEG files.")
        return

    fif_files = sorted(DATA_DIR.rglob("*.fif")) + sorted(DATA_DIR.rglob("*.fif.gz"))
    if not fif_files:
        print(f"No .fif EEG files found in {DATA_DIR}")
        print("Add LEMON .fif files under data/ and rerun this script.")
        return

    metadata = load_metadata(PHENOTYPE_CSV_PATH)
    if metadata.empty:
        print("Metadata could not be loaded. Please fix phenotype CSV and retry.")
        return

    X, y, extracted_features = build_dataset(data_dir=DATA_DIR, metadata=metadata)
    if X.empty or y.empty:
        print("Dataset assembly failed. Check participant IDs and EEG file naming.")
        return

    model = train_model(X, y)
    evaluate_model(model, X, y)

    feature_columns = list(X.columns)
    plot_feature_importance(model, feature_columns, output_path=FEATURE_IMPORTANCE_PATH)
    save_outputs(model, feature_columns, extracted_features)


if __name__ == "__main__":
    main()