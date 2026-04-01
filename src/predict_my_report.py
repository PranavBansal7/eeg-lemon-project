from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "rf_model.pkl"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "processed" / "feature_columns.json"

PERSON_AGE = 22.0
PERSON_GENDER = 1.0  # male

TARGET_NAMES = [
    "working_memory",
    "attention",
    "executive_function",
    "intelligence",
]

BANDS = [
    "delta",
    "theta",
    "low_alpha",
    "alpha",
    "high_alpha",
    "low_beta",
    "beta",
    "high_beta",
]

# ============================== REPORT_FEATURES ===============================
# Report-adapted EEG anchors for known channels.
# Structure:
# REPORT_FEATURES[channel][state][band]
REPORT_FEATURES = {
    "cz": {
        "eo": {
            "delta": 26.0,
            "theta": 18.0,
            "low_alpha": 14.0,
            "high_alpha": 12.0,
            "alpha": 13.0,
            "low_beta": 9.0,
            "beta": 10.0,
            "high_beta": 7.0,
        },
        "ec": {
            "delta": 19.0,
            "theta": 15.0,
            "low_alpha": 24.0,
            "high_alpha": 18.0,
            "alpha": 24.0,
            "low_beta": 11.0,
            "beta": 10.0,
            "high_beta": 7.0,
        },
    },
    "f3": {
        "eo": {
            "delta": 25.0,
            "theta": 20.0,
            "low_alpha": 14.0,
            "high_alpha": 14.0,
            "alpha": 15.0,
            "low_beta": 9.0,
            "beta": 9.0,
            "high_beta": 5.0,
        },
        "ec": {
            "delta": 18.0,
            "theta": 18.0,
            "low_alpha": 21.0,
            "high_alpha": 20.0,
            "alpha": 23.0,
            "low_beta": 10.0,
            "beta": 10.0,
            "high_beta": 5.0,
        },
    },
    "f4": {
        "eo": {
            "delta": 24.0,
            "theta": 19.0,
            "low_alpha": 15.0,
            "high_alpha": 14.0,
            "alpha": 15.0,
            "low_beta": 9.0,
            "beta": 10.0,
            "high_beta": 6.0,
        },
        "ec": {
            "delta": 18.0,
            "theta": 17.0,
            "low_alpha": 22.0,
            "high_alpha": 19.0,
            "alpha": 23.0,
            "low_beta": 10.0,
            "beta": 10.0,
            "high_beta": 6.0,
        },
    },
    "fz": {
        "eo": {
            "delta": 25.0,
            "theta": 19.0,
            "low_alpha": 14.0,
            "high_alpha": 13.0,
            "alpha": 14.0,
            "low_beta": 10.0,
            "beta": 10.0,
            "high_beta": 6.0,
        },
        "ec": {
            "delta": 19.0,
            "theta": 16.0,
            "low_alpha": 22.0,
            "high_alpha": 18.0,
            "alpha": 23.0,
            "low_beta": 10.0,
            "beta": 10.0,
            "high_beta": 6.0,
        },
    },
    "o1": {
        "eo": {
            "delta": 20.0,
            "theta": 16.0,
            "low_alpha": 17.0,
            "high_alpha": 19.0,
            "alpha": 21.0,
            "low_beta": 9.0,
            "beta": 10.0,
            "high_beta": 8.0,
        },
        "ec": {
            "delta": 14.0,
            "theta": 13.0,
            "low_alpha": 30.0,
            "high_alpha": 24.0,
            "alpha": 32.0,
            "low_beta": 7.0,
            "beta": 8.0,
            "high_beta": 6.0,
        },
    },
}
# ============================================================================


def print_exploratory_warning() -> None:
    print("\n" + "=" * 100)
    print("WARNING: EXPLORATORY PERSONAL PREDICTION ONLY")
    print("This output is NOT clinically valid, NOT diagnostic, and must NOT be used for treatment decisions.")
    print("It is a report-adapted model estimate for technical exploration only.")
    print("=" * 100 + "\n")


def sanitize_channel_name(channel_name: str) -> str:
    cleaned = channel_name.strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in cleaned).strip("_")


def averaging_dictionaries(dicts: Sequence[Dict[str, float]]) -> Dict[str, float]:
    sums = {band: 0.0 for band in BANDS}
    counts = {band: 0 for band in BANDS}

    for band_dict in dicts:
        for band in BANDS:
            value = pd.to_numeric(band_dict.get(band), errors="coerce")
            if pd.notna(value):
                sums[band] += float(value)
                counts[band] += 1

    out: Dict[str, float] = {}
    for band in BANDS:
        if counts[band] == 0:
            out[band] = 0.0
        else:
            out[band] = sums[band] / counts[band]
    return out


def select_region_template(
    channel: str,
    frontal_template: Dict[str, float],
    central_template: Dict[str, float],
    posterior_template: Dict[str, float],
    global_template: Dict[str, float],
) -> Dict[str, float]:
    ch = sanitize_channel_name(channel)

    if ch.startswith(("f", "fp", "af")):
        return dict(frontal_template)
    if ch.startswith(("c", "fc", "cp")):
        return dict(central_template)
    if ch.startswith(("p", "po", "o")):
        return dict(posterior_template)
    return dict(global_template)


def parse_feature_name(feature_name: str) -> Optional[Tuple[str, str, Optional[str]]]:
    if feature_name in {"age", "gender"}:
        return None

    state: Optional[str] = None
    base_name = feature_name

    if feature_name.endswith("_eo"):
        state = "eo"
        base_name = feature_name[:-3]
    elif feature_name.endswith("_ec"):
        state = "ec"
        base_name = feature_name[:-3]

    for band in sorted(BANDS, key=len, reverse=True):
        suffix = f"_{band}"
        if base_name.endswith(suffix):
            channel = sanitize_channel_name(base_name[: -len(suffix)])
            if channel:
                return channel, band, state

    return None


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run training first to create models/rf_model.pkl."
        )
    return joblib.load(model_path)


def load_feature_schema(schema_path: Path) -> List[str]:
    if not schema_path.exists():
        raise FileNotFoundError(
            f"Feature schema not found at {schema_path}. "
            "Run training first to create processed/feature_columns.json."
        )

    with schema_path.open("r", encoding="utf-8") as f:
        columns = json.load(f)

    if not isinstance(columns, list) or not columns:
        raise ValueError("Feature schema is empty or invalid.")

    return [str(col) for col in columns]


def prepare_state_maps(report_features: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    state_maps: Dict[str, Dict[str, Dict[str, float]]] = {"eo": {}, "ec": {}}

    for raw_channel, state_dict in report_features.items():
        channel = sanitize_channel_name(raw_channel)

        for state in ("eo", "ec"):
            band_values: Dict[str, float] = {}
            this_state = state_dict.get(state, {})
            other_state = state_dict.get("ec" if state == "eo" else "eo", {})

            for band in BANDS:
                raw_val = this_state.get(band, other_state.get(band, 0.0))
                numeric_val = pd.to_numeric(raw_val, errors="coerce")
                band_values[band] = float(numeric_val) if pd.notna(numeric_val) else 0.0

            state_maps[state][channel] = band_values

    return state_maps


def build_final_input_dataframe(
    feature_columns: Sequence[str],
    report_features: Dict[str, Dict[str, Dict[str, float]]],
    age: float,
    gender: float,
) -> pd.DataFrame:
    parsed_specs: Dict[str, Optional[Tuple[str, str, Optional[str]]]] = {}
    required_channels = set()

    for col in feature_columns:
        spec = parse_feature_name(col)
        parsed_specs[col] = spec
        if spec is not None:
            channel, _, _ = spec
            required_channels.add(channel)

    state_maps = prepare_state_maps(report_features)

    for state in ("eo", "ec"):
        known = state_maps[state]
        global_template = averaging_dictionaries(list(known.values()))

        frontal_sources = [known[ch] for ch in ("f3", "f4", "fz") if ch in known]
        frontal_template = averaging_dictionaries(frontal_sources) if frontal_sources else dict(global_template)

        central_template = dict(known.get("cz", global_template))
        posterior_template = dict(known.get("o1", global_template))

        for channel in required_channels:
            if channel not in known:
                known[channel] = select_region_template(
                    channel=channel,
                    frontal_template=frontal_template,
                    central_template=central_template,
                    posterior_template=posterior_template,
                    global_template=global_template,
                )
            else:
                for band in BANDS:
                    if band not in known[channel]:
                        known[channel][band] = global_template[band]

    row: Dict[str, float] = {}

    for col in feature_columns:
        if col == "age":
            row[col] = float(age)
            continue
        if col == "gender":
            row[col] = float(gender)
            continue

        spec = parsed_specs.get(col)
        if spec is None:
            row[col] = 0.0
            continue

        channel, band, state = spec
        if state in {"eo", "ec"}:
            value = state_maps[state][channel][band]
        else:
            value = (state_maps["eo"][channel][band] + state_maps["ec"][channel][band]) / 2.0

        row[col] = float(value)

    return pd.DataFrame([row], columns=list(feature_columns))


def print_predictions(predictions, target_names: List[str]) -> None:
    values = predictions[0] if hasattr(predictions, "shape") and len(predictions.shape) == 2 else predictions
    values = list(values)

    labels = target_names
    if len(values) != len(labels):
        labels = [f"target_{idx + 1}" for idx in range(len(values))]

    print("Predicted values:")
    for label, value in zip(labels, values):
        print(f"- {label}: {float(value):.6f}")


def main() -> None:
    print_exploratory_warning()

    model = load_model(MODEL_PATH)
    feature_columns = load_feature_schema(FEATURE_COLUMNS_PATH)

    input_df = build_final_input_dataframe(
        feature_columns=feature_columns,
        report_features=REPORT_FEATURES,
        age=PERSON_AGE,
        gender=PERSON_GENDER,
    )

    predictions = model.predict(input_df)
    print_predictions(predictions, TARGET_NAMES)


if __name__ == "__main__":
    main()
