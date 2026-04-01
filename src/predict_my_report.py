from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "rf_model.pkl"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "processed" / "feature_columns.json"

RELATIVE_AMPLITUDES_CSV = PROJECT_ROOT / "assessment_2026-03-05_relative_amplitudes_estimated.csv"
RATIO_METRICS_CSV = PROJECT_ROOT / "assessment_2026-03-05_ratio_metrics.csv"
TOPOMAP_ZSCORES_CSV = PROJECT_ROOT / "assessment_2026-03-05_topomap_zscores_estimated.csv"

REPORT_FALLBACK_DIRS = [
    PROJECT_ROOT / "reports" / "data",
    PROJECT_ROOT / "reports",
    PROJECT_ROOT / "my_data",
    PROJECT_ROOT / "data",
]

DEFAULT_AGE = 22.0
DEFAULT_GENDER = 1.0

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

ANCHOR_CHANNELS = ["cz", "f3", "f4", "fz", "o1"]

RATIO_ADJUSTMENT_STRENGTH = 0.12
TOPOMAP_MODIFIER_STRENGTH = 0.035
EPS = 1e-8


def log(message: str) -> None:
    print(f"[INFO] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def print_exploratory_warning() -> None:
    print("\n" + "=" * 95)
    print("WARNING: EXPLORATORY REPORT-ADAPTED PREDICTION ONLY")
    print("This prediction is NOT clinically validated, NOT diagnostic, and NOT suitable for treatment decisions.")
    print("It is a model-driven approximation from estimated report features and should be treated as experimental.")
    print("=" * 95 + "\n")


def normalize_text(value: object) -> str:
    text = str(value).strip().lower()
    text = text.replace("%", "pct")
    text = text.replace("/", "_")
    text = text.replace("-", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {col: normalize_text(col) for col in df.columns}
    return df.rename(columns=renamed)


def resolve_csv_path(primary_path: Path, fallback_dirs: Sequence[Path]) -> Path:
    if primary_path.exists():
        return primary_path

    for directory in fallback_dirs:
        candidate = directory / primary_path.name
        if candidate.exists():
            return candidate

    searched = [str(primary_path)] + [str(directory / primary_path.name) for directory in fallback_dirs]
    raise FileNotFoundError(f"Could not find CSV file. Searched: {searched}")


def load_csv_with_fallback(primary_path: Path, fallback_dirs: Sequence[Path]) -> pd.DataFrame:
    csv_path = resolve_csv_path(primary_path, fallback_dirs)
    log(f"Loading CSV: {csv_path}")
    return pd.read_csv(csv_path)


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at: {model_path}. Run training first to create models/rf_model.pkl."
        )
    return joblib.load(model_path)


def load_feature_schema(schema_path: Path) -> List[str]:
    if not schema_path.exists():
        raise FileNotFoundError(
            f"Feature schema not found at: {schema_path}. "
            "Run training first to create processed/feature_columns.json."
        )

    with schema_path.open("r", encoding="utf-8") as handle:
        columns = json.load(handle)

    if not isinstance(columns, list) or not columns:
        raise ValueError("Feature schema is empty or invalid.")

    return [str(column) for column in columns]


def normalize_state_label(value: object) -> str:
    state = normalize_text(value)
    if state in {"eo", "eyes_open", "open"}:
        return "eo"
    if state in {"ec", "eyes_closed", "closed"}:
        return "ec"
    if "eyes_open" in state:
        return "eo"
    if "eyes_closed" in state:
        return "ec"
    if "count" in state or "task" in state:
        return "task"
    return state


def infer_state_from_comparison(value: object) -> Optional[str]:
    state = normalize_text(value)
    if "eyes_closed" in state or state.endswith("closed"):
        return "ec"
    if "eyes_open" in state or state.endswith("open"):
        return "eo"
    if "count" in state or "task" in state:
        return "task"
    return None


def canonical_band_name(raw_band: object) -> Optional[str]:
    key = normalize_text(raw_band)
    mapping = {
        "delta": "delta",
        "theta": "theta",
        "slowwaves": "delta",
        "slow_waves": "delta",
        "lowalpha": "low_alpha",
        "low_alpha": "low_alpha",
        "alpha": "alpha",
        "highalpha": "high_alpha",
        "high_alpha": "high_alpha",
        "lowbeta": "low_beta",
        "low_beta": "low_beta",
        "beta": "beta",
        "highbeta": "high_beta",
        "high_beta": "high_beta",
    }
    return mapping.get(key)


def get_first_existing_column(df: pd.DataFrame, options: Sequence[str]) -> Optional[str]:
    for option in options:
        if option in df.columns:
            return option
    return None


def _init_state_dict() -> Dict[str, Dict[str, Dict[str, float]]]:
    return {"eo": {}, "ec": {}}


def build_eo_ec_anchor_dicts(relative_df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, float]]]:
    df = normalize_columns(relative_df)

    channel_col = get_first_existing_column(df, ["electrode", "channel", "site_or_channel", "site"])
    state_col = get_first_existing_column(df, ["state", "condition", "eyes_state"])

    if channel_col is None or state_col is None:
        raise ValueError("Relative amplitudes CSV must include electrode/channel and state columns.")

    band_column_candidates = {
        "delta": ["delta_pct", "delta", "delta_percent"],
        "theta": ["theta_pct", "theta", "theta_percent"],
        "low_alpha": ["low_alpha_pct", "low_alpha", "lowalpha_pct", "lowalpha"],
        "alpha": ["alpha_pct", "alpha", "alpha_percent"],
        "high_alpha": ["high_alpha_pct", "high_alpha", "highalpha_pct", "highalpha"],
        "low_beta": ["low_beta_pct", "low_beta", "lowbeta_pct", "lowbeta"],
        "beta": ["beta_pct", "beta", "beta_percent"],
        "high_beta": ["high_beta_pct", "high_beta", "highbeta_pct", "highbeta"],
    }

    resolved_band_cols: Dict[str, str] = {}
    for band, candidates in band_column_candidates.items():
        col = get_first_existing_column(df, candidates)
        if col is not None:
            resolved_band_cols[band] = col

    if len(resolved_band_cols) < 6:
        raise ValueError("Relative amplitudes CSV missing too many band columns.")

    anchors: Dict[str, Dict[str, Dict[str, float]]] = _init_state_dict()

    for _, row in df.iterrows():
        channel = normalize_text(row[channel_col])
        state = normalize_state_label(row[state_col])
        if channel == "" or state not in {"eo", "ec", "task"}:
            continue

        if state == "task":
            continue

        if channel not in anchors[state]:
            anchors[state][channel] = {}

        for band, col in resolved_band_cols.items():
            value = pd.to_numeric(row[col], errors="coerce")
            if pd.isna(value):
                continue
            anchors[state][channel][band] = float(value)

    for state in ("eo", "ec"):
        for channel in list(anchors[state].keys()):
            for band in BANDS:
                anchors[state][channel].setdefault(band, np.nan)

    log(
        "Built anchor dictionaries from relative amplitudes: "
        f"EO channels={len(anchors['eo'])}, EC channels={len(anchors['ec'])}"
    )
    return anchors


def parse_site_channels(site_or_channel: object) -> List[str]:
    site = normalize_text(site_or_channel)
    if not site:
        return []

    if "_" in site:
        parts = [part for part in site.split("_") if part]
        if parts:
            return parts

    return [site]


def apply_ratio_pair_adjustment(
    anchors: Dict[str, Dict[str, Dict[str, float]]],
    state: str,
    channel: str,
    numerator_band: str,
    denominator_band: str,
    desired_ratio: float,
    strength: float,
) -> None:
    if state not in anchors or channel not in anchors[state]:
        return

    if not np.isfinite(desired_ratio) or desired_ratio <= 0:
        return

    channel_dict = anchors[state][channel]
    numerator = float(channel_dict.get(numerator_band, np.nan))
    denominator = float(channel_dict.get(denominator_band, np.nan))

    if not np.isfinite(numerator) or numerator <= 0:
        numerator = 1.0
    if not np.isfinite(denominator) or denominator <= 0:
        denominator = 1.0

    current_ratio = numerator / max(denominator, EPS)
    if not np.isfinite(current_ratio) or current_ratio <= 0:
        return

    delta = np.log(desired_ratio / current_ratio)
    scale = float(np.exp(np.clip(strength * delta, -0.10, 0.10)))

    channel_dict[numerator_band] = max(numerator * scale, EPS)
    channel_dict[denominator_band] = max(denominator / scale, EPS)


def apply_symmetry_adjustment(
    anchors: Dict[str, Dict[str, Dict[str, float]]],
    state: str,
    left_channel: str,
    right_channel: str,
    band: str,
    desired_ratio: float,
    strength: float,
) -> None:
    if state not in anchors:
        return
    if left_channel not in anchors[state] or right_channel not in anchors[state]:
        return
    if not np.isfinite(desired_ratio) or desired_ratio <= 0:
        return

    left_val = float(anchors[state][left_channel].get(band, np.nan))
    right_val = float(anchors[state][right_channel].get(band, np.nan))

    if not np.isfinite(left_val) or left_val <= 0 or not np.isfinite(right_val) or right_val <= 0:
        return

    current = left_val / max(right_val, EPS)
    if current <= 0:
        return

    delta = np.log(desired_ratio / current)
    scale = float(np.exp(np.clip(strength * delta, -0.08, 0.08)))

    anchors[state][left_channel][band] = max(left_val * scale, EPS)
    anchors[state][right_channel][band] = max(right_val / scale, EPS)


def apply_alpha_response_adjustment(
    anchors: Dict[str, Dict[str, Dict[str, float]]],
    channel: str,
    response_percent: float,
    strength: float,
) -> None:
    if channel not in anchors.get("eo", {}) or channel not in anchors.get("ec", {}):
        return

    desired_ratio = 1.0 + (response_percent / 100.0)
    if desired_ratio <= 0:
        return

    for band in ("alpha", "low_alpha", "high_alpha"):
        eo_value = float(anchors["eo"][channel].get(band, np.nan))
        ec_value = float(anchors["ec"][channel].get(band, np.nan))
        if not np.isfinite(eo_value) or eo_value <= 0:
            eo_value = 1.0
        if not np.isfinite(ec_value) or ec_value <= 0:
            ec_value = 1.0

        current_ratio = ec_value / max(eo_value, EPS)
        if current_ratio <= 0:
            continue

        delta = np.log(desired_ratio / current_ratio)
        scale = float(np.exp(np.clip(strength * delta, -0.07, 0.07)))
        anchors["ec"][channel][band] = max(ec_value * scale, EPS)
        anchors["eo"][channel][band] = max(eo_value / scale, EPS)


def apply_peak_alpha_hint(
    anchors: Dict[str, Dict[str, Dict[str, float]]],
    state: str,
    channel: str,
    peak_alpha_hz: float,
) -> None:
    if state not in anchors or channel not in anchors[state]:
        return

    if not np.isfinite(peak_alpha_hz):
        return

    shift = float(np.clip((peak_alpha_hz - 10.0) / 4.0, -1.0, 1.0))
    channel_dict = anchors[state][channel]

    low_alpha = float(channel_dict.get("low_alpha", np.nan))
    high_alpha = float(channel_dict.get("high_alpha", np.nan))
    alpha = float(channel_dict.get("alpha", np.nan))

    if not np.isfinite(low_alpha) or not np.isfinite(high_alpha):
        return

    transfer = 0.03 * shift * max((low_alpha + high_alpha) / 2.0, EPS)
    channel_dict["high_alpha"] = max(high_alpha + transfer, EPS)
    channel_dict["low_alpha"] = max(low_alpha - transfer, EPS)

    if np.isfinite(alpha):
        channel_dict["alpha"] = max(alpha * (1.0 + 0.01 * shift), EPS)


def apply_ratio_metrics_gently(
    anchors: Dict[str, Dict[str, Dict[str, float]]],
    ratio_df: pd.DataFrame,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    df = normalize_columns(ratio_df)

    metric_col = get_first_existing_column(df, ["metric"])
    site_col = get_first_existing_column(df, ["site_or_channel", "site", "channel"])
    state_col = get_first_existing_column(df, ["state_or_comparison", "state", "comparison"])
    value_col = get_first_existing_column(df, ["value", "metric_value"])

    if metric_col is None or site_col is None or state_col is None or value_col is None:
        warn("Ratio CSV is missing expected columns; skipping ratio adjustments.")
        return anchors

    adjusted = {
        "eo": {channel: bands.copy() for channel, bands in anchors.get("eo", {}).items()},
        "ec": {channel: bands.copy() for channel, bands in anchors.get("ec", {}).items()},
    }

    for _, row in df.iterrows():
        metric = normalize_text(row[metric_col])
        channels = parse_site_channels(row[site_col])
        state_hint = infer_state_from_comparison(row[state_col])
        raw_value = pd.to_numeric(row[value_col], errors="coerce")

        if pd.isna(raw_value):
            continue

        value = float(raw_value)
        states: Iterable[str]
        if state_hint in {"eo", "ec"}:
            states = [state_hint]
        else:
            states = ["eo", "ec"]

        if "theta_beta_ratio" in metric and "response" not in metric and "symmetry" not in metric:
            for state in states:
                for channel in channels:
                    apply_ratio_pair_adjustment(adjusted, state, channel, "theta", "beta", value, RATIO_ADJUSTMENT_STRENGTH)
            continue

        if "theta_alpha_ratio" in metric:
            for state in states:
                for channel in channels:
                    apply_ratio_pair_adjustment(adjusted, state, channel, "theta", "alpha", value, RATIO_ADJUSTMENT_STRENGTH)
            continue

        if "theta_low_beta_ratio" in metric:
            for state in states:
                for channel in channels:
                    apply_ratio_pair_adjustment(adjusted, state, channel, "theta", "low_beta", value, RATIO_ADJUSTMENT_STRENGTH)
            continue

        if "low_alpha_high_alpha_ratio" in metric:
            for state in states:
                for channel in channels:
                    apply_ratio_pair_adjustment(
                        adjusted,
                        state,
                        channel,
                        "low_alpha",
                        "high_alpha",
                        value,
                        RATIO_ADJUSTMENT_STRENGTH,
                    )
            continue

        if metric.endswith("_symmetry") and "f3" in channels and "f4" in channels:
            band = metric.replace("_symmetry", "")
            if band in BANDS:
                for state in states:
                    apply_symmetry_adjustment(adjusted, state, "f3", "f4", band, value, 0.08)
            continue

        if metric == "alpha_response":
            for channel in channels:
                apply_alpha_response_adjustment(adjusted, channel, value, 0.10)
            continue

        if metric == "peak_alpha":
            for state in states:
                for channel in channels:
                    apply_peak_alpha_hint(adjusted, state, channel, value)
            continue

    return adjusted


def build_topomap_lookup(topomap_df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, float]]]:
    df = normalize_columns(topomap_df)

    band_col = get_first_existing_column(df, ["band"])
    state_col = get_first_existing_column(df, ["state"])
    mean_col = get_first_existing_column(df, ["mean_z", "mean"])
    front_col = get_first_existing_column(df, ["front_mean_z", "front_z"])
    back_col = get_first_existing_column(df, ["back_mean_z", "back_z"])
    left_col = get_first_existing_column(df, ["left_mean_z", "left_z"])
    right_col = get_first_existing_column(df, ["right_mean_z", "right_z"])

    if band_col is None or state_col is None:
        warn("Topomap CSV missing required columns; skipping topomap modifiers.")
        return {}

    lookup: Dict[str, Dict[str, Dict[str, float]]] = {}

    for _, row in df.iterrows():
        band = canonical_band_name(row[band_col])
        if band is None:
            continue

        state = normalize_state_label(row[state_col])
        if state not in {"eo", "ec"}:
            continue

        lookup.setdefault(state, {})[band] = {
            "global": float(pd.to_numeric(row[mean_col], errors="coerce")) if mean_col else 0.0,
            "front": float(pd.to_numeric(row[front_col], errors="coerce")) if front_col else np.nan,
            "back": float(pd.to_numeric(row[back_col], errors="coerce")) if back_col else np.nan,
            "left": float(pd.to_numeric(row[left_col], errors="coerce")) if left_col else np.nan,
            "right": float(pd.to_numeric(row[right_col], errors="coerce")) if right_col else np.nan,
        }

    return lookup


def channel_group(channel: str) -> str:
    if channel.startswith(("fp", "af", "f")):
        return "frontal"
    if channel.startswith(("c", "fc", "cp")) or channel == "cz":
        return "central"
    if channel.startswith(("o", "p", "po", "oz")):
        return "posterior"
    return "other"


def channel_topomap_region(channel: str) -> str:
    group = channel_group(channel)
    if group == "frontal":
        return "front"
    if group == "posterior":
        return "back"
    return "global"


def channel_side(channel: str) -> Optional[str]:
    match = re.search(r"(\d+)$", channel)
    if not match:
        return None
    number = int(match.group(1))
    if number % 2 == 0:
        return "right"
    return "left"


def apply_topomap_modifiers(
    anchors: Dict[str, Dict[str, Dict[str, float]]],
    topomap_lookup: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    if not topomap_lookup:
        return anchors

    adjusted = {
        "eo": {channel: bands.copy() for channel, bands in anchors.get("eo", {}).items()},
        "ec": {channel: bands.copy() for channel, bands in anchors.get("ec", {}).items()},
    }

    for state in ("eo", "ec"):
        state_lookup = topomap_lookup.get(state, {})
        for channel, band_map in adjusted[state].items():
            region = channel_topomap_region(channel)
            side = channel_side(channel)

            for band, value in list(band_map.items()):
                if not np.isfinite(value):
                    continue

                band_lookup = state_lookup.get(band)
                if not band_lookup:
                    continue

                z_values = [band_lookup.get("global", np.nan)]
                z_values.append(band_lookup.get(region, np.nan))
                if side in {"left", "right"}:
                    z_values.append(band_lookup.get(side, np.nan))

                finite_z = [z for z in z_values if np.isfinite(z)]
                if not finite_z:
                    continue

                z = float(np.mean(finite_z))
                scale = 1.0 + TOPOMAP_MODIFIER_STRENGTH * np.tanh(z / 2.0)
                band_map[band] = max(float(value) * float(scale), EPS)

    return adjusted


def mean_profile(
    channel_dict: Dict[str, Dict[str, float]],
    channels: Sequence[str],
) -> Optional[Dict[str, float]]:
    values: Dict[str, List[float]] = {band: [] for band in BANDS}

    for channel in channels:
        if channel not in channel_dict:
            continue
        for band in BANDS:
            value = float(channel_dict[channel].get(band, np.nan))
            if np.isfinite(value):
                values[band].append(value)

    profile: Dict[str, float] = {}
    for band, band_values in values.items():
        if band_values:
            profile[band] = float(np.mean(band_values))

    if not profile:
        return None

    for band in BANDS:
        profile.setdefault(band, 1.0)

    return profile


def global_band_means(anchors: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, float]:
    values: Dict[str, List[float]] = {band: [] for band in BANDS}

    for state in ("eo", "ec"):
        for _, band_map in anchors.get(state, {}).items():
            for band in BANDS:
                value = float(band_map.get(band, np.nan))
                if np.isfinite(value):
                    values[band].append(value)

    means: Dict[str, float] = {}
    for band, band_values in values.items():
        means[band] = float(np.mean(band_values)) if band_values else 1.0
    return means


def propagate_missing_channels(
    anchors: Dict[str, Dict[str, Dict[str, float]]],
    required_channels: Sequence[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    propagated = {
        "eo": {channel: bands.copy() for channel, bands in anchors.get("eo", {}).items()},
        "ec": {channel: bands.copy() for channel, bands in anchors.get("ec", {}).items()},
    }

    global_means = global_band_means(propagated)

    for state in ("eo", "ec"):
        state_channels = propagated[state]

        frontal_profile = mean_profile(state_channels, ["f3", "f4", "fz"])
        central_profile = mean_profile(state_channels, ["cz"])
        posterior_profile = mean_profile(state_channels, ["o1"])
        all_profile = mean_profile(state_channels, list(state_channels.keys()))

        if all_profile is None:
            all_profile = global_means.copy()

        if frontal_profile is None:
            frontal_profile = all_profile.copy()
        if central_profile is None:
            central_profile = all_profile.copy()
        if posterior_profile is None:
            posterior_profile = all_profile.copy()

        for channel in required_channels:
            if channel not in state_channels:
                group = channel_group(channel)
                if group == "frontal":
                    template = frontal_profile
                elif group == "central":
                    template = central_profile
                elif group == "posterior":
                    template = posterior_profile
                else:
                    template = all_profile

                state_channels[channel] = {band: float(template.get(band, global_means[band])) for band in BANDS}
            else:
                for band in BANDS:
                    value = float(state_channels[channel].get(band, np.nan))
                    if not np.isfinite(value):
                        state_channels[channel][band] = float(global_means[band])

    return propagated


def parse_feature_name(column: str) -> Optional[Tuple[str, str, Optional[str]]]:
    if column in {"age", "gender"}:
        return None

    state: Optional[str] = None
    base = column
    if column.endswith("_eo"):
        state = "eo"
        base = column[:-3]
    elif column.endswith("_ec"):
        state = "ec"
        base = column[:-3]

    for band in sorted(BANDS, key=len, reverse=True):
        suffix = f"_{band}"
        if base.endswith(suffix):
            channel = base[: -len(suffix)]
            if channel:
                return channel, band, state

    return None


def combine_eo_ec_anchors(
    anchors: Dict[str, Dict[str, Dict[str, float]]],
    required_channels: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    combined: Dict[str, Dict[str, float]] = {}

    all_channels = set(required_channels)
    all_channels.update(anchors.get("eo", {}).keys())
    all_channels.update(anchors.get("ec", {}).keys())

    for channel in all_channels:
        combined[channel] = {}
        for band in BANDS:
            values: List[float] = []
            for state in ("eo", "ec"):
                value = float(anchors.get(state, {}).get(channel, {}).get(band, np.nan))
                if np.isfinite(value):
                    values.append(value)
            combined[channel][band] = float(np.mean(values)) if values else np.nan

    for band in BANDS:
        band_values = [
            combined[channel][band]
            for channel in combined
            if np.isfinite(combined[channel][band])
        ]
        fallback = float(np.mean(band_values)) if band_values else 1.0
        for channel in combined:
            if not np.isfinite(combined[channel][band]):
                combined[channel][band] = fallback

    return combined


def build_final_input_dataframe(
    feature_columns: Sequence[str],
    anchors: Dict[str, Dict[str, Dict[str, float]]],
    age: float,
    gender: float,
) -> pd.DataFrame:
    parsed_specs = {}
    required_channels = set()

    for column in feature_columns:
        parsed = parse_feature_name(column)
        parsed_specs[column] = parsed
        if parsed is not None:
            channel, _, _ = parsed
            required_channels.add(channel)

    propagated = propagate_missing_channels(anchors, sorted(required_channels))
    combined = combine_eo_ec_anchors(propagated, sorted(required_channels))

    global_band_defaults = {
        band: float(np.mean([combined[channel][band] for channel in combined])) if combined else 1.0
        for band in BANDS
    }

    row: Dict[str, float] = {}
    unresolved_columns: List[str] = []

    for column in feature_columns:
        if column == "age":
            row[column] = float(age)
            continue
        if column == "gender":
            row[column] = float(gender)
            continue

        spec = parsed_specs.get(column)
        if spec is None:
            unresolved_columns.append(column)
            row[column] = 0.0
            continue

        channel, band, state = spec

        if state in {"eo", "ec"}:
            value = propagated.get(state, {}).get(channel, {}).get(band, np.nan)
            if not np.isfinite(value):
                value = combined.get(channel, {}).get(band, np.nan)
        else:
            value = combined.get(channel, {}).get(band, np.nan)

        if not np.isfinite(value):
            value = global_band_defaults.get(band, 0.0)

        row[column] = float(value)

    if unresolved_columns:
        preview = ", ".join(unresolved_columns[:8])
        if len(unresolved_columns) > 8:
            preview += ", ..."
        warn(f"Could not parse {len(unresolved_columns)} schema columns; filled with 0.0. Examples: {preview}")

    return pd.DataFrame([row], columns=list(feature_columns))


def load_report_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    relative_df = load_csv_with_fallback(RELATIVE_AMPLITUDES_CSV, REPORT_FALLBACK_DIRS)
    ratio_df = load_csv_with_fallback(RATIO_METRICS_CSV, REPORT_FALLBACK_DIRS)
    topomap_df = load_csv_with_fallback(TOPOMAP_ZSCORES_CSV, REPORT_FALLBACK_DIRS)
    return relative_df, ratio_df, topomap_df


def print_predictions(predictions: np.ndarray) -> None:
    print("Predicted values:")

    values = predictions[0] if getattr(predictions, "ndim", 1) == 2 else predictions
    values = list(values)

    target_names = TARGET_NAMES
    if len(values) != len(target_names):
        target_names = [f"target_{i + 1}" for i in range(len(values))]

    for name, value in zip(target_names, values):
        print(f"- {name}: {float(value):.6f}")


def main() -> None:
    print_exploratory_warning()

    model = load_model(MODEL_PATH)
    feature_columns = load_feature_schema(FEATURE_COLUMNS_PATH)

    relative_df, ratio_df, topomap_df = load_report_tables()

    anchors = build_eo_ec_anchor_dicts(relative_df)
    anchors = apply_ratio_metrics_gently(anchors, ratio_df)

    topomap_lookup = build_topomap_lookup(topomap_df)
    anchors = apply_topomap_modifiers(anchors, topomap_lookup)

    input_df = build_final_input_dataframe(
        feature_columns=feature_columns,
        anchors=anchors,
        age=DEFAULT_AGE,
        gender=DEFAULT_GENDER,
    )

    predictions = model.predict(input_df)
    print_predictions(np.asarray(predictions))


if __name__ == "__main__":
    main()
