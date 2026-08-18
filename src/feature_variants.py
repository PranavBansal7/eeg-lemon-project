"""Construct compact baseline and exploratory EEG feature variants.

This module intentionally separates two stories:

- `FEATURE_VARIANTS` is the small public/default set used by the main benchmark
- `ADVANCED_FEATURE_VARIANTS` keeps broader exploratory ideas available without
  making them the main baseline suite
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

# These are the public/default variants because they form a simple, interpretable
# progression: start with EO+EC as separate feature spaces, add interpretable regional
# summaries, then add explicit EO-minus-EC contrast. That keeps the main benchmark easy
# to explain without leaning on more niche EEG feature engineering ideas.
FEATURE_VARIANTS = [
    "eo_ec_concat",
    "eo_ec_concat_plus_regions",
    "eo_ec_concat_plus_diff_plus_regions",
]

# These variants are still supported for exploratory or research-oriented work, but they
# are intentionally demoted out of the main story. They are useful for ablations and
# experimentation when we want to test narrower views or more handcrafted heuristics.
ADVANCED_FEATURE_VARIANTS = [
    "eo_only",
    "ec_only",
    "eo_ec_diff",
    "eo_ec_logratio",
    "eo_ec_concat_plus_diff",
    "eo_ec_concat_plus_logratio",
    "eo_ec_concat_plus_diff_plus_regions_plus_ratios",
    "eo_ec_concat_plus_diff_plus_regions_plus_ratios_plus_asymmetry",
]

ALL_FEATURE_VARIANTS = FEATURE_VARIANTS + ADVANCED_FEATURE_VARIANTS

FEATURE_VARIANT_DESCRIPTIONS = {
    "eo_ec_concat": (
        "Baseline: keep EO and EC as separate feature subspaces in the same row."
    ),
    "eo_ec_concat_plus_regions": (
        "Baseline plus interpretable region-level summaries such as frontal and occipital means."
    ),
    "eo_ec_concat_plus_diff_plus_regions": (
        "Public best-story variant: EO, EC, EO-minus-EC differences, and region summaries."
    ),
    "eo_only": "Exploratory ablation that keeps only eyes-open features.",
    "ec_only": "Exploratory ablation that keeps only eyes-closed features.",
    "eo_ec_diff": "Exploratory ablation that keeps only EO-minus-EC difference features.",
    "eo_ec_logratio": "Exploratory contrast view using log(EO+eps)-log(EC+eps).",
    "eo_ec_concat_plus_diff": "EO and EC together plus explicit EO-minus-EC differences.",
    "eo_ec_concat_plus_logratio": "EO and EC together plus log-ratio contrast features.",
    "eo_ec_concat_plus_diff_plus_regions_plus_ratios": (
        "Exploratory extension that adds handcrafted within-channel ratio features."
    ),
    "eo_ec_concat_plus_diff_plus_regions_plus_ratios_plus_asymmetry": (
        "Most handcrafted exploratory variant: ratios plus frontal alpha asymmetry."
    ),
}

BAND_SUFFIXES = [
    "low_alpha",
    "high_alpha",
    "low_beta",
    "high_beta",
    "delta",
    "theta",
    "alpha",
    "beta",
]

REGION_PREFIX_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("frontal", ("fp", "af", "f")),
    ("central", ("fc", "c", "cp")),
    ("parietal", ("p",)),
    ("occipital", ("o",)),
    ("temporal", ("ft", "tp", "t")),
]

RATIO_FEATURE_SPECS: List[Tuple[str, str, str]] = [
    ("theta_beta_logratio", "theta", "beta"),
    ("theta_alpha_logratio", "theta", "alpha"),
    ("lowalpha_highalpha_logratio", "low_alpha", "high_alpha"),
]


def _sorted_by_base(mapping: Dict[str, str]) -> List[Tuple[str, str]]:
    return sorted(mapping.items(), key=lambda item: item[0])


def describe_feature_variant(name: str) -> str:
    """Return the short human explanation for a feature variant name."""
    if name not in FEATURE_VARIANT_DESCRIPTIONS:
        raise ValueError(f"Unknown feature variant: {name}")
    return FEATURE_VARIANT_DESCRIPTIONS[name]


def split_eo_ec_columns(columns: Sequence[str]) -> Tuple[Dict[str, str], Dict[str, str], List[str]]:
    eo_map: Dict[str, str] = {}
    ec_map: Dict[str, str] = {}
    context_columns: List[str] = []

    for col in columns:
        if col.endswith("_eo"):
            base = col[:-3]
            if not base:
                raise ValueError(f"Invalid EO feature column name: {col}")
            if base in eo_map:
                raise ValueError(f"Duplicate EO base feature detected for base '{base}'.")
            eo_map[base] = col
        elif col.endswith("_ec"):
            base = col[:-3]
            if not base:
                raise ValueError(f"Invalid EC feature column name: {col}")
            if base in ec_map:
                raise ValueError(f"Duplicate EC base feature detected for base '{base}'.")
            ec_map[base] = col
        else:
            context_columns.append(col)

    if not eo_map:
        raise ValueError("No EO feature columns detected from suffix '_eo'.")
    if not ec_map:
        raise ValueError("No EC feature columns detected from suffix '_ec'.")

    return eo_map, ec_map, sorted(context_columns)


def require_safe_pairing(eo_map: Dict[str, str], ec_map: Dict[str, str]) -> List[str]:
    eo_bases = set(eo_map.keys())
    ec_bases = set(ec_map.keys())

    only_eo = sorted(eo_bases - ec_bases)
    only_ec = sorted(ec_bases - eo_bases)
    if only_eo or only_ec:
        raise ValueError(
            "EO/EC feature pairing mismatch detected. "
            f"Unmatched EO bases: {len(only_eo)}, unmatched EC bases: {len(only_ec)}"
        )

    return sorted(eo_bases)


def parse_channel_and_band(base_feature_name: str) -> Tuple[str, str]:
    for band in sorted(BAND_SUFFIXES, key=len, reverse=True):
        suffix = f"_{band}"
        if base_feature_name.endswith(suffix):
            channel = base_feature_name[: -len(suffix)]
            if not channel:
                raise ValueError(f"Missing channel prefix in feature base: {base_feature_name}")
            return channel, band

    raise ValueError(f"Could not parse EEG band suffix from feature base: {base_feature_name}")


def infer_region(channel_name: str) -> str:
    channel = channel_name.strip().lower()

    region_rules = {
        "frontal": ["fp", "af", "f"],
        "central": ["fc", "cp", "c"],
        "parietal": ["p"],
        "occipital": ["o"],
        "temporal": ["ft", "tp", "t"],
    }

    matches = []
    for region, prefixes in region_rules.items():
        for prefix in prefixes:
            if channel.startswith(prefix):
                matches.append((len(prefix), region, prefix))

    if not matches:
        raise ValueError(f"Unknown region mapping for channel '{channel_name}'")

    max_len = max(m[0] for m in matches)
    best = [(region, prefix) for length, region, prefix in matches if length == max_len]

    unique_regions = sorted({region for region, _ in best})
    if len(unique_regions) == 1:
        return unique_regions[0]

    raise ValueError(
        f"Ambiguous region mapping for channel '{channel_name}': {unique_regions}"
    )


def build_regional_mean_features(X: pd.DataFrame, eo_map: Dict[str, str], ec_map: Dict[str, str], paired_bases: List[str]) -> pd.DataFrame:
    bands_seen: List[str] = []
    grouped_columns: Dict[Tuple[str, str, str], List[str]] = {}

    for base in paired_bases:
        channel, band = parse_channel_and_band(base)
        region = infer_region(channel)
        if band not in bands_seen:
            bands_seen.append(band)

        grouped_columns.setdefault(("eo", region, band), []).append(eo_map[base])
        grouped_columns.setdefault(("ec", region, band), []).append(ec_map[base])

    regional_features: Dict[str, pd.Series] = {}
    for state in ("eo", "ec"):
        for region_name, _ in REGION_PREFIX_RULES:
            for band in bands_seen:
                key = (state, region_name, band)
                cols = grouped_columns.get(key, [])
                if len(cols) == 0:
                    raise ValueError(
                        "Insufficient regional coverage to compute region means safely. "
                        f"Missing {state}/{region_name}/{band}."
                    )
                regional_features[f"region_{region_name}_{band}_{state}"] = X[cols].mean(axis=1)

    return pd.DataFrame(regional_features, index=X.index)


def build_channel_band_index(feature_map: Dict[str, str], state_name: str) -> Dict[str, Dict[str, str]]:
    channel_band_index: Dict[str, Dict[str, str]] = {}

    for base, col in _sorted_by_base(feature_map):
        channel, band = parse_channel_and_band(base)
        band_map = channel_band_index.setdefault(channel, {})
        if band in band_map:
            raise ValueError(
                f"Duplicate {state_name} channel/band feature detected for channel '{channel}', band '{band}'."
            )
        band_map[band] = col

    return channel_band_index


def ensure_logratio_inputs_valid(X: pd.DataFrame, columns: Sequence[str], feature_name: str, eps: float) -> None:
    invalid_mask = (X[list(columns)] <= -eps).any(axis=1)
    if bool(invalid_mask.any()):
        raise ValueError(
            f"Found values <= -eps while constructing '{feature_name}'; cannot compute stable log-ratio features safely."
        )


def compute_logratio_series(
    X: pd.DataFrame,
    numerator_col: str,
    denominator_col: str,
    feature_name: str,
    eps: float,
) -> pd.Series:
    ensure_logratio_inputs_valid(
        X=X,
        columns=[numerator_col, denominator_col],
        feature_name=feature_name,
        eps=eps,
    )
    return np.log(X[numerator_col] + eps) - np.log(X[denominator_col] + eps)


def build_per_channel_ratio_features(X: pd.DataFrame, eo_map: Dict[str, str], ec_map: Dict[str, str], eps: float) -> pd.DataFrame:
    derived_features: Dict[str, pd.Series] = {}

    for state_name, feature_map in (("eo", eo_map), ("ec", ec_map)):
        channel_band_index = build_channel_band_index(feature_map=feature_map, state_name=state_name)

        for channel in sorted(channel_band_index):
            band_map = channel_band_index[channel]

            for ratio_name, numerator_band, denominator_band in RATIO_FEATURE_SPECS:
                numerator_col = band_map.get(numerator_band)
                denominator_col = band_map.get(denominator_band)

                if numerator_col is None and denominator_col is None:
                    continue
                if numerator_col is None or denominator_col is None:
                    raise ValueError(
                        "Missing required band pairing while constructing per-channel ratio feature "
                        f"'{channel}_{ratio_name}_{state_name}'. Required bands: '{numerator_band}' and "
                        f"'{denominator_band}'."
                    )

                feature_name = f"{channel}_{ratio_name}_{state_name}"
                derived_features[feature_name] = compute_logratio_series(
                    X=X,
                    numerator_col=numerator_col,
                    denominator_col=denominator_col,
                    feature_name=feature_name,
                    eps=eps,
                )

    if not derived_features:
        raise ValueError("No per-channel ratio features could be constructed safely.")

    return pd.DataFrame(derived_features, index=X.index)


def build_frontal_alpha_asymmetry_features(
    X: pd.DataFrame,
    eo_map: Dict[str, str],
    ec_map: Dict[str, str],
    eps: float,
) -> pd.DataFrame:
    asymmetry_features: Dict[str, pd.Series] = {}

    for state_name, feature_map in (("eo", eo_map), ("ec", ec_map)):
        channel_band_index = build_channel_band_index(feature_map=feature_map, state_name=state_name)
        f3_alpha_col = channel_band_index.get("f3", {}).get("alpha")
        f4_alpha_col = channel_band_index.get("f4", {}).get("alpha")

        if f3_alpha_col is None or f4_alpha_col is None:
            raise ValueError(
                f"Missing F3/F4 alpha features for state '{state_name}'; cannot compute frontal alpha asymmetry safely."
            )

        feature_name = f"frontal_alpha_asymmetry_{state_name}"
        asymmetry_features[feature_name] = compute_logratio_series(
            X=X,
            numerator_col=f4_alpha_col,
            denominator_col=f3_alpha_col,
            feature_name=feature_name,
            eps=eps,
        )

    return pd.DataFrame(asymmetry_features, index=X.index)


def build_feature_variant(X: pd.DataFrame, variant: str, eps: float) -> pd.DataFrame:
    if variant not in ALL_FEATURE_VARIANTS:
        raise ValueError(f"Unknown feature variant: {variant}")
    if eps <= 0.0:
        raise ValueError("eps must be > 0 for log-ratio stability.")

    eo_map, ec_map, context_columns = split_eo_ec_columns(X.columns)
    paired_bases = require_safe_pairing(eo_map, ec_map)

    eo_columns = [col for _, col in _sorted_by_base(eo_map)]
    ec_columns = [col for _, col in _sorted_by_base(ec_map)]

    out_parts: List[pd.DataFrame] = []

    # The default progression keeps EO/EC separate,
    # optionally add region summaries, then add explicit EO-minus-EC contrast.
    # Narrower ablations such as EO-only and EC-only still stay available here.
    if variant == "eo_only":
        out_parts.append(X[eo_columns].copy())
    elif variant == "ec_only":
        out_parts.append(X[ec_columns].copy())
    elif variant == "eo_ec_concat":
        out_parts.append(X[eo_columns].copy())
        out_parts.append(X[ec_columns].copy())
    elif variant == "eo_ec_diff":
        diff_dict = {f"{base}_diff": X[eo_map[base]] - X[ec_map[base]] for base in paired_bases}
        out_parts.append(pd.DataFrame(diff_dict, index=X.index))
    elif variant == "eo_ec_logratio":
        invalid_mask = (X[[eo_map[base] for base in paired_bases]] <= -eps).any(axis=1) | (
            X[[ec_map[base] for base in paired_bases]] <= -eps
        ).any(axis=1)
        if bool(invalid_mask.any()):
            raise ValueError("Found EO/EC values <= -eps; cannot compute stable log-ratio features safely.")

        logratio_dict = {
            f"{base}_logratio": np.log(X[eo_map[base]] + eps) - np.log(X[ec_map[base]] + eps)
            for base in paired_bases
        }
        out_parts.append(pd.DataFrame(logratio_dict, index=X.index))
    elif variant == "eo_ec_concat_plus_diff":
        out_parts.append(X[eo_columns].copy())
        out_parts.append(X[ec_columns].copy())
        diff_dict = {f"{base}_diff": X[eo_map[base]] - X[ec_map[base]] for base in paired_bases}
        out_parts.append(pd.DataFrame(diff_dict, index=X.index))
    elif variant == "eo_ec_concat_plus_logratio":
        invalid_mask = (X[[eo_map[base] for base in paired_bases]] <= -eps).any(axis=1) | (
            X[[ec_map[base] for base in paired_bases]] <= -eps
        ).any(axis=1)
        if bool(invalid_mask.any()):
            raise ValueError("Found EO/EC values <= -eps; cannot compute stable log-ratio features safely.")

        out_parts.append(X[eo_columns].copy())
        out_parts.append(X[ec_columns].copy())
        logratio_dict = {
            f"{base}_logratio": np.log(X[eo_map[base]] + eps) - np.log(X[ec_map[base]] + eps)
            for base in paired_bases
        }
        out_parts.append(pd.DataFrame(logratio_dict, index=X.index))
    elif variant == "eo_ec_concat_plus_regions":
        out_parts.append(X[eo_columns].copy())
        out_parts.append(X[ec_columns].copy())
        out_parts.append(build_regional_mean_features(X, eo_map=eo_map, ec_map=ec_map, paired_bases=paired_bases))
    elif variant == "eo_ec_concat_plus_diff_plus_regions":
        out_parts.append(X[eo_columns].copy())
        out_parts.append(X[ec_columns].copy())
        diff_dict = {f"{base}_diff": X[eo_map[base]] - X[ec_map[base]] for base in paired_bases}
        out_parts.append(pd.DataFrame(diff_dict, index=X.index))
        out_parts.append(build_regional_mean_features(X, eo_map=eo_map, ec_map=ec_map, paired_bases=paired_bases))
    # Advanced variants stay available for exploratory comparisons, but they are
    # intentionally not part of the default benchmark story.
    elif variant == "eo_ec_concat_plus_diff_plus_regions_plus_ratios":
        out_parts.append(X[eo_columns].copy())
        out_parts.append(X[ec_columns].copy())
        diff_dict = {f"{base}_diff": X[eo_map[base]] - X[ec_map[base]] for base in paired_bases}
        out_parts.append(pd.DataFrame(diff_dict, index=X.index))
        out_parts.append(build_regional_mean_features(X, eo_map=eo_map, ec_map=ec_map, paired_bases=paired_bases))
        out_parts.append(build_per_channel_ratio_features(X, eo_map=eo_map, ec_map=ec_map, eps=eps))
    elif variant == "eo_ec_concat_plus_diff_plus_regions_plus_ratios_plus_asymmetry":
        out_parts.append(X[eo_columns].copy())
        out_parts.append(X[ec_columns].copy())
        diff_dict = {f"{base}_diff": X[eo_map[base]] - X[ec_map[base]] for base in paired_bases}
        out_parts.append(pd.DataFrame(diff_dict, index=X.index))
        out_parts.append(build_regional_mean_features(X, eo_map=eo_map, ec_map=ec_map, paired_bases=paired_bases))
        out_parts.append(build_per_channel_ratio_features(X, eo_map=eo_map, ec_map=ec_map, eps=eps))
        out_parts.append(build_frontal_alpha_asymmetry_features(X, eo_map=eo_map, ec_map=ec_map, eps=eps))

    if context_columns:
        out_parts.append(X[context_columns].copy())

    if not out_parts:
        raise ValueError(f"No features created for variant '{variant}'.")

    out = pd.concat(out_parts, axis=1)
    if out.empty:
        raise ValueError(f"Variant '{variant}' generated an empty feature matrix.")

    return out
