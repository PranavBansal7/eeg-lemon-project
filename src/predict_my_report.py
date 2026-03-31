from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "rf_model.pkl"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "processed" / "feature_columns.json"


TARGET_NAMES = [
	"attention",
	"fluid_intelligence",
	"working_memory",
	"executive_function",
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


# =========================== PERSONAL_FEATURES ===========================
# Enter your own values here. Keep names in this format:
#   <channel_name_lowercase>_<band>
# Example keys:
#   "fp1_delta", "fp1_theta", "f3_alpha", "o2_beta"
#
# Keep band names exactly as:
# delta, theta, low_alpha, alpha, high_alpha, low_beta, beta, high_beta
#
# If you do not provide a feature that the model expects,
# the script will warn you and fill a default value.
PERSONAL_FEATURES: Dict[str, float] = {
	"age": 30.0,
	"gender": 0.0,  # use your numeric mapping from training metadata

	# Example EEG entries (replace with your own values):
	"fp1_delta": 0.0,
	"fp1_theta": 0.0,
	"fp1_low_alpha": 0.0,
	"fp1_alpha": 0.0,
	"fp1_high_alpha": 0.0,
	"fp1_low_beta": 0.0,
	"fp1_beta": 0.0,
	"fp1_high_beta": 0.0,
}
# =======================================================================


def load_feature_schema(schema_path: Path) -> List[str]:
	if not schema_path.exists():
		raise FileNotFoundError(
			f"Feature schema not found at: {schema_path}. "
			"Run training first to create processed/feature_columns.json."
		)

	with schema_path.open("r", encoding="utf-8") as file_handle:
		feature_columns = json.load(file_handle)

	if not isinstance(feature_columns, list) or not feature_columns:
		raise ValueError("Feature schema is empty or invalid.")

	return [str(column) for column in feature_columns]


def load_model(model_path: Path):
	if not model_path.exists():
		raise FileNotFoundError(
			f"Model file not found at: {model_path}. "
			"Run training first to create models/rf_model.pkl."
		)
	return joblib.load(model_path)


def build_personal_dataframe(
	personal_features: Dict[str, float],
	schema_columns: List[str],
) -> pd.DataFrame:
	band_set = set(BANDS)
	cleaned_features: Dict[str, float] = {}
	for key, value in personal_features.items():
		feature_name = str(key)
		if feature_name not in {"age", "gender"}:
			if "_" not in feature_name:
				warnings.warn(
					f"Feature '{feature_name}' does not match '<channel>_<band>' naming.",
					stacklevel=2,
				)
			else:
				band_name = feature_name.rsplit("_", 1)[-1]
				if band_name not in band_set:
					warnings.warn(
						f"Feature '{feature_name}' uses unsupported band '{band_name}'.",
						stacklevel=2,
					)

		cleaned_features[feature_name] = float(value)

	missing_columns = [column for column in schema_columns if column not in cleaned_features]
	extra_columns = [column for column in cleaned_features if column not in schema_columns]

	default_values: Dict[str, float] = {column: 0.0 for column in schema_columns}
	default_values["age"] = 30.0
	default_values["gender"] = 0.0

	if missing_columns:
		warning_preview = ", ".join(missing_columns[:12])
		if len(missing_columns) > 12:
			warning_preview += ", ..."
		warnings.warn(
			"Missing features were filled with defaults. "
			f"Total missing: {len(missing_columns)}. "
			f"Examples: {warning_preview}",
			stacklevel=2,
		)

	if extra_columns:
		warning_preview = ", ".join(extra_columns[:12])
		if len(extra_columns) > 12:
			warning_preview += ", ..."
		warnings.warn(
			"Extra features not present in training schema were ignored. "
			f"Total extra: {len(extra_columns)}. "
			f"Examples: {warning_preview}",
			stacklevel=2,
		)

	row_data: Dict[str, float] = {}
	for column in schema_columns:
		if column in cleaned_features:
			row_data[column] = cleaned_features[column]
		else:
			row_data[column] = default_values.get(column, 0.0)

	personal_df = pd.DataFrame([row_data], columns=schema_columns)
	return personal_df


def print_exploratory_warning() -> None:
	print("\n" + "=" * 90)
	print("WARNING: EXPLORATORY PERSONAL PREDICTION ONLY")
	print("This output is NOT scientifically reliable for clinical or real-world interpretation.")
	print("Your personal report is not equivalent to full raw LEMON EEG acquisition and preprocessing.")
	print("Do not use this result for diagnosis, treatment, high-stakes decisions, or self-evaluation.")
	print("=" * 90 + "\n")


def print_predictions(predictions, target_names: List[str]) -> None:
	print("Predicted values:")

	if hasattr(predictions, "shape") and len(predictions.shape) == 2:
		values = predictions[0]
	else:
		values = predictions

	values = list(values)
	if len(values) != len(target_names):
		target_names = [f"target_{index + 1}" for index in range(len(values))]

	for target_name, predicted_value in zip(target_names, values):
		print(f"- {target_name}: {float(predicted_value):.6f}")


def main() -> None:
	warnings.simplefilter("always", UserWarning)

	print_exploratory_warning()

	model = load_model(MODEL_PATH)
	feature_columns = load_feature_schema(FEATURE_COLUMNS_PATH)
	personal_df = build_personal_dataframe(PERSONAL_FEATURES, feature_columns)

	predictions = model.predict(personal_df)
	print_predictions(predictions, TARGET_NAMES)


if __name__ == "__main__":
	main()
