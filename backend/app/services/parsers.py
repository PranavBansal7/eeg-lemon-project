"""Parse uploaded demo inputs into the anchor representation used by the API."""

from __future__ import annotations

import csv
import io
from typing import Any, Mapping

from fastapi import UploadFile

from app.config import ANCHOR_CHANNELS, BANDS, CONDITIONS


def parse_gender_value(value: Any) -> float:
    """Accept flexible gender inputs and normalize them to the model's numeric form."""
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().lower()
    male_values = {"m", "male", "man", "boy"}
    female_values = {"f", "female", "woman", "girl"}

    if text in male_values:
        return 1.0
    if text in female_values:
        return 0.0

    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(
            "Gender must be numeric (e.g., 0/1) or one of: male/female."
        ) from exc


def parse_float_value(value: Any, field_name: str) -> float:
    """Convert a value to float with a field-specific error message."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a numeric value.") from exc


async def parse_csv_upload(file: UploadFile) -> list[dict[str, Any]]:
    """Read an uploaded CSV file into row dictionaries for later parsing."""
    content = await file.read()
    if not content:
        raise ValueError("Uploaded CSV file is empty.")

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("latin-1")

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("CSV header row is missing.")

    rows = []
    for row in reader:
        if any(str(value).strip() for value in row.values() if value is not None):
            rows.append(row)

    if not rows:
        raise ValueError("CSV has no data rows.")

    return rows


def extract_anchor_payload_from_row(
    row: Mapping[str, Any],
    row_number: int,
) -> tuple[float, float, dict[str, dict[str, dict[str, float]]]]:
    """Convert one CSV row into age, gender, and nested anchor EEG values.

    The parser accepts several header styles so the demo CSV workflow stays easy
    to test. The normalized output matches the `anchors[condition][channel][band]`
    shape used by the rest of the backend.
    """
    normalized_row = {
        _normalize_key(key): value for key, value in row.items() if key is not None
    }

    age_raw = _lookup_value(normalized_row, ["age"])
    if age_raw in (None, ""):
        raise ValueError(f"Row {row_number}: missing required column 'age'.")
    age = parse_float_value(age_raw, f"Row {row_number} age")

    gender_raw = _lookup_value(normalized_row, ["gender", "sex"])
    if gender_raw in (None, ""):
        raise ValueError(f"Row {row_number}: missing required column 'gender'.")
    gender = parse_gender_value(gender_raw)

    anchors: dict[str, dict[str, dict[str, float]]] = {
        condition: {channel: {} for channel in ANCHOR_CHANNELS} for condition in CONDITIONS
    }

    missing_features: list[str] = []
    for condition in CONDITIONS:
        for channel in ANCHOR_CHANNELS:
            for band in BANDS:
                # Accept a few common conditioned header layouts so uploaded CSVs
                # can stay readable without forcing one exact naming pattern.
                conditioned_candidates = [
                    f"{channel}_{band}_{condition}",
                    f"{condition}_{channel}_{band}",
                    f"{channel}_{condition}_{band}",
                    f"{condition}.{channel}.{band}",
                    f"{channel}.{band}.{condition}",
                ]

                value = _lookup_value(normalized_row, conditioned_candidates)
                if value in (None, ""):
                    value = _lookup_value(
                        normalized_row,
                        [f"{channel}_{band}", f"{channel}.{band}"],
                    )

                if value in (None, ""):
                    missing_features.append(f"{channel}_{band}_{condition}")
                    continue

                anchors[condition][channel][band] = parse_float_value(
                    value,
                    f"Row {row_number} feature {channel}_{band}_{condition}",
                )

    if missing_features:
        preview = ", ".join(missing_features[:8])
        if len(missing_features) > 8:
            preview += f", ... ({len(missing_features)} missing total)"
        raise ValueError(f"Row {row_number}: missing EEG features: {preview}")

    return age, gender, anchors


def _lookup_value(row: Mapping[str, Any], candidates: list[str]) -> Any:
    """Return the first matching value from a list of candidate header names."""
    for candidate in candidates:
        key = _normalize_key(candidate)
        if key in row:
            return row[key]
    return None


def _normalize_key(text: str) -> str:
    """Normalize headers so simple punctuation differences do not matter."""
    return text.strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")
