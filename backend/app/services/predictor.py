from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np

from app.config import (
    ANCHOR_CHANNELS,
    BANDS,
    CONDITIONS,
    FEATURE_SCHEMA_PATH,
    MODEL_PATH,
    TARGET_NAMES,
)
from app.schemas import ManualPredictionRequest
from app.services.feature_builder import FeatureBuilder
from app.services.parsers import parse_gender_value


class PredictionService:
    """Load trained artifacts and provide prediction helpers for all API routes."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        schema_path: Path = FEATURE_SCHEMA_PATH,
        target_names: Sequence[str] = TARGET_NAMES,
    ) -> None:
        self.model_path = model_path
        self.schema_path = schema_path
        self.target_names = list(target_names)

        self.model: Any = None
        self.feature_columns: list[str] = []
        self.feature_builder: FeatureBuilder | None = None

    def load(self) -> "PredictionService":
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        if not self.schema_path.exists():
            raise FileNotFoundError(f"Feature schema file not found at {self.schema_path}")

        self.model = joblib.load(self.model_path)

        raw_schema = json.loads(self.schema_path.read_text())
        if isinstance(raw_schema, dict) and "feature_columns" in raw_schema:
            columns = raw_schema["feature_columns"]
        elif isinstance(raw_schema, list):
            columns = raw_schema
        else:
            raise ValueError(
                "Feature schema must be a list or a dict with 'feature_columns'."
            )

        self.feature_columns = [str(column) for column in columns]
        self.feature_builder = FeatureBuilder(self.feature_columns)
        return self

    def predict_from_manual(self, payload: ManualPredictionRequest) -> dict[str, float]:
        anchors = self._manual_payload_to_anchors(payload)
        return self.predict_from_anchors(
            age=payload.age,
            gender=payload.gender,
            anchors=anchors,
        )

    def predict_from_anchors(
        self,
        age: float,
        gender: Any,
        anchors: Mapping[str, Mapping[str, Mapping[str, Any]]],
    ) -> dict[str, float]:
        self._ensure_loaded()

        age_value = float(age)
        gender_value = parse_gender_value(gender)
        normalized_anchors = self._normalize_anchors(anchors)

        feature_vector = self.feature_builder.build_vector(
            normalized_anchors,
            age=age_value,
            gender=gender_value,
        )

        raw_output = self.model.predict(np.array([feature_vector], dtype=float))
        prediction_row = self._extract_prediction_rows(raw_output, expected_rows=1)[0]
        return self._to_target_dict(prediction_row)

    def predict_batch(
        self,
        records: Sequence[tuple[float, Any, Mapping[str, Mapping[str, Mapping[str, Any]]]]],
    ) -> list[dict[str, float]]:
        self._ensure_loaded()

        feature_vectors: list[list[float]] = []
        for age, gender, anchors in records:
            age_value = float(age)
            gender_value = parse_gender_value(gender)
            normalized_anchors = self._normalize_anchors(anchors)

            feature_vectors.append(
                self.feature_builder.build_vector(
                    normalized_anchors,
                    age=age_value,
                    gender=gender_value,
                )
            )

        raw_output = self.model.predict(np.array(feature_vectors, dtype=float))
        rows = self._extract_prediction_rows(raw_output, expected_rows=len(records))
        return [self._to_target_dict(row) for row in rows]

    def _ensure_loaded(self) -> None:
        if self.model is None or self.feature_builder is None:
            raise RuntimeError("PredictionService is not loaded. Call load() first.")

    @staticmethod
    def _manual_payload_to_anchors(
        payload: ManualPredictionRequest,
    ) -> dict[str, dict[str, dict[str, float]]]:
        def dump_model(model: Any) -> dict[str, Any]:
            if hasattr(model, "model_dump"):
                return model.model_dump()
            return model.dict()

        return {
            "eo": dump_model(payload.eo),
            "ec": dump_model(payload.ec),
        }

    @staticmethod
    def _normalize_anchors(
        anchors: Mapping[str, Mapping[str, Mapping[str, Any]]],
    ) -> dict[str, dict[str, dict[str, float]]]:
        normalized: dict[str, dict[str, dict[str, float]]] = {}

        for condition, channels in anchors.items():
            condition_key = str(condition).lower()
            if condition_key not in CONDITIONS:
                continue

            normalized.setdefault(condition_key, {})
            for channel, band_values in channels.items():
                channel_key = str(channel).lower()
                normalized[condition_key].setdefault(channel_key, {})

                if not isinstance(band_values, Mapping):
                    continue

                for band, value in band_values.items():
                    band_key = str(band).lower()
                    if band_key not in BANDS:
                        continue

                    try:
                        normalized[condition_key][channel_key][band_key] = float(value)
                    except (TypeError, ValueError):
                        continue

        for condition in CONDITIONS:
            normalized.setdefault(condition, {})
            for channel in ANCHOR_CHANNELS:
                normalized[condition].setdefault(channel, {})

        return normalized

    def _extract_prediction_rows(self, raw_output: Any, expected_rows: int) -> list[list[float]]:
        output_array = np.asarray(raw_output, dtype=float)

        if output_array.ndim == 1:
            if expected_rows == 1:
                output_array = output_array.reshape(1, -1)
            else:
                output_array = output_array.reshape(expected_rows, -1)
        elif output_array.ndim > 2:
            output_array = output_array.reshape(expected_rows, -1)

        return output_array.tolist()

    def _to_target_dict(self, values: Sequence[float]) -> dict[str, float]:
        if len(values) < len(self.target_names):
            raise RuntimeError(
                "Model output has fewer targets than expected. "
                f"Expected {len(self.target_names)}, got {len(values)}."
            )

        return {
            target_name: float(values[index])
            for index, target_name in enumerate(self.target_names)
        }
