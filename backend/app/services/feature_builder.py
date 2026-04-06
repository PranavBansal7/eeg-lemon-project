"""Build full model feature vectors from a small anchor-input representation.

The training pipeline saves a wide feature schema. The demo API does not ask the
user for every one of those values. Instead, it asks for a small set of anchor
channels and expands those anchors into the full saved schema with simple,
explainable region-based fallbacks.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from app.config import (
    ANCHOR_CHANNELS,
    CONDITIONS,
    FRONTAL_ANCHORS,
    BANDS,
)


AnchorValues = Mapping[str, Mapping[str, Mapping[str, float]]]


class FeatureBuilder:
    """Expand anchor EEG inputs into the full feature vector expected by the model."""

    def __init__(self, feature_columns: Sequence[str]) -> None:
        # Store the saved schema in normalized lowercase form so lookup stays
        # consistent across manual input, CSV upload, and demo workflows.
        self.feature_columns = [col.lower() for col in feature_columns]

    def build_vector(self, anchors: AnchorValues, age: float, gender: float) -> list[float]:
        """Walk the saved feature schema in order and fill each feature value.

        Direct anchor values are used first. When a requested feature is not an
        anchor channel, the builder falls back to simple region-based estimates.
        This keeps the demo path easy to explain while still matching the saved
        training schema exactly.
        """
        vector: list[float] = []

        for feature_name in self.feature_columns:
            if feature_name == "age":
                vector.append(float(age))
                continue

            if feature_name == "gender":
                vector.append(float(gender))
                continue

            conditioned_parts = self._parse_conditioned_feature(feature_name)
            if conditioned_parts is not None:
                # Conditioned feature example: `f3_alpha_eo`
                channel, band, condition = conditioned_parts
                vector.append(
                    self._estimate_conditioned_channel_value(anchors, channel, band, condition)
                )
                continue

            plain_parts = self._parse_plain_feature(feature_name)
            if plain_parts is not None:
                # Plain feature example: `f3_alpha`; use both conditions if needed.
                channel, band = plain_parts
                vector.append(self._estimate_plain_channel_value(anchors, channel, band))
                continue

            # Unknown or unsupported schema items fall back to zero so the vector
            # length still matches the saved training schema.
            vector.append(0.0)

        return vector

    @staticmethod
    def _parse_conditioned_feature(feature_name: str) -> Optional[tuple[str, str, str]]:
        if not (feature_name.endswith("_eo") or feature_name.endswith("_ec")):
            return None

        base_name, condition = feature_name.rsplit("_", 1)
        if "_" not in base_name:
            return None

        channel, band = base_name.split("_", 1)
        return channel, band, condition

    @staticmethod
    def _parse_plain_feature(feature_name: str) -> Optional[tuple[str, str]]:
        if "_" not in feature_name:
            return None

        channel, band = feature_name.split("_", 1)
        if channel in {"age", "gender"}:
            return None

        return channel, band

    def _estimate_conditioned_channel_value(
        self,
        anchors: AnchorValues,
        channel: str,
        band: str,
        condition: str,
    ) -> float:
        # First prefer the exact anchor value. If that channel was not provided,
        # fall back to a simple region estimate for the same condition.
        direct_value = self._anchor_value(anchors, condition, channel, band)
        if direct_value is not None:
            return direct_value

        group = self._channel_group(channel)
        if group == "frontal":
            return self._frontal_condition_mean(anchors, condition, band)
        if group == "central":
            return self._central_condition_value(anchors, condition, band)
        if group == "posterior":
            return self._posterior_condition_value(anchors, condition, band)

        return self._condition_band_mean(anchors, condition, band)

    def _estimate_plain_channel_value(
        self,
        anchors: AnchorValues,
        channel: str,
        band: str,
    ) -> float:
        # Plain features are estimated across EO and EC because they are not tied
        # to one condition in the saved schema.
        if channel in ANCHOR_CHANNELS:
            channel_values = [
                self._anchor_value(anchors, condition, channel, band)
                for condition in CONDITIONS
            ]
            channel_values = [value for value in channel_values if value is not None]
            if channel_values:
                return self._mean(channel_values)

        group = self._channel_group(channel)
        if group == "frontal":
            return self._mean(
                [self._frontal_condition_mean(anchors, condition, band) for condition in CONDITIONS]
            )
        if group == "central":
            return self._mean(
                [self._central_condition_value(anchors, condition, band) for condition in CONDITIONS]
            )
        if group == "posterior":
            return self._mean(
                [
                    self._posterior_condition_value(anchors, condition, band)
                    for condition in CONDITIONS
                ]
            )

        return self._plain_band_mean(anchors, band)

    @staticmethod
    def _channel_group(channel: str) -> str:
        """Assign a channel to a simple region bucket for demo-time estimation."""
        c = channel.lower()

        if c.startswith(("fp", "af", "f", "fc", "ft")):
            return "frontal"

        if c == "cz" or c.startswith(("c", "cp")):
            return "central"

        if c.startswith(("p", "po", "o", "t", "tp")):
            return "posterior"

        return "other"

    def _anchor_value(
        self,
        anchors: AnchorValues,
        condition: str,
        channel: str,
        band: str,
    ) -> Optional[float]:
        """Return an exact anchor value when it exists."""
        try:
            value = anchors[condition][channel][band]
            return float(value)
        except (KeyError, TypeError, ValueError):
            return None

    def _frontal_condition_mean(
        self,
        anchors: AnchorValues,
        condition: str,
        band: str,
    ) -> float:
        """Use the mean of the frontal anchors F3/F4/FZ for frontal fallbacks."""
        values = [
            self._anchor_value(anchors, condition, frontal_channel, band)
            for frontal_channel in FRONTAL_ANCHORS
        ]
        values = [value for value in values if value is not None]
        if values:
            return self._mean(values)
        return self._condition_band_mean(anchors, condition, band)

    def _central_condition_value(
        self,
        anchors: AnchorValues,
        condition: str,
        band: str,
    ) -> float:
        value = self._anchor_value(anchors, condition, "cz", band)
        if value is not None:
            return value
        return self._condition_band_mean(anchors, condition, band)

    def _posterior_condition_value(
        self,
        anchors: AnchorValues,
        condition: str,
        band: str,
    ) -> float:
        value = self._anchor_value(anchors, condition, "o1", band)
        if value is not None:
            return value
        return self._condition_band_mean(anchors, condition, band)

    def _condition_band_mean(
        self,
        anchors: AnchorValues,
        condition: str,
        band: str,
    ) -> float:
        """Last-condition fallback: average whatever anchor values exist for that band."""
        values = [
            self._anchor_value(anchors, condition, channel, band)
            for channel in ANCHOR_CHANNELS
        ]
        values = [value for value in values if value is not None]
        if values:
            return self._mean(values)

        condition_values = []
        for channel in ANCHOR_CHANNELS:
            band_values = anchors.get(condition, {}).get(channel, {})
            condition_values.extend(
                [float(v) for b, v in band_values.items() if b in BANDS and v is not None]
            )

        if condition_values:
            return self._mean(condition_values)

        return 0.0

    def _plain_band_mean(self, anchors: AnchorValues, band: str) -> float:
        """Last-global fallback: average the band across EO and EC."""
        values = [self._condition_band_mean(anchors, condition, band) for condition in CONDITIONS]
        values = [value for value in values if value is not None]
        if values:
            return self._mean(values)
        return 0.0

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        """Return a safe mean that falls back to 0.0 for empty inputs."""
        if not values:
            return 0.0
        return float(sum(values) / len(values))
