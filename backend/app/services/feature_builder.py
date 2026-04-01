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
    """Build full model feature vectors from a small set of anchor EEG inputs."""

    def __init__(self, feature_columns: Sequence[str]) -> None:
        self.feature_columns = [col.lower() for col in feature_columns]

    def build_vector(self, anchors: AnchorValues, age: float, gender: float) -> list[float]:
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
                channel, band, condition = conditioned_parts
                vector.append(
                    self._estimate_conditioned_channel_value(anchors, channel, band, condition)
                )
                continue

            plain_parts = self._parse_plain_feature(feature_name)
            if plain_parts is not None:
                channel, band = plain_parts
                vector.append(self._estimate_plain_channel_value(anchors, channel, band))
                continue

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
        values = [self._condition_band_mean(anchors, condition, band) for condition in CONDITIONS]
        values = [value for value in values if value is not None]
        if values:
            return self._mean(values)
        return 0.0

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))
