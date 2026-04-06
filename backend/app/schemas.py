"""Pydantic request/response models for the lightweight demo API."""

from typing import List, Union

from pydantic import BaseModel, Field


class BandValues(BaseModel):
    """Bandpower values for one channel in one resting-state condition."""

    delta: float = Field(..., description="Bandpower in the 1-4 Hz delta range.")
    theta: float = Field(..., description="Bandpower in the 4-8 Hz theta range.")
    low_alpha: float = Field(..., description="Bandpower in the 8-10 Hz low-alpha range.")
    high_alpha: float = Field(..., description="Bandpower in the 10-12 Hz high-alpha range.")
    alpha: float = Field(..., description="Bandpower in the 8-12 Hz alpha range.")
    low_beta: float = Field(..., description="Bandpower in the 12-20 Hz low-beta range.")
    beta: float = Field(..., description="Bandpower in the 12-30 Hz beta range.")
    high_beta: float = Field(..., description="Bandpower in the 20-30 Hz high-beta range.")


class ConditionEEGValues(BaseModel):
    """Anchor-channel EEG values for one condition such as EO or EC."""

    cz: BandValues = Field(..., description="Bandpower values for the CZ anchor channel.")
    f3: BandValues = Field(..., description="Bandpower values for the F3 anchor channel.")
    f4: BandValues = Field(..., description="Bandpower values for the F4 anchor channel.")
    fz: BandValues = Field(..., description="Bandpower values for the FZ anchor channel.")
    o1: BandValues = Field(..., description="Bandpower values for the O1 anchor channel.")


class ManualPredictionRequest(BaseModel):
    """Manual prediction payload built around anchor EEG inputs."""

    age: float = Field(
        ...,
        gt=0,
        lt=130,
        description="Subject age used as a lightweight metadata feature.",
    )
    gender: Union[str, int, float] = Field(
        ...,
        description="Gender input accepted as text (for example male/female) or numeric coding.",
    )
    eo: ConditionEEGValues = Field(
        ...,
        description="Eyes-open anchor EEG values. These are expanded into the saved feature schema.",
    )
    ec: ConditionEEGValues = Field(
        ...,
        description="Eyes-closed anchor EEG values. These are expanded into the saved feature schema.",
    )


class PredictionResult(BaseModel):
    """Predicted values for the four saved benchmark targets."""

    working_memory: float = Field(..., description="Predicted working-memory score.")
    attention: float = Field(..., description="Predicted attention score.")
    executive_function: float = Field(..., description="Predicted executive-function score.")
    intelligence: float = Field(..., description="Predicted intelligence score.")


class PredictionResponse(BaseModel):
    predictions: PredictionResult = Field(..., description="Predicted targets for one subject.")


class CsvPredictionItem(BaseModel):
    row_index: int = Field(..., description="1-based CSV row number from the uploaded file.")
    predictions: PredictionResult = Field(..., description="Predicted targets for that CSV row.")


class CsvPredictionResponse(BaseModel):
    rows: int = Field(..., description="Number of parsed data rows in the uploaded CSV.")
    predictions: List[CsvPredictionItem] = Field(
        ...,
        description="Row-indexed prediction results for the uploaded CSV.",
    )


class PdfPredictionResponse(BaseModel):
    message: str = Field(..., description="High-level message about the current PDF workflow status.")
    experimental: bool = Field(
        ...,
        description="True when the route is acting as an exploratory or future extension rather than a full prediction path.",
    )
    filename: str = Field(..., description="Original uploaded PDF filename.")
    size_bytes: int = Field(..., description="Uploaded file size in bytes.")
    notes: str = Field(..., description="Extra guidance about the intended future-extension flow.")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Human-readable error message for the request.")
