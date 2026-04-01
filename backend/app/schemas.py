from typing import List, Union

from pydantic import BaseModel, Field


class BandValues(BaseModel):
    delta: float
    theta: float
    low_alpha: float
    high_alpha: float
    alpha: float
    low_beta: float
    beta: float
    high_beta: float


class ConditionEEGValues(BaseModel):
    cz: BandValues
    f3: BandValues
    f4: BandValues
    fz: BandValues
    o1: BandValues


class ManualPredictionRequest(BaseModel):
    age: float = Field(..., gt=0, lt=130)
    gender: Union[str, int, float]
    eo: ConditionEEGValues
    ec: ConditionEEGValues


class PredictionResult(BaseModel):
    working_memory: float
    attention: float
    executive_function: float
    intelligence: float


class PredictionResponse(BaseModel):
    predictions: PredictionResult


class CsvPredictionItem(BaseModel):
    row_index: int
    predictions: PredictionResult


class CsvPredictionResponse(BaseModel):
    rows: int
    predictions: List[CsvPredictionItem]


class PdfPredictionResponse(BaseModel):
    message: str
    experimental: bool
    filename: str
    size_bytes: int
    notes: str


class ErrorResponse(BaseModel):
    detail: str
