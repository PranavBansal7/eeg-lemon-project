# Run locally:
# uvicorn app.main:app --reload

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import ANCHOR_CHANNELS, BANDS, CONDITIONS
from app.schemas import (
    CsvPredictionItem,
    CsvPredictionResponse,
    ErrorResponse,
    ManualPredictionRequest,
    PdfPredictionResponse,
    PredictionResponse,
    PredictionResult,
)
from app.services.parsers import extract_anchor_payload_from_row, parse_csv_upload
from app.services.predictor import PredictionService

logger = logging.getLogger(__name__)


def build_demo_anchor_values() -> dict[str, dict[str, dict[str, float]]]:
    base_band_values = {
        "delta": 2.40,
        "theta": 2.10,
        "low_alpha": 1.85,
        "high_alpha": 1.75,
        "alpha": 1.80,
        "low_beta": 1.45,
        "beta": 1.35,
        "high_beta": 1.20,
    }

    channel_offsets = {
        "cz": 0.00,
        "f3": 0.08,
        "f4": 0.06,
        "fz": 0.04,
        "o1": -0.03,
    }

    condition_offsets = {"eo": 0.05, "ec": 0.00}

    anchors: dict[str, dict[str, dict[str, float]]] = {}
    for condition in CONDITIONS:
        anchors[condition] = {}
        for channel in ANCHOR_CHANNELS:
            anchors[condition][channel] = {}
            for band in BANDS:
                anchors[condition][channel][band] = round(
                    base_band_values[band]
                    + channel_offsets[channel]
                    + condition_offsets[condition],
                    4,
                )

    return anchors


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor = PredictionService()
    try:
        predictor.load()
        app.state.prediction_service = predictor
        logger.info("Prediction artifacts loaded successfully.")
    except Exception:
        logger.exception("Failed to load prediction artifacts during startup.")
        raise

    yield


app = FastAPI(
    title="EEG Cognition Prediction API",
    description=(
        "FastAPI backend for predicting working memory, attention, "
        "executive function, and intelligence from EEG anchor features."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_prediction_service() -> PredictionService:
    service = getattr(app.state, "prediction_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction service is not initialized yet.",
        )
    return service


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    if isinstance(exc.detail, str):
        detail = exc.detail
    else:
        detail = "Request failed."
    return JSONResponse(status_code=exc.status_code, content={"detail": detail})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Invalid request payload.",
            "errors": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc: Exception):
    logger.exception("Unhandled server error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Unexpected server error. Check input format and server logs."
        },
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "name": "EEG Cognition Prediction API",
        "status": "ok",
        "health": "/health",
        "docs": "/docs",
    }


@app.post(
    "/predict-manual",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def predict_manual(
    payload: ManualPredictionRequest,
    predictor: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    try:
        prediction = predictor.predict_from_manual(payload)
        return PredictionResponse(predictions=PredictionResult(**prediction))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(
    "/predict-from-csv",
    response_model=CsvPredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def predict_from_csv(
    file: UploadFile = File(...),
    predictor: PredictionService = Depends(get_prediction_service),
) -> CsvPredictionResponse:
    filename = (file.filename or "").lower()
    if not filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    try:
        rows = await parse_csv_upload(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    records = []
    for row_index, row in enumerate(rows, start=1):
        try:
            age, gender, anchors = extract_anchor_payload_from_row(row, row_index)
            records.append((age, gender, anchors))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        predictions = predictor.predict_batch(records)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    prediction_rows = [
        CsvPredictionItem(
            row_index=index,
            predictions=PredictionResult(**prediction),
        )
        for index, prediction in enumerate(predictions, start=1)
    ]

    return CsvPredictionResponse(rows=len(prediction_rows), predictions=prediction_rows)


@app.post(
    "/predict-from-pdf",
    response_model=PdfPredictionResponse,
    responses={400: {"model": ErrorResponse}},
)
async def predict_from_pdf(file: UploadFile = File(...)) -> PdfPredictionResponse:
    filename = file.filename or "uploaded.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded PDF file is empty.")

    # Placeholder for a future PDF-to-features extractor pipeline.
    return PdfPredictionResponse(
        message=(
            "PDF approximation is experimental. No EEG prediction is generated from "
            "PDF input yet."
        ),
        experimental=True,
        filename=filename,
        size_bytes=len(content),
        notes=(
            "Future extension: extract anchor EEG values from report text/tables, "
            "then reuse the same prediction pipeline as manual/CSV routes."
        ),
    )


@app.get(
    "/predict-demo",
    response_model=PredictionResponse,
    responses={500: {"model": ErrorResponse}},
)
async def predict_demo(
    predictor: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    demo_anchors = build_demo_anchor_values()
    prediction = predictor.predict_from_anchors(
        age=29,
        gender="male",
        anchors=demo_anchors,
    )
    return PredictionResponse(predictions=PredictionResult(**prediction))
