"""FastAPI application for the AI Plant Disease Detector backend."""

import io
import logging
import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exception_handlers import http_exception_handler as _default_http_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from app import __version__
from app.model_loader import ARCHITECTURE, DEFAULT_MODEL_PATH, get_model_service
from app.predictor import predict_image
from app.schemas import (
    ClassesResponse,
    ErrorResponse,
    HealthResponse,
    InfoResponse,
    PredictionResponse,
    TopPrediction,
)

logger = logging.getLogger("uvicorn.error")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}

service = get_model_service()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model once at startup (never retrain)."""
    service.load()
    yield


app = FastAPI(
    title="AI Plant Disease Detector API",
    version=__version__,
    description="Inference API for the unified plant ResNet50 checkpoint.",
    lifespan=lifespan,
)


def _cors_origins() -> List[str]:
    """Allowed origins come from FRONTEND_ORIGIN (comma separated).

    When unset, local development servers are allowed so the app can be
    tested on a workstation. Production deployments must set
    FRONTEND_ORIGIN to the deployed frontend URL.
    """
    raw = os.getenv("FRONTEND_ORIGIN", "").strip()
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return [
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.detail).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(error="Invalid request payload").model_dump(),
    )


@app.get("/", response_model=InfoResponse)
def root():
    return InfoResponse(
        name="AI Plant Disease Detector API",
        version=__version__,
        status="online",
        description=(
            "Inference API for the unified plant ResNet50 checkpoint. "
            "The model identifies the crop/plant type of a leaf image "
            f"across {service.num_classes or 'unknown'} classes."
        ),
        endpoints={
            "health": "/health",
            "classes": "/api/classes",
            "predict": "/api/predict",
        },
    )


@app.get("/health", response_model=HealthResponse)
def health():
    if service.loaded:
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            device=service.device,
            architecture=ARCHITECTURE,
            model_path=service.model_path,
            num_classes=service.num_classes,
        )
    # Degraded: the API is up but the model is unusable, so the
    # health check must fail (HTTP 503) for load balancers and
    # readiness probes.
    return JSONResponse(
        status_code=503,
        content=HealthResponse(
            status="degraded",
            model_loaded=False,
            device=service.device,
            architecture=ARCHITECTURE,
            model_path=service.model_path,
            num_classes=None,
            error=service.load_error,
        ).model_dump(),
    )


@app.get("/api/classes", response_model=ClassesResponse)
def list_classes():
    if not service.loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded, class list unavailable.",
        )
    return ClassesResponse(
        success=True,
        num_classes=service.num_classes,
        classes=service.class_names,
    )


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    if not service.loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded, prediction unavailable.",
        )

    # Reject oversized uploads from the Content-Length header when present.
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
                )
        except ValueError:
            pass

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )

    # Validate the actual image content, not the filename or MIME type.
    try:
        image = Image.open(io.BytesIO(content))
        image.verify()
        image = Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid, decodable image.",
        )

    if image.format not in ALLOWED_IMAGE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported image format '{image.format}'. "
                f"Allowed formats: {', '.join(sorted(ALLOWED_IMAGE_FORMATS))}."
            ),
        )

    try:
        image = image.convert("RGB")
        result = predict_image(service, image, top_k=3)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    return PredictionResponse(
        success=True,
        predicted_class=result.predicted_class,
        confidence=round(result.confidence, 6),
        top_predictions=[
            TopPrediction(class_name=name, confidence=round(conf, 6))
            for name, conf in result.top_predictions
        ],
    )
