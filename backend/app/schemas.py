"""Pydantic schemas for the API request/response models."""

from typing import List, Optional

from pydantic import BaseModel, Field


class TopPrediction(BaseModel):
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    success: bool
    predicted_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    top_predictions: List[TopPrediction]


class ClassesResponse(BaseModel):
    success: bool
    num_classes: int
    classes: List[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    architecture: str
    backend: Optional[str] = None
    model_path: str
    num_classes: Optional[int] = None
    error: Optional[str] = None


class InfoResponse(BaseModel):
    name: str
    version: str
    status: str
    description: str
    endpoints: dict


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
