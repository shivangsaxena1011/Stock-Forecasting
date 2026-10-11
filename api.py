"""FastAPI backend application for Stock Price Prediction with LSTM.
Provides RESTful endpoints for health checks, real market historical data, and model training.
"""
from __future__ import annotations

import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.stock_forecaster import (
    DEFAULT_TICKERS,
    fetch_history,
    train_and_forecast,
    validate_ticker,
)

app = FastAPI(
    title="Stock Price LSTM Forecasting API",
    description="Production-ready FastAPI backend for genuine stock time-series forecasting using LSTM.",
    version="1.0.0",
)

# CORS configuration
origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ForecastRequest(BaseModel):
    ticker: str = Field(..., example="TCS.NS", description="Yahoo Finance ticker symbol")
    period: str = Field("5y", example="5y", description="Historical period: 1y, 2y, 5y, 10y")
    lookback: int = Field(60, ge=10, le=120, description="Lookback window in trading sessions")
    train_fraction: float = Field(0.80, ge=0.5, lt=1.0, description="Chronological train split")
    epochs: int = Field(25, ge=1, le=100, description="Training epochs")


class MetricValues(BaseModel):
    MAE: float
    RMSE: float
    MAPE_percent: float


class ForecastResponse(BaseModel):
    ticker: str
    last_observed_date: str
    last_observed_close: float
    next_session_prediction: float
    engine_type: str
    metrics: MetricValues
    baseline_metrics: MetricValues
    dates: List[str]
    actual_close: List[float]
    predicted_close: List[float]
    baseline_close: List[float]


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint for container orchestrators and deployment platforms."""
    return {"status": "ok", "service": "stock-forecaster-api", "version": "1.0.0"}


@app.get("/tickers", tags=["Metadata"])
def get_tickers():
    """List recommended Indian NSE and US equity tickers."""
    return DEFAULT_TICKERS


@app.post("/predict", response_model=ForecastResponse, tags=["Forecasting"])
def predict_stock(request: ForecastRequest):
    """Fetch genuine market history and execute chronological LSTM forecasting."""
    try:
        clean_ticker = validate_ticker(request.ticker)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        history = fetch_history(clean_ticker, period=request.period)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to fetch market data: {e}")

    try:
        result = train_and_forecast(
            history,
            clean_ticker,
            lookback=request.lookback,
            train_fraction=request.train_fraction,
            epochs=request.epochs,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training error: {e}")

    return ForecastResponse(
        ticker=result.ticker,
        last_observed_date=str(result.next_session_input_date.date()),
        last_observed_close=float(history["Close"].iloc[-1]),
        next_session_prediction=round(result.next_session_prediction, 4),
        engine_type=result.engine_type,
        metrics=MetricValues(**result.metrics),
        baseline_metrics=MetricValues(**result.baseline_metrics),
        dates=[str(d.date()) for d in result.test_frame.index],
        actual_close=[round(float(v), 4) for v in result.actual],
        predicted_close=[round(float(v), 4) for v in result.predicted],
        baseline_close=[round(float(v), 4) for v in result.baseline],
    )
