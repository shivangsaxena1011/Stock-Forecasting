"""FastAPI backend application for Stock Price Prediction with LSTM.
Provides production-hardened RESTful endpoints for health checks, market tickers, and model training.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import List, Literal, Optional
from collections import deque

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.stock_forecaster import (
    DEFAULT_TICKERS,
    fetch_history,
    train_and_forecast,
    validate_ticker,
)

logger = logging.getLogger("stock_forecaster_api")
logging.basicConfig(level=logging.INFO)

# Configurable settings via environment variables
ALLOWED_PERIODS = ("1y", "2y", "5y", "10y")
DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8501",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8501",
    "http://127.0.0.1:8000",
]
env_origins = os.getenv("ALLOWED_ORIGINS", "")
if env_origins.strip():
    ALLOWED_ORIGINS = [o.strip() for o in env_origins.split(",") if o.strip()]
else:
    ALLOWED_ORIGINS = DEFAULT_ALLOWED_ORIGINS

# In-memory sliding-window rate limiter (requests per client IP per window)
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "20"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

# Concurrency limit for expensive /predict training calls
MAX_CONCURRENT_PREDICTIONS = int(os.getenv("MAX_CONCURRENT_PREDICTIONS", "2"))
prediction_semaphore = threading.Semaphore(MAX_CONCURRENT_PREDICTIONS)

# Thread-safe rate limiter state
_rate_limiter_lock = threading.Lock()
_client_requests: dict[str, deque[float]] = {}


def check_rate_limit(client_ip: str) -> bool:
    """Sliding-window in-memory rate limiter per IP address.
    
    For multi-instance distributed deployments, a shared store (e.g., Redis or Upstash)
    should replace this in-memory implementation.
    """
    now = time.time()
    with _rate_limiter_lock:
        if client_ip not in _client_requests:
            _client_requests[client_ip] = deque()
        timestamps = _client_requests[client_ip]
        # Evict timestamps older than the window
        while timestamps and timestamps[0] < now - RATE_LIMIT_WINDOW_SECONDS:
            timestamps.popleft()
        if len(timestamps) >= RATE_LIMIT_REQUESTS:
            return False
        timestamps.append(now)
        return True


app = FastAPI(
    title="Stock Price LSTM Forecasting API",
    description=(
        "Production-ready, hardened FastAPI backend for genuine stock time-series forecasting using LSTM. "
        "Configured with rate limiting, concurrency gating, and explicit CORS allowed origins."
    ),
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limiting_middleware(request: Request, call_next):
    # Only rate limit forecast predictions and public API mutations
    client_ip = request.client.host if request.client else "unknown"
    if request.url.path == "/predict" and not check_rate_limit(client_ip):
        logger.warning("Rate limit exceeded for client %s on /predict", client_ip)
        return Response(
            content='{"detail": "Rate limit exceeded. Please wait before submitting more forecast requests."}',
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            media_type="application/json",
        )
    return await call_next(request)


class ForecastRequest(BaseModel):
    ticker: str = Field(
        ...,
        json_schema_extra={"example": "TCS.NS"},
        description="Yahoo Finance ticker symbol",
    )
    period: Literal["1y", "2y", "5y", "10y"] = Field(
        "5y",
        json_schema_extra={"example": "5y"},
        description="Historical period allowlist: 1y, 2y, 5y, 10y",
    )
    lookback: int = Field(60, ge=10, le=120, description="Lookback window in trading sessions")
    train_fraction: float = Field(0.80, ge=0.5, lt=1.0, description="Chronological train split fraction")
    epochs: int = Field(25, ge=1, le=100, description="Training epochs (1 to 100)")


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
    return {"status": "ok", "service": "stock-forecaster-api", "version": "1.1.0"}


@app.get("/tickers", tags=["Metadata"])
def get_tickers():
    """List recommended Indian NSE and US equity tickers."""
    return DEFAULT_TICKERS


@app.post("/predict", response_model=ForecastResponse, tags=["Forecasting"])
def predict_stock(request: ForecastRequest):
    """Fetch genuine market history and execute chronological LSTM forecasting."""
    # 1. Validate ticker syntax
    try:
        clean_ticker = validate_ticker(request.ticker)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid ticker format: {val_err}"
        )

    # 2. Concurrency gating for resource-intensive model fitting
    acquired = prediction_semaphore.acquire(blocking=False)
    if not acquired:
        logger.warning("Predict request rejected: maximum concurrent training capacity reached.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The model training engine is currently at maximum capacity. Please retry shortly."
        )

    try:
        # 3. Ingest market history from provider
        try:
            history = fetch_history(clean_ticker, period=request.period)
        except RuntimeError as rt_err:
            logger.warning("Market data unavailable for ticker '%s': %s", clean_ticker, rt_err)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Unable to retrieve market data for '{clean_ticker}'. The symbol may be invalid, delisted, or the upstream provider is unavailable."
            )
        except Exception as prov_err:
            logger.error("Unexpected provider error fetching data for '%s': %s", clean_ticker, prov_err, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Upstream market data provider error. Please check network connectivity and retry."
            )

        # 4. Train model and forecast
        try:
            result = train_and_forecast(
                history,
                clean_ticker,
                lookback=request.lookback,
                train_fraction=request.train_fraction,
                epochs=request.epochs,
            )
        except ValueError as data_err:
            logger.warning("Data validation error during training for '%s': %s", clean_ticker, data_err)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Data validation error: {data_err}"
            )
        except Exception as train_err:
            logger.error("Model training exception for '%s': %s", clean_ticker, train_err, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An internal error occurred during model training and sequence evaluation."
            )

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
    finally:
        prediction_semaphore.release()
