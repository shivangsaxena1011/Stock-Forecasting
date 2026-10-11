"""Unit and integration tests for the FastAPI backend (api.py)."""
from unittest import mock
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import app, check_rate_limit, _client_requests


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "stock-forecaster-api"


def test_tickers_endpoint(client):
    response = client.get("/tickers")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "Tata Consultancy Services (NSE)" in data
    assert "Apple (NASDAQ)" in data


def test_predict_invalid_ticker_format(client):
    response = client.post(
        "/predict",
        json={"ticker": "INVALID TICKER WITH SPACES", "period": "5y", "lookback": 60, "train_fraction": 0.8, "epochs": 5},
    )
    assert response.status_code == 400
    assert "Invalid ticker" in response.json()["detail"]


def test_predict_invalid_period(client):
    response = client.post(
        "/predict",
        json={"ticker": "TCS.NS", "period": "3y", "lookback": 60, "train_fraction": 0.8, "epochs": 5},
    )
    # Pydantic Literal validation produces 422 Unprocessable Entity
    assert response.status_code == 422


def test_predict_provider_failure(client):
    with mock.patch("api.fetch_history", side_effect=RuntimeError("Provider connection reset")):
        response = client.post(
            "/predict",
            json={"ticker": "TCS.NS", "period": "5y", "lookback": 60, "train_fraction": 0.8, "epochs": 5},
        )
        assert response.status_code == 502
        assert "Unable to retrieve market data" in response.json()["detail"]


def test_predict_model_training_failure(client):
    dummy_history = pd.DataFrame(
        {"Close": np.linspace(100, 200, 200)},
        index=pd.date_range("2023-01-01", periods=200, freq="B"),
    )
    with mock.patch("api.fetch_history", return_value=dummy_history):
        with mock.patch("api.train_and_forecast", side_effect=Exception("Internal CUDA crash")):
            response = client.post(
                "/predict",
                json={"ticker": "TCS.NS", "period": "5y", "lookback": 60, "train_fraction": 0.8, "epochs": 5},
            )
            assert response.status_code == 500
            assert "An internal error occurred" in response.json()["detail"]
            # Sensitive internal traceback must NOT be in the response
            assert "CUDA" not in response.json()["detail"]


def test_predict_successful_forecast(client):
    from src.stock_forecaster import ForecastResult, RobustMinMaxScaler
    
    dummy_history = pd.DataFrame(
        {"Close": np.linspace(100, 200, 200)},
        index=pd.date_range("2023-01-01", periods=200, freq="B"),
    )
    dummy_result = ForecastResult(
        ticker="TCS.NS",
        history=dummy_history,
        test_frame=dummy_history.iloc[160:],
        actual=np.linspace(180, 200, 40),
        predicted=np.linspace(179, 199, 40),
        baseline=np.linspace(179.5, 199.5, 40),
        next_session_prediction=201.5,
        next_session_input_date=dummy_history.index[-1],
        metrics={"MAE": 1.0, "RMSE": 1.2, "MAPE_percent": 0.5},
        baseline_metrics={"MAE": 0.5, "RMSE": 0.7, "MAPE_percent": 0.25},
        model=None,
        scaler=RobustMinMaxScaler(),
        lookback=20,
        engine_type="pure_numpy",
    )
    with mock.patch("api.fetch_history", return_value=dummy_history):
        with mock.patch("api.train_and_forecast", return_value=dummy_result):
            response = client.post(
                "/predict",
                json={"ticker": "TCS.NS", "period": "5y", "lookback": 20, "train_fraction": 0.8, "epochs": 2},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["ticker"] == "TCS.NS"
            assert "next_session_prediction" in data
            assert data["next_session_prediction"] == 201.5
            assert "metrics" in data
            assert data["metrics"]["MAE"] == 1.0
            assert "baseline_metrics" in data
            assert len(data["actual_close"]) == 40


def test_rate_limiter_unit():
    ip = "192.168.1.100"
    _client_requests.pop(ip, None)
    # Request up to the limit
    for _ in range(20):
        assert check_rate_limit(ip) is True
    # 21st request should be rejected
    assert check_rate_limit(ip) is False
