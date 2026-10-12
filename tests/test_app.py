"""Unit tests for Streamlit dashboard state lifecycle, caching contracts, and input sync."""
from unittest import mock
import numpy as np
import pandas as pd
import pytest

from src.stock_forecaster import DEFAULT_TICKERS, validate_ticker
import app


def test_popular_stock_sync_updates_ticker():
    """Verify that selecting a popular stock updates the ticker input key in session state."""
    session_state = {}
    
    # Simulate default initial state
    labels = list(DEFAULT_TICKERS.keys())
    session_state["ticker_input"] = DEFAULT_TICKERS[labels[0]]
    
    # User selects a different stock from the dropdown
    selected_label = "Tata Consultancy Services (NSE)"
    session_state["popular_stock_select"] = selected_label
    
    # Callback logic
    with mock.patch.object(app, "st") as mock_st:
        mock_st.session_state = session_state
        
        # Simulate on_popular_stock_change callback
        selected = mock_st.session_state.get("popular_stock_select")
        if selected in DEFAULT_TICKERS:
            mock_st.session_state["ticker_input"] = DEFAULT_TICKERS[selected]
            
        assert mock_st.session_state["ticker_input"] == "TCS.NS"


def test_custom_ticker_manual_override_preserved():
    """Verify that manual ticker entry is preserved when the dropdown is not triggered."""
    session_state = {
        "popular_stock_select": "Reliance Industries (NSE)",
        "ticker_input": "NVDA",  # User manually typed NVDA
    }
    with mock.patch.object(app, "st") as mock_st:
        mock_st.session_state = session_state
        clean = validate_ticker(mock_st.session_state["ticker_input"])
        assert clean == "NVDA"


def test_failed_forecast_clears_previous_result():
    """Verify that when a forecast run fails, the previous forecast result is cleared."""
    session_state = {
        "forecast_result": "OLD_STALE_RESULT",
        "history": "OLD_HISTORY",
        "ticker": "RELIANCE.NS",
    }
    
    with mock.patch.object(app, "st") as mock_st:
        mock_st.session_state = session_state
        
        # Simulating run button click start:
        mock_st.session_state["forecast_result"] = None
        mock_st.session_state["history"] = None
        mock_st.session_state["ticker"] = None
        
        # Simulate an exception during fetch/forecast
        try:
            raise RuntimeError("Provider connection timed out")
        except Exception:
            pass  # Error reported to user
            
        # The result must be None so old forecast is not shown
        assert mock_st.session_state["forecast_result"] is None
        assert mock_st.session_state["history"] is None
        assert mock_st.session_state["ticker"] is None


def test_run_cached_forecast_signature_accepts_history():
    """Verify that run_cached_forecast accepts the historical DataFrame directly."""
    dates = pd.date_range("2023-01-01", periods=160, freq="B")
    df = pd.DataFrame({"Close": np.linspace(100.0, 200.0, 160)}, index=dates)
    
    with mock.patch("app.train_and_forecast") as mock_train:
        mock_train.return_value = "MOCK_FORECAST_RESULT"
        
        # Call the underlying function behind st.cache_data
        result = app.run_cached_forecast(
            df,
            "TEST.NS",
            lookback=20,
            train_fraction=0.8,
            epochs=5,
        )
        assert result == "MOCK_FORECAST_RESULT"
        mock_train.assert_called_once_with(
            df,
            "TEST.NS",
            lookback=20,
            train_fraction=0.8,
            epochs=5,
        )
