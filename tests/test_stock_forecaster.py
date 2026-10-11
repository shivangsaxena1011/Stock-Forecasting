import unittest.mock as mock
import numpy as np
import pandas as pd
import pytest

from src.stock_forecaster import (
    RobustMinMaxScaler,
    make_sequences,
    chronological_split,
    validate_ticker,
    calculate_metrics,
    _metrics,
    PureNumPyLSTM,
    train_and_forecast,
    fetch_history,
    save_outputs,
)


# =====================================================================
# 1. Scaler & Normalization Unit Tests
# =====================================================================

def test_robust_min_max_scaler_fit_transform_inverse():
    scaler = RobustMinMaxScaler()
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    scaled = scaler.fit_transform(data)
    assert scaled[0] == 0.0
    assert scaled[-1] == 1.0
    assert scaled[2] == 0.5
    
    inversed = scaler.inverse_transform(scaled)
    np.testing.assert_allclose(inversed, data)


def test_robust_min_max_scaler_constant_series():
    scaler = RobustMinMaxScaler()
    data = np.array([5.0, 5.0, 5.0])
    scaled = scaler.fit_transform(data)
    assert not np.isnan(scaled).any()
    inversed = scaler.inverse_transform(scaled)
    np.testing.assert_allclose(inversed, data)


def test_robust_min_max_scaler_no_test_leakage():
    scaler = RobustMinMaxScaler()
    train_data = np.array([100.0, 200.0])
    test_data = np.array([50.0, 300.0])
    
    scaler.fit(train_data)
    assert scaler.data_min_ == 100.0
    assert scaler.data_max_ == 200.0
    
    test_scaled = scaler.transform(test_data)
    assert test_scaled[0] == -0.5
    assert test_scaled[1] == 2.0
    assert scaler.data_min_ == 100.0


# =====================================================================
# 2. Sequence Generation & Leak-Prevention Tests
# =====================================================================

def test_make_sequences_shape_and_targets():
    values = np.arange(10, dtype=float)
    X, y = make_sequences(values, lookback=3)
    assert X.shape == (7, 3, 1)
    assert y.shape == (7,)
    np.testing.assert_array_equal(X[0, :, 0], [0, 1, 2])
    assert y[0] == 3
    assert y[-1] == 9


def test_make_sequences_target_alignment_prevents_leakage():
    values = np.arange(100, dtype=float)
    lookback = 10
    X, y = make_sequences(values, lookback=lookback)
    for i in range(len(y)):
        assert y[i] not in X[i, :, 0]
        assert y[i] == X[i, -1, 0] + 1.0


def test_make_sequences_rejects_short_data():
    with pytest.raises(ValueError):
        make_sequences([1, 2, 3], lookback=3)


def test_make_sequences_lookback_boundary():
    with pytest.raises(ValueError):
        make_sequences([1, 2, 3], lookback=0)


def test_chronological_split_keeps_order():
    frame = pd.DataFrame({"Close": np.arange(100, dtype=float)})
    train, test = chronological_split(frame, 0.8)
    assert len(train) == 80
    assert len(test) == 20
    assert train.iloc[-1, 0] < test.iloc[0, 0]


def test_chronological_split_invalid_fraction():
    frame = pd.DataFrame({"Close": np.arange(100, dtype=float)})
    with pytest.raises(ValueError):
        chronological_split(frame, 0.3)
    with pytest.raises(ValueError):
        chronological_split(frame, 1.0)


# =====================================================================
# 3. Validation & Metrics Tests
# =====================================================================

def test_validate_ticker():
    assert validate_ticker(" tcs.ns ") == "TCS.NS"
    assert validate_ticker("aapl") == "AAPL"
    with pytest.raises(ValueError):
        validate_ticker("bad ticker with spaces")
    with pytest.raises(ValueError):
        validate_ticker("")


def test_metrics_perfect_prediction():
    result = calculate_metrics(np.array([10, 20, 30]), np.array([10, 20, 30]))
    assert result["MAE"] == 0.0
    assert result["RMSE"] == 0.0
    assert result["MAPE_percent"] == 0.0


def test_metrics_handles_zero_values():
    result = calculate_metrics(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    assert result["MAE"] == 1.0
    assert result["RMSE"] == 1.0
    assert np.isnan(result["MAPE_percent"])


def test_metrics_handles_empty_arrays():
    result = calculate_metrics(np.array([]), np.array([]))
    assert np.isnan(result["MAE"])
    assert np.isnan(result["RMSE"])
    assert np.isnan(result["MAPE_percent"])


def test_backward_compatibility_alias():
    result = _metrics(np.array([10, 20]), np.array([10, 20]))
    assert result["MAE"] == 0.0


# =====================================================================
# 4. PureNumPyLSTM Mathematical Tests
# =====================================================================

def test_pure_numpy_lstm_numerical_gradient_check():
    input_dim = 1
    hidden_dim = 2
    dense_dim = 2
    seq_len = 3
    batch_size = 2
    
    np.random.seed(42)
    X = np.random.randn(batch_size, seq_len, input_dim).astype(np.float64)
    y = np.random.randn(batch_size, 1).astype(np.float64)
    
    model = PureNumPyLSTM(input_dim=input_dim, hidden_dim=hidden_dim, dense_dim=dense_dim, lr=0.01, seed=42)
    out, cache = model.forward(X)
    d_out = 2.0 * (out - y)
    dW, db, dW_d1, db_d1, dW_d2, db_d2 = model.backward(d_out, cache)
    
    def loss_fn():
        o, _ = model.forward(X)
        return np.mean((o - y) ** 2)
        
    eps = 1e-6
    for i in range(model.W_d2.shape[0]):
        for j in range(model.W_d2.shape[1]):
            orig = model.W_d2[i, j]
            model.W_d2[i, j] = orig + eps
            l_plus = loss_fn()
            model.W_d2[i, j] = orig - eps
            l_minus = loss_fn()
            model.W_d2[i, j] = orig
            num_grad = (l_plus - l_minus) / (2 * eps)
            np.testing.assert_allclose(dW_d2[i, j], num_grad, rtol=1e-4, atol=1e-6)
            
    for i in range(model.W.shape[0]):
        for j in range(model.W.shape[1]):
            orig = model.W[i, j]
            model.W[i, j] = orig + eps
            l_plus = loss_fn()
            model.W[i, j] = orig - eps
            l_minus = loss_fn()
            model.W[i, j] = orig
            num_grad = (l_plus - l_minus) / (2 * eps)
            np.testing.assert_allclose(dW[i, j], num_grad, rtol=1e-3, atol=1e-5)


def test_pure_numpy_lstm_learning_convergence():
    t = np.linspace(0, 10, 120)
    series = np.sin(t) + 3.0
    X, y = make_sequences(series, lookback=10)
    
    model = PureNumPyLSTM(input_dim=1, hidden_dim=16, dense_dim=8, lr=0.01, seed=42)
    initial_preds = model.predict(X[:80])
    initial_loss = np.mean((initial_preds.flatten() - y[:80]) ** 2)
    
    model.fit(X[:80], y[:80], X[80:], y[80:], epochs=15, batch_size=16)
    final_preds = model.predict(X[:80])
    final_loss = np.mean((final_preds.flatten() - y[:80]) ** 2)
    
    assert final_loss < initial_loss * 0.1
    assert not np.isnan(final_preds).any()


def test_pure_numpy_lstm_best_weights_restoration():
    t = np.linspace(0, 5, 80)
    series = np.sin(t) + 2.0
    X, y = make_sequences(series, lookback=8)
    
    model = PureNumPyLSTM(input_dim=1, hidden_dim=8, dense_dim=4, lr=0.01, seed=42)
    model.fit(X[:50], y[:50], X_val=X[50:], y_val=y[50:], epochs=25, batch_size=8, patience=2)
    val_preds = model.predict(X[50:])
    val_mse = np.mean((val_preds.flatten() - y[50:]) ** 2)
    assert not np.isnan(val_mse)
    assert val_mse < 1.0


# =====================================================================
# 5. Pipeline Smoke & CSV Export Tests
# =====================================================================

def test_train_and_forecast_synthetic_pipeline_smoke(tmp_path):
    dates = pd.date_range("2022-01-01", periods=160, freq="D")
    df = pd.DataFrame({"Close": np.linspace(100.0, 200.0, 160)}, index=dates)
    
    res = train_and_forecast(df, "TEST", lookback=15, epochs=3, force_numpy_engine=True)
    assert res.ticker == "TEST"
    assert len(res.actual) == len(res.predicted)
    assert len(res.actual) == len(res.baseline)
    assert res.next_session_prediction > 0
    assert "MAE" in res.metrics
    assert "RMSE" in res.metrics
    assert "MAPE_percent" in res.metrics
    assert "MAE" in res.baseline_metrics
    assert res.engine_type in ["pure_numpy", "tensorflow"]
    
    # Test CSV export
    out_file = save_outputs(res, output_dir=tmp_path)
    assert out_file.exists()
    saved_df = pd.read_csv(out_file)
    assert "Date" in saved_df.columns
    assert "Actual_Close" in saved_df.columns
    assert "LSTM_Predicted_Close" in saved_df.columns
    assert "Persistence_Baseline" in saved_df.columns
    assert len(saved_df) == len(res.actual)


# =====================================================================
# 6. Mocked Market Data Offline Tests (Zero Internet Required)
# =====================================================================

def test_fetch_history_mock_empty_raises_runtime_error():
    with mock.patch("src.stock_forecaster._download_market_data", return_value=pd.DataFrame()):
        with pytest.raises(RuntimeError, match="No market data was returned"):
            fetch_history("TCS.NS")


def test_fetch_history_mock_missing_close_raises_runtime_error():
    mock_df = pd.DataFrame({"Open": [100.0] * 120}, index=pd.date_range("2020-01-01", periods=120))
    with mock.patch("src.stock_forecaster._download_market_data", return_value=mock_df):
        with pytest.raises(RuntimeError, match="does not contain a 'Close' column"):
            fetch_history("TCS.NS")


def test_fetch_history_mock_insufficient_rows_raises_runtime_error():
    mock_df = pd.DataFrame({"Close": [100.0] * 50}, index=pd.date_range("2020-01-01", periods=50))
    with mock.patch("src.stock_forecaster._download_market_data", return_value=mock_df):
        with pytest.raises(RuntimeError, match="Only 50 usable trading observations"):
            fetch_history("TCS.NS")


def test_fetch_history_mock_multiindex_resolution():
    dates = pd.date_range("2020-01-01", periods=150)
    tuples = [("Close", "AAPL"), ("Open", "AAPL"), ("High", "AAPL"), ("Low", "AAPL"), ("Volume", "AAPL")]
    cols = pd.MultiIndex.from_tuples(tuples, names=["Price", "Ticker"])
    data = np.ones((150, 5)) * 150.0
    mock_df = pd.DataFrame(data, index=dates, columns=cols)
    
    with mock.patch("src.stock_forecaster._download_market_data", return_value=mock_df):
        res = fetch_history("AAPL")
        assert "Close" in res.columns
        assert len(res) == 150
        assert res["Close"].iloc[0] == 150.0


def test_fetch_history_missing_yfinance_raises_actionable_import_error():
    with mock.patch.dict("sys.modules", {"yfinance": None}):
        with pytest.raises(ImportError, match="The 'yfinance' library is required"):
            fetch_history("AAPL")


# =====================================================================
# 7. Live Market Data Integration Tests (Opt-in marker)
# =====================================================================

@pytest.mark.integration
def test_fetch_history_live_smoke():
    try:
        import yfinance
    except ImportError:
        pytest.skip("yfinance is not installed in this environment")
    df = fetch_history("AAPL", period="1y")
    assert len(df) >= 100
    assert "Close" in df.columns
