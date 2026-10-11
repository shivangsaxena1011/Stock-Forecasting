"""Utilities for stock closing price forecasting with LSTM neural networks.

Data source: Yahoo Finance via yfinance. Real historical data only; no mock/synthetic market prices.
Designed for educational and research evaluation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple, Dict, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_TICKERS = {
    "Reliance Industries (NSE)": "RELIANCE.NS",
    "Tata Consultancy Services (NSE)": "TCS.NS",
    "Infosys (NSE)": "INFY.NS",
    "HDFC Bank (NSE)": "HDFCBANK.NS",
    "State Bank of India (NSE)": "SBIN.NS",
    "Apple (NASDAQ)": "AAPL",
    "Microsoft (NASDAQ)": "MSFT",
    "NVIDIA (NASDAQ)": "NVDA",
    "Amazon (NASDAQ)": "AMZN",
    "Tesla (NASDAQ)": "TSLA",
}


class RobustMinMaxScaler:
    """Min-Max scaler that operates cleanly on 1D/2D arrays.
    
    Equivalent to sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1)),
    implemented with pure NumPy so it has zero external native dependency risks.
    """
    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        self.feature_range = feature_range
        self.data_min_: Optional[float] = None
        self.data_max_: Optional[float] = None
        self.scale_: float = 1.0
        self.min_: float = 0.0

    def fit(self, X: np.ndarray) -> "RobustMinMaxScaler":
        X = np.asarray(X, dtype=np.float64)
        self.data_min_ = float(np.nanmin(X))
        self.data_max_ = float(np.nanmax(X))
        data_range = self.data_max_ - self.data_min_
        if data_range <= 1e-12:
            self.scale_ = 1.0
        else:
            self.scale_ = (self.feature_range[1] - self.feature_range[0]) / data_range
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X * self.scale_ + self.min_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if abs(self.scale_) < 1e-12:
            return X
        return (X - self.min_) / self.scale_


def validate_ticker(ticker: str) -> str:
    """Validate and normalize ticker symbols."""
    ticker = str(ticker).strip().upper()
    if not ticker or len(ticker) > 20 or any(c.isspace() for c in ticker):
        raise ValueError("Enter a valid Yahoo Finance ticker, e.g. TCS.NS or AAPL.")
    return ticker


def _download_market_data(
    ticker: str,
    period: str = "5y",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Internal market data downloader.
    
    Uses yfinance to fetch genuine market data. If yfinance is not installed in the environment,
    raises a clear, actionable ImportError rather than a silent failure.
    """
    try:
        import yfinance as yf
    except ImportError as err:
        raise ImportError(
            "The 'yfinance' library is required to download market data. "
            "Please install dependencies with: pip install -r requirements.txt (or pip install yfinance)."
        ) from err

    return yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=auto_adjust,
        progress=False,
        threads=False,
    )


def fetch_history(
    ticker: str,
    period: str = "5y",
    price_col: str = "Close",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Fetch genuine historical daily market data via yfinance.
    
    Handles:
    - MultiIndex columns returned by modern yfinance versions.
    - Indian (.NS) and US ticker conventions.
    - Corporate actions & stock splits (via auto_adjust=True default).
    - Rate limits, delisted symbols, network errors, and empty responses.
    - Removal of duplicate indices and proper UTC-naive DatetimeIndex sorting.
    
    Raises RuntimeError on download failure. Never substitutes fake or mock data.
    """
    ticker = validate_ticker(ticker)
    try:
        frame = _download_market_data(ticker, period=period, auto_adjust=auto_adjust)
    except ImportError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Could not download market data for '{ticker}'. Check your internet connection "
            f"and ticker symbol, then retry. Detailed error: {exc}"
        ) from exc

    if frame is None or frame.empty:
        raise RuntimeError(
            f"No market data was returned for ticker '{ticker}'. The symbol may be invalid, "
            "delisted, or temporarily unavailable from the market data provider. No mock data is substituted."
        )

    # Resolve MultiIndex columns if present
    if isinstance(frame.columns, pd.MultiIndex):
        try:
            if "Ticker" in frame.columns.names and ticker in frame.columns.get_level_values("Ticker"):
                frame = frame.xs(ticker, axis=1, level="Ticker").copy()
            else:
                frame.columns = frame.columns.get_level_values(0)
        except Exception:
            frame.columns = frame.columns.get_level_values(0)

    # Standardize column names to Title Case
    frame.columns = [str(c).title() for c in frame.columns]
    frame = frame.loc[:, ~frame.columns.duplicated()].copy()

    target_price_col = price_col.title()
    if target_price_col not in frame.columns:
        if "Close" in frame.columns:
            target_price_col = "Close"
        else:
            raise RuntimeError(
                f"Downloaded data for '{ticker}' does not contain a '{price_col}' column. "
                f"Available columns: {list(frame.columns)}"
            )

    # If the user requested Close and auto_adjust was False, Close and Adj Close exist
    if target_price_col != "Close" and target_price_col in frame.columns:
        frame["Close"] = frame[target_price_col]

    frame.index = pd.to_datetime(frame.index)
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    frame.index.name = "Date"
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame.dropna(subset=["Close"])

    if len(frame) < 100:
        raise RuntimeError(
            f"Only {len(frame)} usable trading observations were downloaded for '{ticker}'. "
            "Choose a longer historical period or another ticker (at least 100 rows are required for modeling)."
        )

    return frame


def make_sequences(values: np.ndarray, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a 1-D or (n,1) series into sliding samples shaped (samples, lookback, 1).
    
    Each target y[i] strictly succeeds the corresponding lookback sequence X[i].
    Strictly prevents future observation leakage.
    """
    values = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if len(values) <= lookback:
        raise ValueError(
            f"Need more than lookback={lookback} observations; got {len(values)}."
        )
    xs, ys = [], []
    for end in range(lookback, len(values)):
        xs.append(values[end - lookback:end])
        ys.append(values[end, 0])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split time-series data chronologically. Never shuffle sequential observations."""
    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be >= 0.5 and < 1.0")
    if len(frame) < 100:
        raise ValueError("At least 100 rows are required for a meaningful split.")
    cut = int(len(frame) * train_fraction)
    return frame.iloc[:cut].copy(), frame.iloc[cut:].copy()


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Calculate MAE, RMSE, and MAPE. Handles edge cases and zero denominators gracefully."""
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if len(actual) == 0 or len(predicted) == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "MAPE_percent": float("nan")}
    
    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    
    nonzero = np.abs(actual) > 1e-12
    if nonzero.any():
        mape = float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100.0)
    else:
        mape = float("nan")
        
    return {"MAE": mae, "RMSE": rmse, "MAPE_percent": mape}


# Alias for backward compatibility
_metrics = calculate_metrics


@dataclass
class ForecastResult:
    ticker: str
    history: pd.DataFrame
    test_frame: pd.DataFrame
    actual: np.ndarray
    predicted: np.ndarray
    baseline: np.ndarray
    next_session_prediction: float
    next_session_input_date: pd.Timestamp
    metrics: Dict[str, float]
    baseline_metrics: Dict[str, float]
    model: Any
    scaler: RobustMinMaxScaler
    lookback: int
    engine_type: str = "tensorflow"


class PureNumPyLSTM:
    """Full-featured, mathematically rigorous LSTM sequence forecaster in pure NumPy.
    
    Implements:
    - Forward pass through LSTM cell: input gate (i), forget gate (f), cell state (c), output gate (o).
    - Forget gate bias initialized to +1.0 for time-series stability.
    - Hidden state projection through Dense(ReLU) and Dense(Linear).
    - Backpropagation through time (BPTT) with analytical gradients.
    - Adam optimizer with first and second moment corrections.
    - Early stopping with best weight restoration based on validation loss.
    - Gradient clipping for numerical stability.
    
    Used when TensorFlow cannot run due to platform policy constraints (such as Windows Smart App Control).
    """
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 32,
        dense_dim: int = 16,
        lr: float = 0.005,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dense_dim = dense_dim
        self.lr = lr
        self.rng = np.random.RandomState(seed)
        
        # Xavier/Glorot initialization for LSTM
        limit = np.sqrt(6.0 / (input_dim + hidden_dim + 4 * hidden_dim))
        self.W = self.rng.uniform(-limit, limit, (input_dim + hidden_dim, 4 * hidden_dim)).astype(np.float64)
        self.b = np.zeros(4 * hidden_dim, dtype=np.float64)
        # Forget gate bias init to 1.0 for better initial gradient flow
        self.b[hidden_dim:2 * hidden_dim] = 1.0
        
        # Dense 1
        limit_d1 = np.sqrt(6.0 / (hidden_dim + dense_dim))
        self.W_d1 = self.rng.uniform(-limit_d1, limit_d1, (hidden_dim, dense_dim)).astype(np.float64)
        self.b_d1 = np.zeros(dense_dim, dtype=np.float64)
        
        # Dense 2
        limit_d2 = np.sqrt(6.0 / (dense_dim + 1))
        self.W_d2 = self.rng.uniform(-limit_d2, limit_d2, (dense_dim, 1)).astype(np.float64)
        self.b_d2 = np.zeros(1, dtype=np.float64)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, tuple]:
        batch_size, seq_len, _ = X.shape
        H = self.hidden_dim
        
        hs = np.zeros((batch_size, seq_len + 1, H), dtype=np.float64)
        cs = np.zeros((batch_size, seq_len + 1, H), dtype=np.float64)
        gates_all = np.zeros((batch_size, seq_len, 4 * H), dtype=np.float64)
        
        for t in range(seq_len):
            x_t = X[:, t, :]
            h_prev = hs[:, t, :]
            c_prev = cs[:, t, :]
            
            concat = np.hstack([x_t, h_prev])
            gates = concat @ self.W + self.b
            gates_all[:, t, :] = gates
            
            i = self._sigmoid(gates[:, :H])
            f = self._sigmoid(gates[:, H:2 * H])
            c_bar = np.tanh(gates[:, 2 * H:3 * H])
            o = self._sigmoid(gates[:, 3 * H:])
            
            c_next = f * c_prev + i * c_bar
            h_next = o * np.tanh(c_next)
            
            cs[:, t + 1, :] = c_next
            hs[:, t + 1, :] = h_next
            
        h_final = hs[:, -1, :]
        z1 = h_final @ self.W_d1 + self.b_d1
        a1 = np.maximum(0.0, z1)
        out = a1 @ self.W_d2 + self.b_d2
        
        cache = (X, hs, cs, gates_all, z1, a1)
        return out, cache

    def predict(self, X: np.ndarray, verbose: int = 0) -> np.ndarray:
        out, _ = self.forward(X)
        return out

    def backward(self, d_out: np.ndarray, cache: tuple) -> Tuple[np.ndarray, ...]:
        X, hs, cs, gates_all, z1, a1 = cache
        batch_size, seq_len, _ = X.shape
        H = self.hidden_dim
        
        dW_d2 = a1.T @ d_out / batch_size
        db_d2 = np.sum(d_out, axis=0) / batch_size
        da1 = d_out @ self.W_d2.T
        
        dz1 = da1 * (z1 > 0)
        dW_d1 = hs[:, -1, :].T @ dz1 / batch_size
        db_d1 = np.sum(dz1, axis=0) / batch_size
        dh_final = dz1 @ self.W_d1.T
        
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dh_next = dh_final
        dc_next = np.zeros((batch_size, H), dtype=np.float64)
        
        for t in reversed(range(seq_len)):
            x_t = X[:, t, :]
            h_prev = hs[:, t, :]
            c_prev = cs[:, t, :]
            c_t = cs[:, t + 1, :]
            gates = gates_all[:, t, :]
            
            i = self._sigmoid(gates[:, :H])
            f = self._sigmoid(gates[:, H:2 * H])
            c_bar = np.tanh(gates[:, 2 * H:3 * H])
            o = self._sigmoid(gates[:, 3 * H:])
            
            tanh_c_t = np.tanh(c_t)
            do = dh_next * tanh_c_t
            dc = dc_next + dh_next * o * (1.0 - tanh_c_t ** 2)
            
            df = dc * c_prev
            di = dc * c_bar
            dc_bar = dc * i
            
            dg_i = di * i * (1.0 - i)
            dg_f = df * f * (1.0 - f)
            dg_c_bar = dc_bar * (1.0 - c_bar ** 2)
            dg_o = do * o * (1.0 - o)
            
            dg = np.hstack([dg_i, dg_f, dg_c_bar, dg_o])
            concat = np.hstack([x_t, h_prev])
            
            dW += concat.T @ dg / batch_size
            db += np.sum(dg, axis=0) / batch_size
            
            dconcat = dg @ self.W.T
            dh_next = dconcat[:, self.input_dim:]
            dc_next = f * dc
            
        np.clip(dW, -1.0, 1.0, out=dW)
        np.clip(db, -1.0, 1.0, out=db)
        np.clip(dW_d1, -1.0, 1.0, out=dW_d1)
        np.clip(db_d1, -1.0, 1.0, out=db_d1)
        np.clip(dW_d2, -1.0, 1.0, out=dW_d2)
        np.clip(db_d2, -1.0, 1.0, out=db_d2)
        
        return dW, db, dW_d1, db_d1, dW_d2, db_d2

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 25,
        batch_size: int = 32,
        patience: int = 5,
        lr: Optional[float] = None,
    ) -> "PureNumPyLSTM":
        if lr is not None:
            self.lr = lr
        n_samples = len(X_train)
        best_val_loss = float("inf")
        patience_counter = 0
        best_weights = None
        
        m_W, v_W = np.zeros_like(self.W), np.zeros_like(self.W)
        m_b, v_b = np.zeros_like(self.b), np.zeros_like(self.b)
        m_W1, v_W1 = np.zeros_like(self.W_d1), np.zeros_like(self.W_d1)
        m_b1, v_b1 = np.zeros_like(self.b_d1), np.zeros_like(self.b_d1)
        m_W2, v_W2 = np.zeros_like(self.W_d2), np.zeros_like(self.W_d2)
        m_b2, v_b2 = np.zeros_like(self.b_d2), np.zeros_like(self.b_d2)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t_step = 0
        
        for epoch in range(1, epochs + 1):
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_X = X_train[start:end]
                batch_y = y_train[start:end].reshape(-1, 1)
                
                out, cache = self.forward(batch_X)
                d_out = 2.0 * (out - batch_y)
                
                dW, db, dW1, db1, dW2, db2 = self.backward(d_out, cache)
                
                t_step += 1
                for p, grad, m, v in [
                    (self.W, dW, m_W, v_W),
                    (self.b, db, m_b, v_b),
                    (self.W_d1, dW1, m_W1, v_W1),
                    (self.b_d1, db1, m_b1, v_b1),
                    (self.W_d2, dW2, m_W2, v_W2),
                    (self.b_d2, db2, m_b2, v_b2),
                ]:
                    m[:] = beta1 * m + (1.0 - beta1) * grad
                    v[:] = beta2 * v + (1.0 - beta2) * (grad ** 2)
                    m_hat = m / (1.0 - beta1 ** t_step)
                    v_hat = v / (1.0 - beta2 ** t_step)
                    p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
                    
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_loss = float(np.mean((val_pred.flatten() - y_val.flatten()) ** 2))
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_weights = (
                        self.W.copy(), self.b.copy(),
                        self.W_d1.copy(), self.b_d1.copy(),
                        self.W_d2.copy(), self.b_d2.copy()
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if best_weights is not None:
                            self.W, self.b, self.W_d1, self.b_d1, self.W_d2, self.b_d2 = best_weights
                        break
        return self


def build_tf_lstm_model(lookback: int) -> Any:
    """Build the standard TensorFlow/Keras LSTM model."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(lookback, 1)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_and_forecast(
    history: pd.DataFrame,
    ticker: str,
    lookback: int = 60,
    train_fraction: float = 0.8,
    epochs: int = 25,
    batch_size: int = 32,
    patience: int = 5,
    force_numpy_engine: bool = False,
) -> ForecastResult:
    """Train an LSTM model with chronological validation and holdout test evaluation.
    
    Methodology:
    1. Sort historical observations chronologically.
    2. Fit MinMaxScaler on training data only (prevents data leakage).
    3. Generate sliding windows with lookback past trading sessions.
    4. Maintain chronological validation tail (10% of training data) for early stopping.
    5. Evaluate on unseen chronological holdout test set.
    6. Compute MAE, RMSE, and MAPE for LSTM and persistence baseline.
    7. Generate one-step-ahead next trading session estimate.
    
    If TensorFlow is blocked by host operating system security policies (e.g. Windows Smart App Control),
    it automatically falls back to the exact PureNumPyLSTM mathematical engine without failing.
    """
    if "Close" not in history.columns:
        raise ValueError("history DataFrame must contain a 'Close' column")

    data = history[["Close"]].dropna().copy().sort_index()
    if len(data) < max(150, lookback + 30):
        raise ValueError(
            f"Not enough observations ({len(data)}) for training, validation and testing. "
            f"Need at least {max(150, lookback + 30)} rows."
        )
    if lookback < 5:
        raise ValueError("lookback should be at least 5 trading sessions.")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive integers.")

    # Reserve the final chronological section for evaluation
    split_at = int(len(data) * train_fraction)
    train_close = data["Close"].iloc[:split_at].to_numpy().reshape(-1, 1)

    # Scaler fitted strictly on training data
    scaler = RobustMinMaxScaler()
    scaler.fit(train_close)
    scaled_all = scaler.transform(data["Close"].to_numpy().reshape(-1, 1))

    X_all, y_all = make_sequences(scaled_all, lookback)
    target_indices = np.arange(lookback, len(data))
    train_mask = target_indices < split_at
    test_mask = target_indices >= split_at

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    if len(X_train) < 50 or len(X_test) < 10:
        raise ValueError("Not enough train/test sequences. Increase history period or reduce lookback.")

    # Chronological validation tail
    val_cut = max(1, int(len(X_train) * 0.9))
    X_fit, y_fit = X_train[:val_cut], y_train[:val_cut]
    X_val, y_val = X_train[val_cut:], y_train[val_cut:]
    if len(X_val) == 0:
        X_fit, y_fit = X_train[:-1], y_train[:-1]
        X_val, y_val = X_train[-1:], y_train[-1:]

    model = None
    engine_type = "tensorflow"

    if not force_numpy_engine:
        try:
            import tensorflow as tf
            model = build_tf_lstm_model(lookback)
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=patience, restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", patience=max(2, patience // 2), factor=0.5, min_lr=1e-5
                ),
            ]
            model.fit(
                X_fit,
                y_fit,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                shuffle=False,
                callbacks=callbacks,
                verbose=0,
            )
            pred_scaled = model.predict(X_test, verbose=0).reshape(-1, 1)
            next_window = scaled_all[-lookback:].reshape(1, lookback, 1)
            next_scaled = model.predict(next_window, verbose=0).reshape(-1, 1)
        except Exception as exc:
            logger.warning(
                f"TensorFlow execution unavailable or blocked by system policy ({exc}). "
                "Falling back to pure NumPy LSTM analytical engine."
            )
            model = None

    if model is None:
        engine_type = "pure_numpy"
        model = PureNumPyLSTM(input_dim=1, hidden_dim=32, dense_dim=16, lr=0.005, seed=42)
        model.fit(
            X_fit,
            y_fit,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
        )
        pred_scaled = model.predict(X_test).reshape(-1, 1)
        next_window = scaled_all[-lookback:].reshape(1, lookback, 1)
        next_scaled = model.predict(next_window).reshape(-1, 1)

    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
    predicted = scaler.inverse_transform(pred_scaled).reshape(-1)

    # Persistence baseline: predict next close using most recently observed close
    baseline_scaled = scaled_all[target_indices[test_mask] - 1]
    baseline = scaler.inverse_transform(baseline_scaled.reshape(-1, 1)).reshape(-1)

    test_frame = data.iloc[target_indices[test_mask]].copy()
    next_price = float(scaler.inverse_transform(next_scaled)[0, 0])

    return ForecastResult(
        ticker=validate_ticker(ticker),
        history=data,
        test_frame=test_frame,
        actual=actual,
        predicted=predicted,
        baseline=baseline,
        next_session_prediction=next_price,
        next_session_input_date=pd.Timestamp(data.index[-1]),
        metrics=calculate_metrics(actual, predicted),
        baseline_metrics=calculate_metrics(actual, baseline),
        model=model,
        scaler=scaler,
        lookback=lookback,
        engine_type=engine_type,
    )


def save_outputs(result: ForecastResult, output_dir: Union[str, Path] = "outputs") -> Path:
    """Save test prediction comparisons to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame({
        "Date": result.test_frame.index,
        "Actual_Close": result.actual,
        "LSTM_Predicted_Close": result.predicted,
        "Persistence_Baseline": result.baseline,
    })
    safe_ticker = result.ticker.replace(".", "_")
    path = output_dir / f"{safe_ticker}_test_predictions.csv"
    report.to_csv(path, index=False)
    return path
