from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.stock_forecaster import (
    DEFAULT_TICKERS,
    fetch_history,
    train_and_forecast,
    save_outputs,
    validate_ticker,
)

st.set_page_config(
    page_title="LSTM Stock Forecast Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom responsive CSS for mobile/tablet/desktop viewports
st.markdown(
    """
    <style>
    /* Responsive container adjustments */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            padding-bottom: 2rem !important;
        }
        div[data-testid="stMetric"] {
            background: #f8fafc;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.4rem !important;
        }
        .stButton button {
            min-height: 48px;
            font-size: 1.05rem;
        }
    }
    .badge-info {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        background-color: #e0f2fe;
        color: #0369a1;
        font-size: 0.85rem;
        font-weight: 500;
        margin-right: 6px;
    }
    .status-box {
        background-color: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 12px;
        margin: 12px 0;
        border-radius: 0 6px 6px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 Stock Price Forecasting with LSTM")
st.markdown(
    '<span class="badge-info">Authentic Market Data</span>'
    '<span class="badge-info">Chronological Holdout Validation</span>'
    '<span class="badge-info">Persistence Baseline Benchmark</span>',
    unsafe_allow_html=True,
)
st.caption("Educational time-series forecasting lab • Indian NSE & US markets • Zero fabricated data")

# Caching market data to avoid re-downloading on every Streamlit interaction (e.g. slider moves)
@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_history(ticker: str, period: str) -> pd.DataFrame:
    return fetch_history(ticker, period=period)

# Caching model training results based on key parameters to avoid unnecessary retraining
@st.cache_data(show_spinner=False)
def run_cached_forecast(ticker: str, period: str, lookback: int, train_fraction: float, epochs: int):
    history = get_cached_history(ticker, period=period)
    result = train_and_forecast(
        history,
        ticker,
        lookback=lookback,
        train_fraction=train_fraction,
        epochs=epochs,
    )
    return history, result

with st.sidebar:
    st.header("⚙️ Forecast Settings")
    
    labels = list(DEFAULT_TICKERS.keys())
    selected_label = st.selectbox(
        "Popular Stocks",
        labels,
        index=0,
        help="Select a benchmark Indian NSE or US stock, or type your own below."
    )
    ticker_default = DEFAULT_TICKERS[selected_label]
    
    ticker_input = st.text_input(
        "Yahoo Finance Ticker Symbol",
        value=ticker_default,
        help="e.g. RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, AAPL, MSFT, NVDA"
    )
    
    period = st.selectbox(
        "Historical Period",
        ["1y", "2y", "5y", "10y"],
        index=2,
        help="Amount of historical daily OHLCV market history to download."
    )
    
    with st.expander("Model Hyperparameters", expanded=False):
        lookback = st.slider("Lookback Window (trading days)", 10, 120, 60, 5)
        epochs = st.slider("Maximum Training Epochs", 5, 100, 25, 5)
        train_fraction = st.slider("Chronological Train Split", 0.60, 0.90, 0.80, 0.05)
    
    run = st.button("🚀 Fetch Data & Train LSTM", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption(
        "**Data Policy:** Genuine market data via `yfinance`. If the download fails or the symbol is invalid, "
        "an error is displayed. No synthetic prices are ever generated."
    )

st.info(
    "💡 **Methodology:** Forecasts the next trading session closing price from a sliding historical window. "
    "MinMaxScaler is fitted solely on training observations to eliminate look-ahead leakage. "
    "This tool is for education and research, not financial advice."
)

if run:
    try:
        clean_ticker = validate_ticker(ticker_input)
        with st.spinner(f"📥 Downloading genuine historical market data for {clean_ticker}…"):
            history = get_cached_history(clean_ticker, period=period)
            
        latest_date = history.index[-1].date()
        latest_close = float(history["Close"].iloc[-1])
        st.success(
            f"✅ Ingested {len(history):,} trading observations for **{clean_ticker}** (Provider: Yahoo Finance). "
            f"Date Range: **{history.index[0].date()}** to **{latest_date}** (Latest Close: **{latest_close:,.2f}**)."
        )
        
        with st.spinner("🧠 Training LSTM and evaluating on chronological holdout…"):
            history, result = run_cached_forecast(
                clean_ticker,
                period=period,
                lookback=lookback,
                train_fraction=train_fraction,
                epochs=epochs,
            )
            
        st.session_state["forecast_result"] = result
        st.session_state["history"] = history
        st.session_state["ticker"] = clean_ticker
        
    except Exception as exc:
        st.error(f"❌ Forecast execution failed: {exc}")
        st.caption("Please check the ticker symbol, network connectivity, or adjust the historical period.")

result = st.session_state.get("forecast_result")
if result is not None:
    history = st.session_state["history"]
    ticker = st.session_state["ticker"]
    
    st.markdown("---")
    st.subheader(f"📊 Forecast Report for {ticker}")
    
    # Responsive top metrics cards with explicit backend engine disclosure
    m1, m2, m3, m4 = st.columns([1, 1, 1, 1])
    with m1:
        st.metric(
            label="LSTM Next-Session Estimate",
            value=f"{result.next_session_prediction:,.2f}",
            delta=f"{result.next_session_prediction - float(history['Close'].iloc[-1]):+,.2f} vs last close",
            delta_color="off",
        )
    with m2:
        st.metric(
            label="Last Observed Close",
            value=f"{float(history['Close'].iloc[-1]):,.2f}",
        )
    with m3:
        st.metric(
            label="Last Observation Date",
            value=str(result.next_session_input_date.date()),
        )
    with m4:
        engine_label = "TensorFlow / Keras" if result.engine_type == "tensorflow" else "Pure NumPy LSTM"
        st.metric(
            label="Training Engine",
            value=engine_label,
            help="Discloses which computational backend actually trained the LSTM model."
        )
        
    st.caption(
        "ℹ️ The next-session estimate is for the following trading session (dependent on exchange holiday calendar). "
        "It represents a single mathematical projection, not a guaranteed outcome or investment advice."
    )
    
    # Performance Evaluation & Baseline Comparison
    st.markdown("### 🎯 Unseen Holdout Evaluation vs Persistence Baseline")
    col_metrics, col_chart_hist = st.columns([1, 1])
    
    with col_metrics:
        metrics_records = [
            {
                "Model": "LSTM Neural Network",
                "MAE": result.metrics["MAE"],
                "RMSE": result.metrics["RMSE"],
                "MAPE (%)": result.metrics["MAPE_percent"],
            },
            {
                "Model": "Persistence Baseline (Last Close)",
                "MAE": result.baseline_metrics["MAE"],
                "RMSE": result.baseline_metrics["RMSE"],
                "MAPE (%)": result.baseline_metrics["MAPE_percent"],
            },
        ]
        metrics_df = pd.DataFrame(metrics_records).set_index("Model")
        st.dataframe(
            metrics_df.style.format({
                "MAE": "{:,.4f}",
                "RMSE": "{:,.4f}",
                "MAPE (%)": "{:.2f}%",
            }),
            use_container_width=True,
        )
        
        # Transparent scientific reporting on underperformance
        if result.metrics["RMSE"] < result.baseline_metrics["RMSE"]:
            st.success(
                f"📈 **LSTM outperformed persistence baseline on RMSE:** {result.metrics['RMSE']:.4f} vs {result.baseline_metrics['RMSE']:.4f}. "
                "Note: A single historical holdout does not prove future outperformance or trading profitability."
            )
        else:
            st.warning(
                f"📉 **Persistence Baseline outperformed the LSTM on RMSE:** {result.baseline_metrics['RMSE']:.4f} vs {result.metrics['RMSE']:.4f}. "
                "In accordance with financial market economics (Efficient Market Hypothesis), daily stock price levels "
                "exhibit strong random walk properties where yesterday's close is statistically difficult for univariate models to beat. "
                "This outcome is reported honestly; we do not tune on the holdout test set to conceal it."
            )
            
    with col_chart_hist:
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Scatter(
                x=history.index,
                y=history["Close"],
                name="Historical Close",
                line=dict(color="#2563eb", width=1.5),
            )
        )
        fig_hist.update_layout(
            title=f"{ticker} Complete Historical Series",
            xaxis_title="Date",
            yaxis_title="Close Price",
            height=300,
            margin=dict(l=10, r=10, t=35, b=10),
            hovermode="x unified",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Actual vs Predicted Main Chart
    st.markdown("### 🔍 Actual vs Predicted on Chronological Holdout Set")
    fig_eval = go.Figure()
    fig_eval.add_trace(
        go.Scatter(
            x=result.test_frame.index,
            y=result.actual,
            name="Actual Close",
            line=dict(color="#1e293b", width=2),
        )
    )
    fig_eval.add_trace(
        go.Scatter(
            x=result.test_frame.index,
            y=result.predicted,
            name="LSTM Prediction",
            line=dict(color="#0284c7", width=2),
        )
    )
    fig_eval.add_trace(
        go.Scatter(
            x=result.test_frame.index,
            y=result.baseline,
            name="Persistence Baseline",
            line=dict(color="#94a3b8", width=1.5, dash="dot"),
        )
    )
    fig_eval.update_layout(
        xaxis_title="Date",
        yaxis_title="Closing Price",
        hovermode="x unified",
        height=450,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_eval, use_container_width=True)
    
    # Export and Data Download
    output_path = save_outputs(result)
    csv_bytes = pd.read_csv(output_path).to_csv(index=False).encode("utf-8")
    
    col_dl1, col_dl2 = st.columns([1, 3])
    with col_dl1:
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv_bytes,
            file_name=output_path.name,
            mime="text/csv",
            use_container_width=True,
        )
    with col_dl2:
        st.caption(f"File also saved locally to `{output_path}` with {len(result.test_frame)} evaluated dates.")

else:
    st.markdown("### 📖 About this Application")
    st.markdown(
        """
        This application provides an end-to-end machine learning lab for stock closing price estimation:
        
        1. **Genuine Historical Market Data:** Directly downloaded from Yahoo Finance via `yfinance` for Indian NSE stocks (e.g., `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`) and US equities (`AAPL`, `MSFT`, `NVDA`, `TSLA`).
        2. **Strict Chronological Splitting:** Avoids random train-test splitting to ensure no lookahead leakage.
        3. **Train-Only Feature Scaling:** Normalization boundaries are learned strictly from the training period.
        4. **Deep LSTM Sequence Modeling:** Sliding lookback windows map sequential prices to next-session targets.
        5. **Rigorous Persistence Benchmarking:** Compares model against a naive persistence baseline (last observed close) with MAE, RMSE, and MAPE metrics.
        6. **Honest Out-of-Sample Reporting:** Displays true model metrics whether the model beats or underperforms the baseline.
        
        👉 **To begin, select a stock from the sidebar and click 'Fetch Data & Train LSTM'.**
        """
    )
