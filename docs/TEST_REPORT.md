# TEST REPORT — STOCK PRICE PREDICTION USING LSTM

**Date of Execution:** 2026-10-12  
**Test Environment:** Windows 11 (64-bit), Python 3.12.10, pytest 8.4.2  
**Author:** Senior QA Systems & ML Verification Engineer  

---

## 1. Test Execution Summary

| Test Category | Total Planned | Passed | Failed | Deselected | Status |
|---|---|---|---|---|---|
| Offline Unit & Validation Suite (`pytest -v`) | 40 | 40 | 0 | 1 | **PASSED** |
| Real Market Data Integration Smoke (`pytest -m "integration"`) | 1 | 1 | 0 | 40 | **PASSED** |
| Complete Combined Test Run (`pytest -m ""`) | 41 | 41 | 0 | 0 | **PASSED** |
| Model Training Smoke Tests | 2 | 2 | 0 | 0 | **PASSED** |
| Streamlit State & Caching Tests (`test_app.py`) | 4 | 4 | 0 | 0 | **PASSED** |
| FastAPI Backend Endpoints (`test_api.py`) | 8 | 8 | 0 | 0 | **PASSED** |
| Jupyter Notebook Full Execution | 1 | 1 | 0 | 0 | **PASSED** |
| Responsive Viewport Verification (Genuine Rendered Screens) | 9 | 9 | 0 | 0 | **PASSED** |
| CSV Export & Metric Consistency | 2 | 2 | 0 | 0 | **PASSED** |

---

## 2. Detailed Test Cases and Results

### A. Unit Tests & Mathematical Verification (pytest)
- **Default Offline Command:** `python -m pytest -v`
- **Default Offline Output:** `40 passed, 1 deselected in 3.69s`
- **Explicit Integration Command:** `python -m pytest -m "integration" -v`
- **Explicit Integration Output:** `1 passed, 40 deselected in 4.21s`
- **Complete Suite Command:** `python -m pytest -m "" -v`
- **Complete Suite Output:** `41 passed in 4.58s`

#### Test Case Breakdown

| Test Name | File | Description | Result | Output / Evidence |
|---|---|---|---|---|
| `test_health_endpoint` | `test_api.py` | Validates `GET /health` returns HTTP 200 with service metadata | PASSED | `{"status":"ok"}` |
| `test_tickers_endpoint` | `test_api.py` | Validates `GET /tickers` returns dictionary of supported equities | PASSED | Returns 10 equity tickers |
| `test_predict_invalid_ticker_format` | `test_api.py` | Validates ticker syntax errors produce HTTP 400 | PASSED | HTTP 400 Bad Request |
| `test_predict_invalid_period` | `test_api.py` | Validates period allowlist enforcement produces HTTP 422 | PASSED | HTTP 422 Unprocessable |
| `test_predict_provider_failure` | `test_api.py` | Validates provider network errors map to HTTP 502 Bad Gateway | PASSED | HTTP 502 Bad Gateway |
| `test_predict_model_training_failure` | `test_api.py` | Validates internal errors map to sanitized HTTP 500 without stack leaks | PASSED | HTTP 500 sanitized detail |
| `test_predict_successful_forecast` | `test_api.py` | Validates full forecasting payload matches ForecastResponse schema | PASSED | HTTP 200 with valid schema |
| `test_rate_limiter_unit` | `test_api.py` | Tests in-memory sliding-window rate limiter per client IP | PASSED | Rejects on threshold limit |
| `test_popular_stock_sync_updates_ticker` | `test_app.py` | Tests dropdown selection callback updates `st.session_state["ticker_input"]` | PASSED | Ticker synced to TCS.NS |
| `test_custom_ticker_manual_override_preserved` | `test_app.py` | Tests custom typed ticker is preserved without unexpected override | PASSED | NVDA preserved |
| `test_failed_forecast_clears_previous_result` | `test_app.py` | Confirms failed forecast invalidates previous result in session state | PASSED | Stale result cleared |
| `test_run_cached_forecast_signature_accepts_history` | `test_app.py` | Confirms forecast cache binds to historical DataFrame instance | PASSED | Cache key depends on data |
| `test_robust_min_max_scaler_fit_transform_inverse` | `test_stock_forecaster.py` | Verifies forward scaling and exact mathematical inverse | PASSED | Scaled [0, 1], inverse exact |
| `test_robust_min_max_scaler_constant_series` | `test_stock_forecaster.py` | Validates constant array without zero-division error | PASSED | Handled without NaN |
| `test_robust_min_max_scaler_no_test_leakage` | `test_stock_forecaster.py` | Confirms train-fitted scaler ignores out-of-bound test data | PASSED | Bounds invariant on test data |
| `test_make_sequences_shape_and_targets` | `test_stock_forecaster.py` | Verifies window sliding and next-step target alignment | PASSED | Shape: (7, 3, 1), targets match |
| `test_make_sequences_target_alignment_prevents_leakage` | `test_stock_forecaster.py` | Strictly verifies that target $y_i$ is strictly after window $X_i$ | PASSED | Target not in sequence window |
| `test_make_sequences_rejects_short_data` | `test_stock_forecaster.py` | Confirms ValueError when len <= lookback | PASSED | ValueError caught |
| `test_make_sequences_lookback_boundary` | `test_stock_forecaster.py` | Checks lookback < 1 validation | PASSED | ValueError caught |
| `test_chronological_split_keeps_order` | `test_stock_forecaster.py` | Confirms strict time ordering without shuffling | PASSED | Train 80%, Test 20%, sorted |
| `test_chronological_split_invalid_fraction` | `test_stock_forecaster.py` | Checks boundary fractions (< 0.5 or >= 1.0) | PASSED | ValueError caught |
| `test_validate_ticker` | `test_stock_forecaster.py` | Validates whitespace trimming, uppercasing, illegal chars | PASSED | Normalized TCS.NS and AAPL |
| `test_metrics_perfect_prediction` | `test_stock_forecaster.py` | Tests MAE=0, RMSE=0, MAPE=0 on identical arrays | PASSED | All metrics evaluated to 0.0 |
| `test_metrics_handles_zero_values` | `test_stock_forecaster.py` | Tests zero denominator in MAPE | PASSED | Returns NaN safely without crash |
| `test_metrics_handles_empty_arrays` | `test_stock_forecaster.py` | Tests empty input arrays | PASSED | Returns NaN safely |
| `test_backward_compatibility_alias` | `test_stock_forecaster.py` | Tests legacy `_metrics` alias function | PASSED | Alias returned expected dict |
| `test_pure_numpy_lstm_numerical_gradient_check` | `test_stock_forecaster.py` | Compares analytical BPTT gradients to numerical finite differences ($\epsilon=10^{-6}$) | PASSED | Relative error $< 2\times 10^{-5}$ |
| `test_pure_numpy_lstm_learning_convergence` | `test_stock_forecaster.py` | Verifies PureNumPyLSTM strictly reduces MSE loss on sequential sine wave | PASSED | Loss reduced by $> 90\%$ |
| `test_pure_numpy_lstm_best_weights_restoration` | `test_stock_forecaster.py` | Verifies early stopping restores best validation weights | PASSED | Best validation weights restored |
| `test_pure_numpy_lstm_restores_best_weights_on_full_epochs` | `test_stock_forecaster.py` | Verifies best weights restored when training runs full epochs without early trigger | PASSED | Best validation weights restored |
| `test_train_and_forecast_synthetic_pipeline_smoke` | `test_stock_forecaster.py` | Full pipeline smoke test on synthetic DataFrame fixture | PASSED | Holdout metrics & next step computed |
| `test_fetch_history_mock_empty_raises_runtime_error` | `test_stock_forecaster.py` | Verifies RuntimeError when provider returns empty DataFrame | PASSED | RuntimeError raised as expected |
| `test_fetch_history_mock_missing_close_raises_runtime_error` | `test_stock_forecaster.py` | Verifies RuntimeError when DataFrame lacks Close column | PASSED | RuntimeError raised as expected |
| `test_fetch_history_mock_insufficient_rows_raises_runtime_error` | `test_stock_forecaster.py` | Verifies RuntimeError when history has fewer rows than lookback window | PASSED | RuntimeError raised as expected |
| `test_fetch_history_mock_multiindex_resolution` | `test_stock_forecaster.py` | Verifies clean MultiIndex column resolution and date sorting | PASSED | Extracted valid Series correctly |
| `test_fetch_history_missing_yfinance_raises_actionable_import_error` | `test_stock_forecaster.py` | Verifies actionable ImportError with pip instructions when yfinance missing | PASSED | Actionable ImportError raised |
| `test_train_and_forecast_validates_train_fraction` | `test_stock_forecaster.py` | Validates train_fraction boundaries (<0.5 or >=1.0) | PASSED | ValueError raised |
| `test_train_and_forecast_validates_non_finite_values` | `test_stock_forecaster.py` | Validates non-finite (inf/nan) closing price rejection | PASSED | ValueError raised |
| `test_train_and_forecast_validates_lookback_and_epochs` | `test_stock_forecaster.py` | Validates lookback and epoch parameter bounds | PASSED | ValueError raised |
| `test_train_and_forecast_tf_unexpected_error_not_masked` | `test_stock_forecaster.py` | Confirms unexpected TensorFlow programming errors propagate instead of masking | PASSED | Genuine error re-raised |
| `test_fetch_history_live_smoke` | `test_stock_forecaster.py` | Live Yahoo Finance market data smoke test (opt-in integration) | PASSED | Fetched genuine OHLCV data for AAPL |

---

## 3. Real Market Data Integration & Metric Alignment

- **Target Experiment:** `TCS.NS` (Tata Consultancy Services, NSE)
  - **Date Range:** `2021-10-11` to `2026-10-09` (1,236 rows, 5-year history)
  - **Holdout Test Set:** 248 trading sessions (`2025-10-09` to `2026-10-09`)
  - **Lookback:** 60 sessions, 80% train / 20% test
  - **Predictions Output:** [`outputs/TCS_NS_test_predictions.csv`](outputs/TCS_NS_test_predictions.csv)
  - **Recalculated from CSV:**
    - LSTM MAE: **₹192.11**
    - LSTM RMSE: **₹227.12**
    - LSTM MAPE: **8.25%**
    - Persistence MAE: **₹32.14**
    - Persistence RMSE: **₹45.02**
    - Persistence MAPE: **1.29%**
  - **Consistency Check:** Matches executed notebook outputs cell 7, project report, and test report tables.
  - **Baseline Comparison Outcome:** The persistence baseline outperformed the LSTM across MAE, RMSE, and MAPE. Reported transparently.

---

## 4. Jupyter Notebook End-to-End Execution

- **Command:** `python -m jupyter nbconvert --to notebook --execute --inplace notebooks/Stock_Price_Prediction_LSTM.ipynb`
- **Result:** **PASSED**. Executed successfully from top to bottom on real Yahoo Finance market data.
- **Visualizations:** Matplotlib figure 1 (historical closing series) and figure 2 (holdout actual vs LSTM predicted vs persistence baseline) generated and embedded into the notebook.

---

## 5. Responsive Viewport Visual Verification

Streamlit single-page application visually verified and captured across standard screen dimensions:
1. `outputs/screenshots/viewport_320_mobile_small.png` (320 x 640): Compact layout, stacked inputs, readable typography.
2. `outputs/screenshots/viewport_375_mobile_medium.png` (375 x 667): Clean responsive sidebar and main banner.
3. `outputs/screenshots/viewport_390_mobile_iphone.png` (390 x 844): iPhone viewport with zero horizontal overflow.
4. `outputs/screenshots/viewport_430_mobile_large.png` (430 x 932): Large mobile viewport with full CTA visibility.
5. `outputs/screenshots/viewport_768_tablet.png` (768 x 1024): Tablet viewport with proportional column spacing.
6. `outputs/screenshots/viewport_1024_desktop_small.png` (1024 x 768): Desktop layout with visible side panel.
7. `outputs/screenshots/viewport_1440_desktop_wide.png` (1440 x 900): Full wide desktop layout with expanded controls.
8. `outputs/screenshots/results_1440_desktop.png` (1440 x 1100): Executed forecast results screen on desktop.
9. `outputs/screenshots/results_390_mobile.png` (390 x 1200): Executed forecast results screen on mobile.

---

## 6. Live Application Deployment Status

- **Live URL:** `https://shivang-stock-forecasting.streamlit.app/`
- **HTTP Status Check:** `HTTP 200 OK`
- **Live Verification Notes:** The hosted Streamlit Community Cloud application is active and serving requests. In the event the container enters idle sleep mode, Streamlit automatically wakes and restores the app upon client visit.
