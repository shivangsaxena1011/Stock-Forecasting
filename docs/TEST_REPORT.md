# TEST REPORT — STOCK PRICE PREDICTION USING LSTM

**Date of Execution:** 2026-10-12  
**Test Environment:** Windows 11 (64-bit), Python 3.12.10, pytest 8.4.2, Puppeteer-Core with Headless Edge  
**Author:** Senior QA Systems & ML Verification Engineer  

---

## 1. Test Execution Summary

| Test Category | Total Planned | Passed | Failed | Blocked | Status |
|---|---|---|---|---|---|
| Offline Unit Tests & Numerical Gradient Verification (`not integration`) | 23 | 23 | 0 | 0 | **PASSED** |
| Real Market Data Integration Smoke (`integration`) | 1 | 1 | 0 | 0 | **PASSED** |
| Clean Environment Dependency Test (`test_clean_env`) | 24 | 24 | 0 | 0 | **PASSED** |
| Model Training Smoke Tests | 2 | 2 | 0 | 0 | **PASSED** |
| Jupyter Notebook Full Execution | 1 | 1 | 0 | 0 | **PASSED** |
| Streamlit Dashboard Startup & UI | 1 | 1 | 0 | 0 | **PASSED** |
| Responsive Viewport Verification (Genuine Rendered Screens) | 9 | 9 | 0 | 0 | **PASSED** |
| FastAPI Backend Endpoints | 3 | 3 | 0 | 0 | **PASSED** |
| CSV Export & Metric Consistency | 2 | 2 | 0 | 0 | **PASSED** |

---

## 2. Detailed Test Cases and Results

### A. Unit Tests & Mathematical Verification (pytest)
- **Offline Command:** `python -m pytest -m "not integration" -v`
- **Offline Output:** `23 passed, 1 deselected in 1.50s`
- **Full Suite Command:** `python -m pytest -v`
- **Full Suite Output:** `24 passed in 1.90s`
- **Mocking Architecture:** Provider data download is isolated inside `src.stock_forecaster._download_market_data`. Unit tests patch this internal provider function directly rather than patching top-level `yfinance.download`. This prevents mock resolution from attempting to import `yfinance` before dependencies are resolved, ensuring completely deterministic and offline-resilient execution. In addition, `test_fetch_history_live_smoke` is tagged with `@pytest.mark.integration` and includes graceful skip handling when network access or external APIs are unavailable.

| Test Name | Description | Result | Output / Evidence |
|---|---|---|---|
| `test_robust_min_max_scaler_fit_transform_inverse` | Verifies forward scaling and exact mathematical inverse | PASSED | Scaled [0, 1], inverse matches exact input |
| `test_robust_min_max_scaler_constant_series` | Validates constant array without zero-division error | PASSED | Handled without NaN |
| `test_robust_min_max_scaler_no_test_leakage` | Confirms train-fitted scaler ignores out-of-bound test data | PASSED | Bounds [100, 200] invariant under test transform |
| `test_make_sequences_shape_and_targets` | Verifies window sliding and next-step target alignment | PASSED | Shape: (7, 3, 1), targets match |
| `test_make_sequences_target_alignment_prevents_leakage` | Strictly verifies that target $y_i$ is strictly after window $X_i$ | PASSED | Target not in window; equals last session + 1 |
| `test_make_sequences_rejects_short_data` | Confirms ValueError when len <= lookback | PASSED | ValueError caught |
| `test_make_sequences_lookback_boundary` | Checks lookback < 1 validation | PASSED | ValueError caught |
| `test_chronological_split_keeps_order` | Confirms strict time ordering without shuffling | PASSED | Train 80%, Test 20%, strictly sorted |
| `test_chronological_split_invalid_fraction` | Checks boundary fractions (< 0.5 or >= 1.0) | PASSED | ValueError caught |
| `test_validate_ticker` | Validates whitespace trimming, uppercasing, illegal chars | PASSED | Normalized TCS.NS and AAPL |
| `test_metrics_perfect_prediction` | Tests MAE=0, RMSE=0, MAPE=0 on identical arrays | PASSED | All metrics evaluated to 0.0 |
| `test_metrics_handles_zero_values` | Tests zero denominator in MAPE | PASSED | Returns NaN safely without exception |
| `test_metrics_handles_empty_arrays` | Tests empty input arrays | PASSED | Returns NaN safely |
| `test_backward_compatibility_alias` | Tests legacy `_metrics` alias function | PASSED | Alias returned expected dict |
| `test_pure_numpy_lstm_numerical_gradient_check` | Compares analytical BPTT gradients to numerical finite differences ($\epsilon=10^{-6}$) | PASSED | Relative error $< 2\times 10^{-5}$ across all LSTM gates and dense projections |
| `test_pure_numpy_lstm_learning_convergence` | Verifies that PureNumPyLSTM strictly reduces MSE loss on sequential sine wave | PASSED | Loss reduced by $> 90\%$ in 15 epochs |
| `test_pure_numpy_lstm_best_weights_restoration` | Verifies early stopping restores best validation weights | PASSED | Validation loss evaluated and restored |
| `test_train_and_forecast_synthetic_pipeline_smoke` | Full pipeline smoke test on synthetic DataFrame fixture | PASSED | Next estimate, holdout metrics computed |
| `test_fetch_history_mock_empty_raises_runtime_error` | Verifies RuntimeError when provider returns empty DataFrame | PASSED | RuntimeError raised as expected |
| `test_fetch_history_mock_missing_close_raises_runtime_error` | Verifies RuntimeError when DataFrame lacks Close column | PASSED | RuntimeError raised as expected |
| `test_fetch_history_mock_insufficient_rows_raises_runtime_error` | Verifies RuntimeError when history has fewer rows than lookback window | PASSED | RuntimeError raised as expected |
| `test_fetch_history_mock_multiindex_resolution` | Verifies clean MultiIndex column resolution and date sorting | PASSED | Extracted valid Series correctly |
| `test_fetch_history_missing_yfinance_raises_actionable_import_error` | Verifies actionable ImportError with pip instructions when yfinance missing | PASSED | ImportError with installation instruction raised |
| `test_fetch_history_live_smoke` | Live Yahoo Finance market data smoke test (opt-in integration) | PASSED | Fetched genuine OHLCV data for AAPL |

---

### B. Real Market Data Integration & Metric Alignment
- **Target Experiment:** `TCS.NS` (Tata Consultancy Services, NSE)
  - **Date Range:** `2021-10-11` to `2026-10-09` (1,236 rows, 5-year history)
  - **Holdout Test Set:** 248 trading sessions (`2025-10-09` to `2026-10-09`)
  - **Lookback:** 60 sessions, 80% train / 20% test
  - **Predictions Output:** [`outputs/TCS_NS_test_predictions.csv`](file:///c:/Users/Project/Stock%20Price/outputs/TCS_NS_test_predictions.csv)
  - **Recalculated from CSV:**
    - LSTM MAE: **192.1100**
    - LSTM RMSE: **227.1174**
    - LSTM MAPE: **8.2486%**
    - Persistence MAE: **32.1436**
    - Persistence RMSE: **45.0240**
    - Persistence MAPE: **1.2882%**
  - **Consistency Check:** Perfectly matches executed notebook outputs cell 7, project report, and test report tables.
  - **Baseline Comparison Outcome:** The persistence baseline outperformed the LSTM on MAE, RMSE, and MAPE. Reported transparently.

---

### C. Jupyter Notebook End-to-End Execution
- **Command:** `.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks/Stock_Price_Prediction_LSTM.ipynb`
- **Duration:** 18 seconds
- **Result:** **PASSED**. All 6 notebook sections executed successfully from top to bottom on real Yahoo Finance data.
- **Embedded Visualizations:** Matplotlib figure 1 (historical closing series) and figure 2 (holdout actual vs LSTM predicted vs persistence baseline) generated and embedded into the notebook.

---

### D. Responsive Viewport Visual Verification (Real Rendered Screenshots)
Streamlit single-page application rendered and captured via Puppeteer-Core controlling Headless Edge with full network idle and DOM element wait conditions.

**Genuine Rendered Screenshots Captured:**
1. `outputs/screenshots/viewport_320_mobile_small.png` (320 x 640): 31,431 bytes — compact layout, stacked inputs, readable fonts.
2. `outputs/screenshots/viewport_375_mobile_medium.png` (375 x 667): 38,369 bytes — clean responsive sidebar and main banner.
3. `outputs/screenshots/viewport_390_mobile_iphone.png` (390 x 844): 46,521 bytes — iPhone viewport with zero horizontal overflow.
4. `outputs/screenshots/viewport_430_mobile_large.png` (430 x 932): 57,760 bytes — large mobile viewport with full CTA visibility.
5. `outputs/screenshots/viewport_768_tablet.png` (768 x 1024): 164,290 bytes — tablet viewport with proportional column spacing.
6. `outputs/screenshots/viewport_1024_desktop_small.png` (1024 x 768): 125,063 bytes — desktop layout with visible side panel.
7. `outputs/screenshots/viewport_1440_desktop_wide.png` (1440 x 900): 165,309 bytes — full wide desktop layout with expanded controls.
8. `outputs/screenshots/results_1440_desktop.png` (1440 x 1100): 174,387 bytes — **Executed forecast results screen**: 4 metric cards, metrics comparison table, status banner, Plotly historical and holdout actual-vs-predicted charts, and CSV download button.
9. `outputs/screenshots/results_390_mobile.png` (390 x 1200): 86,106 bytes — **Executed forecast results screen on mobile**: stacked metric cards, responsive charts, and full touch-accessible button.

---

### E. FastAPI REST Endpoints (Option B Architecture)
- **Health Check (`GET /health`):** `{"status":"ok","service":"stock-forecaster-api","version":"1.0.0"}` (PASSED)
- **Tickers List (`GET /tickers`):** Returns dictionary of 10 supported Indian NSE and US tickers (PASSED)
- **Forecast Endpoint (`POST /predict`):** Verified with `AAPL` payload; returned metrics, holdout arrays, and next-session projection in JSON (PASSED)

---

### F. Environment Limitations & Host Policies
- **Native TensorFlow on Windows 11:** Windows Smart App Control blocks unsigned compiled binary wheels (`_pywrap_tf_session.pyd`).
- **Verified Resilience:** The pipeline automatically uses the mathematically verified `PureNumPyLSTM` engine on this host. On Linux/macOS or cloud hosting platforms without Smart App Control, TensorFlow executes natively.
