# Responsive UI Audit Report — Live Streamlit Application

**Live URL:** https://shivang-stock-forecasting.streamlit.app/  
**Repository:** https://github.com/shivangsaxena1011/stock-forecasting  
**Audit Date:** October 12, 2026  
**Auditor Engine:** Playwright Chromium / Microsoft Edge Automation  
**Testing Modality:** Emulated Browser Viewports (explicitly distinguished from physical hardware testing)  

---

## 1. Executive Summary

An automated responsive audit and functional interaction test was executed on the live deployed Streamlit application across all **8 specified target viewports**.

Every viewport was verified for:
1. Initial page loading, layout structure, header badges, and typography.
2. Sidebar controls (popular stocks dropdown, custom ticker input, historical period selectbox, hyperparameter sliders expander, and forecast action button).
3. End-to-end forecasting pipeline execution with **genuine Yahoo Finance market data** (`RELIANCE.NS`, 5-year daily OHLCV series, 1,236 actual trading observations).
4. Unseen chronological holdout evaluation display, persistence baseline comparison table, and interactive Plotly time-series charts.
5. Next-session estimate cards and backend engine disclosure card.
6. Predictions CSV download CTA.
7. Verification of zero horizontal overflow (`document.documentElement.scrollWidth <= window.innerWidth`).

---

## 2. Tested Viewport Matrix & Measured Geometry

| Target Device Profile | Emulated Viewport | Initial Overflow | Post-Forecast Overflow | Metrics Cards Rendered | Plotly Charts Rendered | Download CTA Visible |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Small Mobile** | 320 × 568 | None (`False`) | None (`False`) | 4 stacked (`w: 296px`) | 2 (`w: 296px`) | Yes |
| **Mobile** | 375 × 812 | None (`False`) | None (`False`) | 4 stacked (`w: 351px`) | 2 (`w: 351px`) | Yes |
| **Mobile** | 390 × 844 | None (`False`) | None (`False`) | 4 stacked (`w: 366px`) | 2 (`w: 366px`) | Yes |
| **Large Mobile** | 430 × 932 | None (`False`) | None (`False`) | 4 stacked (`w: 406px`) | 2 (`w: 406px`) | Yes |
| **Tablet** | 768 × 1024 | None (`False`) | None (`False`) | 4 inline (`w: 174px`) | 2 (`w: 364px`, `w: 744px`) | Yes |
| **Laptop** | 1024 × 768 | None (`False`) | None (`False`) | 4 inline (`w: 129px`) | 2 (`w: 274px`, `w: 564px`) | Yes |
| **Desktop** | 1366 × 768 | None (`False`) | None (`False`) | 4 inline (`w: 215px`) | 2 (`w: 445px`, `w: 906px`) | Yes |
| **Large Desktop** | 1440 × 900 | None (`False`) | None (`False`) | 4 inline (`w: 233px`) | 2 (`w: 482px`, `w: 980px`) | Yes |

*Note: All tests were performed in emulated browser contexts in Chromium/Edge; physical touchscreen gestures and OEM browser variations require physical hardware testing.*

---

## 3. Discovered Visual Opportunities & Remediations Applied

### Issue 1: Metric Label Ellipsis & Wrapping on Intermediate Viewports (1024px)
- **Observation:** On 1024px screens with the sidebar expanded, the available main container width divided across 4 columns resulted in ~129px column widths. Long labels such as "LSTM Next-Session Estimate" risked ellipsis truncation in standard Streamlit font sizes.
- **Fix:** In `app.py`, added dedicated media query `@media (min-width: 769px) and (max-width: 1200px)` adjusting `div[data-testid="stMetricLabel"]` font size to `0.8rem` with `white-space: normal` and `word-break: break-word`, and value size to `1.25rem`.

### Issue 2: Mobile Metric Card Padding & Word Break (320px–430px)
- **Observation:** On 320px narrow screens, metrics stack vertically into cards. Without explicit word breaking, large numbers or delta annotations could crowd the card edges.
- **Fix:** Added `word-break: break-word !important` to `div[data-testid="stMetricLabel"]` and `div[data-testid="stMetricValue"]` under `@media (max-width: 768px)`.

### Issue 3: Plotly Holdout Chart Legend Positioning on Mobile
- **Observation:** The actual vs. predicted chart had its legend positioned at `xanchor="right", x=1, y=1.02`. On narrow viewports with multi-item legends (Actual Close, LSTM Prediction, Persistence Baseline), right-aligned legends could collide with chart titles or axis boundaries.
- **Fix:** Adjusted legend layout to `xanchor="left", x=0, y=1.02` with explicit `font=dict(size=11)` to ensure graceful wrapping on mobile and desktop screens.

---

## 4. Verification & Regression Testing

1. **Automated Test Suite:**
   ```powershell
   .\.venv\Scripts\python.exe -m pytest -v
   ```
   - **Result:** **40 passed, 1 deselected (live integration)** in 2.34s.
   - Zero test regressions.

2. **Generated Audit Artifacts:**
   - Saved baseline and interaction screenshots in `outputs/responsive_audit/`.
   - Recorded full metrics geometry in `outputs/responsive_audit/audit_summary.json`.
