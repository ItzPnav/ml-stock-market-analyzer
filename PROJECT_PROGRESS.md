# 📈 Stock Trend Analyzer

## Project Implementation Summary
This project is a **model-driven stock forecasting system** built using **Python + Streamlit**. It allows users to predict future stock prices and trends while preventing data leakage by hiding future records until after the forecast is made. The system supports multiple models (Linear Regression for price forecasting, Logistic Regression for trend prediction, and ARIMA for time-series forecasting) and provides a cross-check mechanism to compare predictions with actual data.

---

## 🧠 Project Vision

The **Stock Trend Analyzer** is a machine learning–based stock forecasting system built using **Python and Streamlit**. The system aims to simulate real-world forecasting conditions by hiding future stock data during model training and then revealing that data afterward to evaluate prediction performance.

The project is designed as an **AI-assisted decision support tool**, not an automated trading system. Its primary goals are:

* Analyze historical stock price data
* Predict short-term price movements
* Predict stock direction (up or down)
* Compare predictions with actual outcomes
* Provide visual insight into prediction accuracy

Key design principles followed:

* Prevent **data leakage**
* Use **manual ML model implementations**
* Maintain **interactive visualization**
* Keep system architecture modular
* Enable **offline execution with real market data**
* Support **live data fetching** via yfinance

The application acts as an educational demonstration of machine learning concepts applied to financial time series data.

---

# 🏗️ Architecture Overview

Current project directory structure:

```
stock predict/
│
├── app.py
├── FILE_STRUCT.md
├── PROJECT_PROGRESS.md
├── requirements.txt
│
├── data/
│   ├── clean_the_csv_files.py
│   └── predownloaded_live_cache/
│       ├── AAPL.csv
│       ├── GLD.csv
│       ├── INFY_NS.csv
│       ├── NVDA.csv
│       ├── RELIANCE_NS.csv
│       ├── TCS_NS.csv
│       └── <any live-searched tickers>.csv
│
├── docs/
│   ├── Machine Learning Based Stock Market Forecasts!.pdf
│   └── official review 1 IOMP stock market predictor (1).pptx
│
└── ML/
    ├── LinearRegressionModel.py
    ├── LogisticRegressionModel.py
    └── ARIMAModel.py  ← in progress
```

System architecture flow:

```
Search Bar (Live) OR Preset Dropdown (Offline)
        │
        ▼
yfinance fetch + save CSV  OR  load from predownloaded_live_cache/
        │
        ▼
Data Loader (Streamlit cache)
        │
        ▼
70/30 Data Split
 ├─ Visible Data (70%)  → shown to user, used for training
 └─ Hidden Data  (30%)  → hidden until cross-check
        │
        ▼
Model Selection (Sidebar)
 ├─ Linear Regression   → price forecast
 ├─ Logistic Regression → UP/DOWN direction
 └─ ARIMA               → time-series forecast (separate section)
        │
        ▼
Forecast Generation
        │
        ▼
Visualization (Matplotlib)
        │
        ▼
Cross-check With Hidden Data
        │
        ▼
Accuracy / Comparison Table
```

---

# ✅ Features Implemented

---

## 1️⃣ Data Handling

* Originally used a mock stock dataset with a `Company` column
* Migrated to one CSV per stock in `data/predownloaded_live_cache/`
* Company-based filtering removed; entire CSV used per stock
* Date column handled as string throughout
* `skiprows=[1]` applied for offline CSVs only (yfinance header quirk)
* Live-fetched CSVs saved as clean files (no extra header row)

---

## 2️⃣ Data Leakage Prevention — 70/30 Split

**Original system (replaced):**
```python
k = min(3, n // 4) if n >= 4 else 1
visible_data = company_data.iloc[:n - k]
hidden_real_data = company_data.iloc[n - k:]
```

**Current system:**
```python
split_index = int(n * 0.70)
k = n - split_index
visible_data     = df.iloc[:split_index]
hidden_real_data = df.iloc[split_index:]
```

* 70% shown to user and used for model training
* 30% hidden until cross-check is triggered
* `k` now scales with dataset size instead of being capped at 3
* Models never see hidden data during training

### Why?
To simulate real-world forecasting and prevent data leakage.

---

## 3️⃣ Model Selection System

Model selector lives in the **sidebar**.

```python
if model_choice.startswith("Linear"):
    ...
elif model_choice.startswith("Logistic"):
    ...
```

* **Linear Regression (Price Forecast)**
* **Logistic Regression (Trend Direction)**
* **ARIMA Time-Series** ← separate section, in progress

Session state resets on both stock change AND model change using a combined `last_selection` key.

---

## 4️⃣ Linear Regression Model

**Purpose:** Predict future stock prices

**File:** `ML/LinearRegressionModel.py`

Features:
* Manual implementation — no sklearn
* Slope-intercept formula
* Recursive forecasting for k days
* Price prediction table
* Forecast chart: known (blue) + predicted (orange dashed)
* Cross-check chart: known + predicted + actual overlay (green)
* Comparison table: Date, Predicted Price, Actual Price, Difference

```
class LinearRegressionModel
train_and_predict()
prepare_data()
```

---

## 5️⃣ Logistic Regression Model

**Purpose:** Predict stock direction (UP/DOWN)

**File:** `ML/LogisticRegressionModel.py`

Features:
* Manual gradient descent implementation
* Sigmoid activation with clipping for numerical stability:
  ```python
  z = np.clip(z, -500, 500)
  ```
* Input normalization
* Direction classification per day
* Trend table display
* Cross-check table includes Match column (✅/❌)
* Accuracy score displayed after cross-check

```
class LogisticRegressionManual
train_and_predict_direction()
prepare_data()
```

---

## 6️⃣ Forecast System

Workflow:
1. Train model on visible 70% data
2. Predict next k days (30% window)
3. Display forecast results (price table or trend table)
4. Show forecast chart

Forecast results persist in UI — not overwritten on rerender.

---

## 7️⃣ Cross-Check System

After forecasting:
* A separate "Cross-check" button appears
* Reveals hidden 30% real records
* Displays comparison:
  * Price vs predicted price (Linear) — overlay chart
  * Predicted trend vs actual trend (Logistic) — table with Match column
* Accuracy calculation for Logistic

**Important:** Forecast graph is NOT overwritten. Cross-check graph appears below it.

```python
accuracy = sum(
    p == a for p, a in zip(predicted_dirs, actual_dirs)
) / len(predicted_dirs)
```

---

## 8️⃣ Session State Management

Key state variables:

| Variable | Purpose |
|----------|---------|
| `forecast_done` | Whether forecast has been run |
| `crosscheck_done` | Whether cross-check has been triggered |
| `model_mode` | `"price"` or `"trend"` |
| `predicted_output` | Stored forecast results |
| `last_selection` | `"{stock}__{model}"` — resets state on change |
| `live_ticker` | CSV filename of live-searched stock |
| `live_ticker_label` | Ticker symbol string of live stock |
| `ticker_error` | Error message if fetch failed |

Prevents stale predictions when switching stocks or models.

---

## 9️⃣ yfinance Live Data Integration ✅

**Date implemented:** 2026-03-05

* **Search bar** added to the main page header (full width, below title)
* User types company name OR ticker symbol
* **Live suggestion dropdown** (`st.selectbox`) filters instantly as user types
* Selecting a suggestion **auto-triggers fetch** — no button press needed
* `STOCK_SUGGESTIONS` dict: 65 popular US + Indian NSE stocks
* Matches on both company name and ticker (case-insensitive)
* On selection → `yf.download(ticker, period="2y", auto_adjust=True)` fetches 2 years
* Downloaded CSV saved to `data/predownloaded_live_cache/<TICKER>.csv`
* Dot notation converted to underscore in filename: `INFY.NS` → `INFY_NS.csv`
* MultiIndex columns flattened after fetch
* Timezone stripped from Date column if present
* **Priority logic**: live search result takes priority over preset dropdown
* Sidebar badge: 🟢 Live or 🟡 Offline
* Invalid ticker → `❌` error shown, no crash
* "Search & Load →" button available as fallback for unlisted tickers

**Key functions:**

```python
fetch_and_save_ticker(ticker)   # download 2y data + save CSV
handle_search(ticker)           # validate + activate ticker
get_filtered_options(query)     # live suggestion filtering (max 6 + fallback)
label_to_ticker(label, raw)     # extract ticker from display label
reset_forecast_state()          # clear all forecast/crosscheck state
```

**`load_csv()` logic:**
```python
@st.cache_data
def load_csv(path, source):
    skip = [1] if source == "offline" else []
    df = pd.read_csv(path, skiprows=skip)
    df["Date"] = df["Date"].astype(str)
    return df
```

Live CSVs are clean (no extra header). Offline preset CSVs need `skiprows=[1]`.

Cache TTL set to 3600 seconds (1 hour) for live data.

---

## 🔟 UI/UX Improvements

* Search bar in header with live filtering selectbox suggestion dropdown
* Sidebar: preset stock selector + model selector + data source badge
* Chart axis optimization:
  * X-axis: date labels sampled with step interval to prevent overlap
  * Y-axis: `MaxNLocator(nbins=8)` to prevent crowding
  * `plt.tight_layout()` on all charts
* Known data expander: `📄 View Raw Known Data`
* Forecasting window displayed: `**Forecasting window:** {k} days (30% of dataset)`
* Clean section separation with `st.markdown("---")`

---

# 🧮 ML Concepts Applied

* **Linear Regression** — Predict next-day stock price using slope-intercept relationship
* **Logistic Regression** — Binary classification predicting stock direction
* **Gradient Descent** — Optimize logistic regression parameters
* **Sigmoid Function** — Convert linear output to probability
* **Sigmoid Clipping** — Numerical stability (`np.clip(z, -500, 500)`)
* **Input Normalization** — Stabilize logistic regression training
* **Recursive Forecasting** — Use predicted value as next input
* **Data Leakage Prevention** — 70/30 split, models never see test data
* **Walk-forward validation** — Compare predictions with future unseen data
* **ARIMA Time-Series** ← in progress (AIC-based order selection)

---

# 🔐 Engineering Practices Followed

* Separation of concerns — ML logic in `ML/` folder, UI in `app.py`
* Manual ML model implementations for educational transparency
* Modular folder structure
* Session state management for persistent, stateful UI
* Offline dataset caching via `@st.cache_data`
* Defensive visualization — no chart overwriting
* Debug print traces throughout all runnable code
* Graceful error handling — `st.stop()` on data load failure

---

# 🐛 Bugs & Fixes Log

| Date | Bug | Fix | File |
|------|-----|-----|------|
| 2026-03-05 | `FileNotFoundError: data/mock_stock_data.csv` | Updated loader to use `predownloaded_live_cache/` | `app.py` |
| 2026-03-05 | `KeyError: 'Company'` after removing multi-company dataset | Removed company filtering; use full CSV | `app.py` |
| 2026-03-05 | Matplotlib X-axis labels overlapping | Sampled dates using step interval | `app.py` |
| 2026-03-05 | Y-axis labels overcrowded | `MaxNLocator(nbins=8)` | `app.py` |
| 2026-03-05 | `NameError: MaxNLocator not defined` | Added `from matplotlib.ticker import MaxNLocator` | `app.py` |
| 2026-03-05 | CSV contained duplicate ticker header rows | `skiprows=[1]` for offline CSVs only | `app.py` |
| 2026-03-05 | `model_choice` NameError crash on forecast | Added model selector to sidebar | `app.py` |
| 2026-03-05 | yfinance returns MultiIndex columns | `raw.columns = raw.columns.get_level_values(0)` | `app.py` |
| 2026-03-05 | yfinance Date column has timezone info | `dt.tz_localize(None)` | `app.py` |

---

# 🚀 Roadmap Status

| # | Feature | Status |
|---|---------|--------|
| 1 | Yahoo Finance live data integration | ✅ Done |
| 2 | Model performance comparison dashboard | ⏳ Pending |
| 3 | ARIMA Time-Series Model | 🔄 In Progress |
| 4 | Model performance leaderboard | ⏳ Pending |
| 5 | Ensemble model (bonus, low priority, not in PPT) | ⏳ Pending |

---

# 🎯 Current Status

✔ Offline stock dataset integration
✔ Multi-stock selection (preset + live search)
✔ yfinance live data fetch + local save
✔ Live suggestion dropdown in header search bar
✔ 70/30 train-test split
✔ Linear regression price forecasting
✔ Logistic regression trend prediction
✔ Forecast visualization
✔ Cross-check validation system
✔ Trend accuracy calculation
✔ Axis optimization for charts
✔ Session state management
✔ ARIMAModel.py skeleton started

🔄 ARIMA wiring into app.py — next session
⚠️ Model comparison dashboard — pending
⚠️ Confusion matrix visualization — pending
❌ Model leaderboard — not implemented

---

# 📅 Session Timeline

* [2026-03-05] Session 1 — Initial build: mock data, Linear + Logistic models, forecast + cross-check, session state
* [2026-03-05] Session 2 — Replaced mock data with offline yfinance CSVs; fixed CSV bugs; axis optimization
* [2026-03-05] Session 3 — Replaced `k` system with 70/30 split; added model selector to sidebar; fixed `model_choice` crash
* [2026-03-05] Session 4 — yfinance live integration; header search bar; live suggestion dropdown with auto-trigger on select; fetch + save to local cache; live/offline priority logic; `ARIMAModel.py` AIC grid search written

---

# 📝 This file will be updated as we continue to implement new features and improvements. Stay tuned for more updates!