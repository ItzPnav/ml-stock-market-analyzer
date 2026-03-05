# 📈 **ml-stock-market-forecaster**

### *ML-Based Stock Forecasting System with Live Data, ARIMA & Model Performance Dashboard*

<div align="center">
<img src="https://img.shields.io/badge/Tech-Machine%20Learning-blue?style=for-the-badge">
<img src="https://img.shields.io/badge/Data-yfinance%20Live-green?style=for-the-badge">
<img src="https://img.shields.io/badge/Models-Linear%20%7C%20Logistic%20%7C%20ARIMA-orange?style=for-the-badge">
<img src="https://img.shields.io/badge/Frontend-Streamlit-red?style=for-the-badge">
<img src="https://img.shields.io/badge/Language-Python-yellow?style=for-the-badge">
<img src="https://img.shields.io/badge/Data-Pandas%20%7C%20NumPy-purple?style=for-the-badge">
</div>

---

## 🚀 Live Demo
```bash
https://ml-stock-market-analyzer.streamlit.app
```

---

# 📌 **Overview**

**ml-stock-market-forecaster** is a machine learning–based stock forecasting system built with **Python + Streamlit**. It simulates real-world forecasting conditions by hiding future stock data during model training, then revealing it afterward for cross-check validation.

It uses:

* 📐 **Manual Linear Regression** — future price forecasting
* 🔀 **Manual Logistic Regression** — UP/DOWN trend direction prediction
* 📉 **ARIMA** — time-series forecasting
* 📡 **yfinance** — live stock data fetching
* 📊 **Matplotlib** — forecast and cross-check visualization
* 🖥️ **Streamlit** — interactive web UI

> Built as an educational AI-assisted decision support tool for academic ML review.

---

# 📚 **Documentation Quick Links**

👉 **[PROJECT_PROGRESS.md](PROJECT_PROGRESS.md)** — *Full feature history & implementation log*
👉 [FILE_STRUCT.md](FILE_STRUCT.md) — *Directory tree*
👉 [requirements.txt](requirements.txt) — *Dependencies*
👉 [CREDITS.md](CREDITS.md) — *Libraries & acknowledgements*

---

# 🧠 **Architecture**

```
Search Bar (Live Ticker) OR Preset Dropdown (Offline CSV)
                    ↓
     yfinance fetch + save CSV  OR  load predownloaded_live_cache/
                    ↓
         Data Loader (Streamlit cache)
                    ↓
              70/30 Data Split
     ┌──────────────────────────────┐
     │  Visible Data (70%)          │  → shown to user, used for training
     │  Hidden Data  (30%)          │  → hidden until cross-check
     └──────────────────────────────┘
                    ↓
           Model Selection (Sidebar)
     ┌──────────────────────────────┐
     │  📐 Linear Regression        │  → price forecast
     │  🔀 Logistic Regression      │  → UP/DOWN direction
     │  📉 ARIMA                    │  → time-series forecast
     └──────────────────────────────┘
                    ↓
          Forecast Generation
                    ↓
       Visualization (Matplotlib)
                    ↓
        Cross-check With Hidden Data
                    ↓
     Accuracy / Model Performance Table
```

---

# 🚀 **Features**

### 📡 Live yfinance Data Integration
Search any company name or ticker symbol. Live data fetched for 2 years and cached locally. Suggestion dropdown with 65+ US and Indian NSE stocks. 🟢 Live / 🟡 Offline badge shown in sidebar.

### 📂 Offline Preset Stocks
Pre-downloaded CSV data for AAPL, NVDA, GLD, RELIANCE.NS, TCS.NS, INFY.NS — runs fully offline, no internet needed.

### 🔒 Data Leakage Prevention
70/30 train-test split — models **never** see future data during training. Hidden 30% is revealed only after the forecast is made, simulating real-world conditions.

### 📐 Linear Regression — Price Forecast
Manual slope-intercept implementation with no sklearn. Recursive multi-day price forecasting. Forecast chart (known + predicted) and cross-check overlay (predicted vs actual in green).

### 🔀 Logistic Regression — Trend Direction
Manual gradient descent with sigmoid activation. Predicts **UP 📈 / DOWN 📉** per day. Cross-check table with Match column (✅ / ❌) and a final accuracy score.

### 📉 ARIMA — Time-Series Forecasting
AIC-based order selection for optimal (p, d, q) parameters. Forecasts future stock prices using historical time-series patterns with a dedicated visualization section in the UI.

### 🏆 Model Performance Dashboard & Leaderboard
Compare Linear Regression, Logistic Regression, and ARIMA side by side. Displays accuracy scores, prediction error metrics, and a ranked leaderboard to evaluate which model performs best on the selected stock.

### 🔍 Cross-check Validation
After forecasting, reveal the hidden real data. Compare predicted vs actual with an overlay chart and a detailed comparison table showing exact differences.

---

# ⚙️ **Tech Stack**

| Layer           | Technology                                               |
|-----------------|----------------------------------------------------------|
| 🖥️ Frontend     | Streamlit                                                |
| 🧠 ML Models    | Manual Linear & Logistic Regression, ARIMA (statsmodels) |
| 🗃️ Data         | Pandas, NumPy                                            |
| 📊 Charts       | Matplotlib                                               |
| 📡 Live Data    | yfinance                                                 |
| 🐍 Language     | Python 3                                                 |

---

# 📦 **Setup (Short Version)**

1. **Clone the repo:**

```bash
git clone https://github.com/YOUR_USERNAME/ml-stock-market-forecaster.git
cd ml-stock-market-forecaster
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the app:**

```bash
streamlit run app.py
```

> 🟡 **Offline mode** works out of the box with pre-downloaded CSVs in `data/predownloaded_live_cache/`.
> 🟢 **Live mode** fetches data from Yahoo Finance automatically on ticker search.

---

# 🛡️ **Production Tips**

* 📥 Pre-download CSVs for all required tickers before demo — avoids live fetch delays
* ⏱️ Use `@st.cache_data` with TTL for live data to reduce repeated API calls
* 🌐 Run on a machine with stable internet for live yfinance fetching
* 💾 Keep `data/predownloaded_live_cache/` committed for fully offline execution

---

# 🤝 **Contributing**

PRs and issues welcome! Feel free to fork and build on top of this.

---

# 📜 **License**

MIT License — use freely. See [LICENSE.md](LICENSE.md) for details.

---

# ❤️ **Credits**

* 📈 Yahoo Finance & `yfinance` library for market data
* 📊 Matplotlib for visualization
* 🖥️ Streamlit for the interactive UI framework
* 🔢 NumPy & Pandas for data handling
* 📉 statsmodels for ARIMA time-series support

---

# 🚀 Made with passion by **Katakam Sri Pranav**

> *Stock forecasting done right.*