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

# 📌 **Overview**

**ml-stock-market-forecaster** is a machine learning–based stock forecasting system built with **Python + Streamlit**. It simulates real-world forecasting conditions by hiding future stock data during model training, then revealing it afterward for cross-check validation.

It uses:

* **Manual Linear Regression** — future price forecasting
* **Manual Logistic Regression** — UP/DOWN trend direction prediction
* **ARIMA** — time-series forecasting
* **yfinance** — live stock data fetching
* **Matplotlib** — forecast and cross-check visualization
* **Streamlit** — interactive web UI

Built as an educational AI-assisted decision support tool for academic ML review.

---

# 📚 **Documentation Quick Links**

👉 **[PROJECT_PROGRESS.md](PROJECT_PROGRESS.md)** — *Full feature history & implementation log*
👉 [FILE_STRUCT.md](FILE_STRUCT.md) — *Directory tree*
👉 [requirements.txt](requirements.txt) — *Dependencies*

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
     │  Linear Regression           │  → price forecast
     │  Logistic Regression         │  → UP/DOWN direction
     │  ARIMA                       │  → time-series forecast
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
Search any company name or ticker symbol. Live data fetched for 2 years and cached locally. Suggestion dropdown with 65+ US and Indian NSE stocks.

### 📂 Offline Preset Stocks
Pre-downloaded CSV data for AAPL, NVDA, GLD, RELIANCE.NS, TCS.NS, INFY.NS — runs fully offline.

### 🔒 Data Leakage Prevention
70/30 train-test split — models never see future data during training. Hidden 30% revealed only after forecast is made.

### 📈 Linear Regression — Price Forecast
Manual slope-intercept implementation. Recursive multi-day forecasting. Forecast chart (known + predicted) and cross-check overlay (predicted vs actual).

### 📉 Logistic Regression — Trend Direction
Manual gradient descent with sigmoid activation. Predicts UP/DOWN per day. Cross-check table with Match column (✅/❌) and accuracy score.

### 📊 ARIMA — Time-Series Forecasting
AIC-based order selection. Separate forecasting section in the UI. *(In progress)*

### 🏆 Model Performance Dashboard & Leaderboard
Compare model accuracy side by side. Ranked leaderboard of model performance. *(Pending)*

### 🔍 Cross-check Validation
After forecasting, reveal hidden real data. Compare predicted vs actual with overlay chart and comparison table.

---

# ⚙️ **Tech Stack**

| Layer         | Technology                        |
|---------------|-----------------------------------|
| Frontend      | Streamlit                         |
| ML Models     | Manual Linear & Logistic Regression, ARIMA (statsmodels) |
| Data Handling | Pandas, NumPy                     |
| Visualization | Matplotlib                        |
| Live Data     | yfinance                          |
| Language      | Python 3                          |

---

# 📦 **Setup (Short Version)**

1. Clone the repo:

```bash
git clone https://github.com/YOUR_USERNAME/ml-stock-market-forecaster.git
cd ml-stock-market-forecaster
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

> Offline mode works out of the box with pre-downloaded CSVs in `data/predownloaded_live_cache/`.
> Live mode fetches data from Yahoo Finance automatically on ticker search.

---

# 🛡️ Production Tips

* Pre-download CSVs for all required tickers before demo — avoids live fetch delays
* Use `@st.cache_data` with TTL for live data to reduce repeated API calls
* Run on a machine with stable internet for live yfinance fetching
* Keep `data/predownloaded_live_cache/` committed for fully offline execution

---

# 🤝 Contributing

PRs and issues welcome!

---

# 📜 License

MIT License — use freely.

---

# ❤️ Credits

* Yahoo Finance & yfinance library for market data
* Matplotlib for visualization
* Streamlit for the interactive UI
* NumPy & Pandas for data handling
* statsmodels for ARIMA

---

# 🚀 Made with passion by **Kattu**

Stock forecasting done right.