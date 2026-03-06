import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import yfinance as yf

from ML.LinearRegressionModel import train_and_predict
from ML.LogisticRegressionModel import train_and_predict_direction
from ML.ARIMAModel import train_and_predict_arima
from ML.LSTMModel import train_and_predict_lstm

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Stock Trend Analyzer",
    layout="wide"
)

# --------------------------------------------------
# Constants
# --------------------------------------------------
CACHE_DIR = "data/predownloaded_live_cache"

PRESET_STOCKS = {
    "Apple (AAPL)":           {"ticker": "AAPL",        "csv": "AAPL.csv"},
    "Reliance (RELIANCE.NS)": {"ticker": "RELIANCE.NS", "csv": "RELIANCE_NS.csv"},
    "TCS (TCS.NS)":           {"ticker": "TCS.NS",      "csv": "TCS_NS.csv"},
    "Infosys (INFY.NS)":      {"ticker": "INFY.NS",     "csv": "INFY_NS.csv"},
    "NVIDIA (NVDA)":          {"ticker": "NVDA",        "csv": "NVDA.csv"},
    "Gold ETF (GLD)":         {"ticker": "GLD",         "csv": "GLD.csv"},
}

STOCK_SUGGESTIONS = {
    "Tesla":                  "TSLA",
    "Apple":                  "AAPL",
    "Microsoft":              "MSFT",
    "Google":                 "GOOGL",
    "Alphabet":               "GOOGL",
    "Amazon":                 "AMZN",
    "Meta":                   "META",
    "Facebook":               "META",
    "NVIDIA":                 "NVDA",
    "Netflix":                "NFLX",
    "AMD":                    "AMD",
    "Intel":                  "INTC",
    "Qualcomm":               "QCOM",
    "Broadcom":               "AVGO",
    "Texas Instruments":      "TXN",
    "Salesforce":             "CRM",
    "Oracle":                 "ORCL",
    "IBM":                    "IBM",
    "Cisco":                  "CSCO",
    "Adobe":                  "ADBE",
    "Uber":                   "UBER",
    "Lyft":                   "LYFT",
    "Airbnb":                 "ABNB",
    "Spotify":                "SPOT",
    "PayPal":                 "PYPL",
    "Visa":                   "V",
    "Mastercard":             "MA",
    "JPMorgan":               "JPM",
    "Goldman Sachs":          "GS",
    "Bank of America":        "BAC",
    "Berkshire Hathaway":     "BRK-B",
    "Johnson & Johnson":      "JNJ",
    "Pfizer":                 "PFE",
    "Moderna":                "MRNA",
    "ExxonMobil":             "XOM",
    "Chevron":                "CVX",
    "Boeing":                 "BA",
    "Caterpillar":            "CAT",
    "Walt Disney":            "DIS",
    "Coca-Cola":              "KO",
    "PepsiCo":                "PEP",
    "McDonald's":             "MCD",
    "Nike":                   "NKE",
    "Walmart":                "WMT",
    "Target":                 "TGT",
    "Gold ETF":               "GLD",
    "S&P 500 ETF":            "SPY",
    "Nasdaq ETF":             "QQQ",
    "Reliance Industries":    "RELIANCE.NS",
    "TCS":                    "TCS.NS",
    "Tata Consultancy":       "TCS.NS",
    "Infosys":                "INFY.NS",
    "Wipro":                  "WIPRO.NS",
    "HDFC Bank":              "HDFCBANK.NS",
    "ICICI Bank":             "ICICIBANK.NS",
    "State Bank of India":    "SBIN.NS",
    "Bajaj Finance":          "BAJFINANCE.NS",
    "Maruti Suzuki":          "MARUTI.NS",
    "Tata Motors":            "TATAMOTORS.NS",
    "HCL Technologies":       "HCLTECH.NS",
    "Axis Bank":              "AXISBANK.NS",
    "Kotak Mahindra":         "KOTAKBANK.NS",
    "Asian Paints":           "ASIANPAINT.NS",
    "Hindustan Unilever":     "HINDUNILVR.NS",
}

TICKER_TO_NAME  = {v: k for k, v in STOCK_SUGGESTIONS.items()}
NO_MATCH_LABEL  = "— No match, search as typed —"


def get_filtered_options(query):
    if not query:
        return []
    q = query.lower().strip()
    results = []
    for company, ticker in STOCK_SUGGESTIONS.items():
        if q in company.lower() or q in ticker.lower():
            results.append(f"{company}  ({ticker})")
    results.append(NO_MATCH_LABEL)
    return results[:8]


def label_to_ticker(label, raw_input):
    if label == NO_MATCH_LABEL:
        return raw_input.strip().upper()
    return label.split("(")[-1].replace(")", "").strip()


# --------------------------------------------------
# Session State Init
# --------------------------------------------------
for key, default in [
    ("live_ticker",              None),
    ("live_ticker_label",        None),
    ("ticker_error",             None),
    ("forecast_done",            False),
    ("crosscheck_done",          False),
    ("model_mode",               None),
    ("predicted_output",         None),
    ("last_selection",           None),
    ("arima_forecast_done",      False),
    ("arima_crosscheck_done",    False),
    ("arima_predicted_output",   None),
    ("arima_best_order",         None),
    ("arima_last_selection",     None),
    ("lstm_forecast_done",       False),
    ("lstm_crosscheck_done",     False),
    ("lstm_predicted_output",    None),
    ("lstm_last_selection",      None),
    ("selected_model",           "Linear Regression (Price Forecast)"),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def fetch_and_save_ticker(ticker):
    print(f"[DEBUG] Fetching: {ticker}")
    try:
        raw = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            return None, f"❌ **{ticker}** not found on Yahoo Finance. Check the symbol."
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
            print(f"[DEBUG] Flattened MultiIndex: {list(raw.columns)}")
        raw.reset_index(inplace=True)
        if hasattr(raw["Date"].dtype, "tz") and raw["Date"].dtype.tz is not None:
            raw["Date"] = raw["Date"].dt.tz_localize(None)
        raw["Date"] = raw["Date"].astype(str)
        raw = raw[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        raw.reset_index(drop=True, inplace=True)
        safe_name = ticker.replace(".", "_") + ".csv"
        save_path = os.path.join(CACHE_DIR, safe_name)
        raw.to_csv(save_path, index=False)
        print(f"[DEBUG] Saved: {save_path} | Rows: {len(raw)}")
        return safe_name, None
    except Exception as e:
        print(f"[DEBUG] fetch ERROR: {e}")
        return None, f"❌ Error fetching **{ticker}**: {str(e)}"


def reset_forecast_state():
    st.session_state.forecast_done    = False
    st.session_state.crosscheck_done  = False
    st.session_state.model_mode       = None
    st.session_state.predicted_output = None
    st.session_state.last_selection   = None


def handle_search(ticker):
    ticker = ticker.strip().upper()
    if not ticker:
        return
    print(f"[DEBUG] handle_search: {ticker}")
    with st.spinner(f"Fetching {ticker} from Yahoo Finance..."):
        csv_name, error = fetch_and_save_ticker(ticker)
    if error:
        st.session_state.ticker_error      = error
        st.session_state.live_ticker       = None
        st.session_state.live_ticker_label = None
    else:
        st.session_state.ticker_error      = None
        st.session_state.live_ticker       = csv_name
        st.session_state.live_ticker_label = ticker
        reset_forecast_state()


# ==================================================
# HEADER
# ==================================================
st.title("📈 Stock Trend Analyzer")
st.caption("Model-based Stock Forecasting System")

# ==================================================
# SEARCH BAR
# ==================================================
search_col, btn_col = st.columns([5, 1])

with search_col:
    search_input = st.text_input(
        label="stock_search",
        label_visibility="collapsed",
        placeholder="🔍  Search company or ticker  (e.g. Tesla, TSLA, RELIANCE.NS)",
        key="search_input_box"
    )

with btn_col:
    st.markdown("<div style='padding-top: 4px;'></div>", unsafe_allow_html=True)
    search_btn = st.button("Search & Load →", use_container_width=True)

filtered_options   = get_filtered_options(search_input)
selected_suggestion = None
if filtered_options:
    selected_suggestion = st.selectbox(
        label="suggestions_box",
        label_visibility="collapsed",
        options=filtered_options,
        index=0,
        key="suggestion_selectbox"
    )

if search_btn and search_input:
    if selected_suggestion and selected_suggestion != NO_MATCH_LABEL:
        ticker_to_fetch = label_to_ticker(selected_suggestion, search_input)
    else:
        ticker_to_fetch = label_to_ticker(NO_MATCH_LABEL, search_input)
    print(f"[DEBUG] Search btn | raw: {search_input} | resolved: {ticker_to_fetch}")
    handle_search(ticker_to_fetch)
    st.rerun()

if st.session_state.ticker_error:
    st.error(st.session_state.ticker_error)
if st.session_state.live_ticker:
    st.success(f"🟢 Live data loaded: **{st.session_state.live_ticker_label}**")

st.markdown("---")

# ==================================================
# MODEL CARD SELECTOR — 4 cards
# ==================================================
st.markdown("#### 🧠 Select Forecasting Model")

MODELS = [
    {
        "key":   "Linear Regression (Price Forecast)",
        "icon":  "📈",
        "title": "Linear Regression",
        "desc":  "Predicts future closing prices using slope-intercept trend.",
        "tag":   "Price Forecast",
        "color": "#1f77b4",
    },
    {
        "key":   "Logistic Regression (Trend Direction)",
        "icon":  "🔀",
        "title": "Logistic Regression",
        "desc":  "Classifies each future day as UP or DOWN using gradient descent.",
        "tag":   "Trend Direction",
        "color": "#2ca02c",
    },
    {
        "key":   "ARIMA (Time-Series Forecast)",
        "icon":  "🔮",
        "title": "ARIMA",
        "desc":  "Time-series model using walk-forward forecasting with AIC order selection.",
        "tag":   "Time-Series",
        "color": "#9467bd",
    },
    {
        "key":   "LSTM (Deep Learning Forecast)",
        "icon":  "🧠",
        "title": "LSTM",
        "desc":  "Deep learning model using 60-day memory sequences to forecast future prices.",
        "tag":   "Deep Learning",
        "color": "#d62728",
    },
]

card_cols = st.columns(4)
for i, m in enumerate(MODELS):
    with card_cols[i]:
        is_selected  = st.session_state.selected_model == m["key"]
        border_color = m["color"] if is_selected else "#444"
        bg_color     = "#16213e" if is_selected else "#1a1a2e"
        shadow       = f'0 0 14px {m["color"]}88' if is_selected else "none"
        checkmark    = " ✅" if is_selected else ""

        st.markdown(f"""
        <div style="
            padding: 16px 18px;
            border-radius: 12px;
            border: 2px solid {border_color};
            background: {bg_color};
            box-shadow: {shadow};
            margin-bottom: 4px;
        ">
            <div style="font-size:28px">{m['icon']}</div>
            <div style="font-size:16px; font-weight:700; color:#fff;">{m['title']}{checkmark}</div>
            <div style="font-size:11px; font-weight:600; color:{m['color']}; text-transform:uppercase; margin:4px 0 8px;">{m['tag']}</div>
            <div style="font-size:13px; color:#bbb; line-height:1.4;">{m['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Select", key=f"model_btn_{i}", use_container_width=True):
            print(f"[DEBUG] Model card selected: {m['key']}")
            st.session_state.selected_model = m["key"]
            reset_forecast_state()
            st.rerun()

model_choice = st.session_state.selected_model
print(f"[DEBUG] Active model_choice: {model_choice}")

st.markdown("---")

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.header("Configuration")
st.sidebar.subheader("📂 Preset Stocks (Offline)")

selected_preset = st.sidebar.selectbox(
    "Choose Stock",
    list(PRESET_STOCKS.keys())
)

st.sidebar.markdown("---")

print(f"[DEBUG] Model: {st.session_state.selected_model} | Preset: {selected_preset}")

# ==================================================
# Determine Active Stock
# ==================================================
if st.session_state.live_ticker:
    active_label = st.session_state.live_ticker_label
    active_csv   = os.path.join(CACHE_DIR, st.session_state.live_ticker)
    data_source  = "live"
else:
    active_label = selected_preset
    active_csv   = os.path.join(CACHE_DIR, PRESET_STOCKS[selected_preset]["csv"])
    data_source  = "offline"

print(f"[DEBUG] Active: {active_label} | Source: {data_source}")

# ==================================================
# Load CSV
# ==================================================
@st.cache_data
def load_csv(path, source):
    skip = [1] if source == "offline" else []
    df = pd.read_csv(path, skiprows=skip)
    df["Date"] = df["Date"].astype(str)
    print(f"[DEBUG] Loaded: {path} | Rows: {len(df)}")
    return df

try:
    df = load_csv(active_csv, data_source)
except Exception as e:
    st.error(f"❌ Failed to load data for **{active_label}**: {e}")
    print(f"[DEBUG] load_csv ERROR: {e}")
    st.stop()

if data_source == "live":
    st.sidebar.success(f"🟢 Live: {active_label}")
else:
    st.sidebar.info(f"🟡 Offline: {active_label}")

# ==================================================
# 70/30 Split
# ==================================================
n           = len(df)
split_index = int(n * 0.70)
k           = n - split_index

visible_data     = df.iloc[:split_index].reset_index(drop=True)
hidden_real_data = df.iloc[split_index:].reset_index(drop=True)

print(f"[DEBUG] n={n} | train={split_index} | test={k}")

# ==================================================
# Session State — Reset on stock/model change
# ==================================================
current_selection = f"{active_label}__{model_choice}"

if st.session_state.last_selection != current_selection:
    print(f"[DEBUG] Selection changed → reset")
    reset_forecast_state()
    st.session_state.last_selection = current_selection

# ==================================================
# Section 1 — Known Data Chart (70%)
# ==================================================
st.subheader(f"📊 {active_label} — Known Data (70%)")

fig0, ax0 = plt.subplots(figsize=(12, 4))
x0 = range(len(visible_data))
ax0.plot(x0, visible_data["Close"], marker="o", markersize=3, label="Known Data (70%)", color="steelblue")

step = max(1, len(visible_data) // 10)
ax0.set_xticks(list(x0)[::step])
ax0.set_xticklabels(visible_data["Date"].iloc[::step], rotation=45, ha="right")
ax0.yaxis.set_major_locator(MaxNLocator(nbins=8))
ax0.set_title(f"{active_label} — Training Window")
ax0.set_xlabel("Date")
ax0.set_ylabel("Closing Price")
ax0.legend()
ax0.grid(True)
plt.tight_layout()
st.pyplot(fig0)

with st.expander("📄 View Raw Known Data"):
    st.dataframe(visible_data)

st.markdown(f"**Forecasting window:** {k} days (30% of dataset)")
st.markdown("---")

# ==================================================
# Section 2 — Forecast Button (Linear + Logistic)
# ==================================================
if not model_choice.startswith("ARIMA") and not model_choice.startswith("LSTM"):
    if st.button("🤖 Run Forecast"):
        print(f"[DEBUG] Forecast clicked | model: {model_choice} | k: {k}")
        st.session_state.crosscheck_done = False

        if model_choice.startswith("Linear"):
            future_prices = train_and_predict(visible_data, days=k)
            print(f"[DEBUG] Linear (first 5): {future_prices[:5]}")
            st.session_state.model_mode       = "price"
            st.session_state.predicted_output = future_prices
            st.session_state.forecast_done    = True
            st.success(f"✅ Linear Regression forecast complete — {k} days predicted.")

        elif model_choice.startswith("Logistic"):
            directions = train_and_predict_direction(visible_data, days=k)
            print(f"[DEBUG] Logistic (first 5): {directions[:5]}")
            st.session_state.model_mode       = "trend"
            st.session_state.predicted_output = directions
            st.session_state.forecast_done    = True
            st.success(f"✅ Logistic Regression prediction complete — {k} days predicted.")

# ==================================================
# Section 3 — Forecast Result (Persistent)
# ==================================================
if st.session_state.forecast_done:

    st.subheader("📈 Forecast Result (Predicted 30%)")
    future_labels = [f"Day +{i+1}" for i in range(k)]

    if st.session_state.model_mode == "price":
        future_prices = st.session_state.predicted_output

        price_df = pd.DataFrame({
            "Day":             future_labels,
            "Predicted Price": [round(p, 4) for p in future_prices]
        })
        with st.expander("📄 View Price Forecast Table"):
            st.dataframe(price_df)

        fig1, ax1 = plt.subplots(figsize=(12, 4))
        x_known    = range(len(visible_data))
        x_forecast = range(len(visible_data), len(visible_data) + k)
        ax1.plot(x_known,    visible_data["Close"], label="Known Data (70%)", color="steelblue", marker="o", markersize=3)
        ax1.plot(x_forecast, future_prices,         label="Predicted (30%)",  color="orange",    linestyle="--", marker="o", markersize=4)

        all_x      = list(x_known) + list(x_forecast)
        all_labels = list(visible_data["Date"]) + future_labels
        step1 = max(1, len(all_x) // 12)
        ax1.set_xticks(all_x[::step1])
        ax1.set_xticklabels(all_labels[::step1], rotation=45, ha="right")
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=8))
        ax1.set_title("Predicted Future Prices")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Price")
        ax1.legend()
        ax1.grid(True)
        plt.tight_layout()
        st.pyplot(fig1)

    elif st.session_state.model_mode == "trend":
        directions = st.session_state.predicted_output
        trend_df = pd.DataFrame({
            "Day":             future_labels,
            "Predicted Trend": ["UP 📈" if d == 1 else "DOWN 📉" for d in directions]
        })
        with st.expander("📄 View Trend Forecast Table"):
            st.dataframe(trend_df)

    st.markdown("---")
    if st.button("🔍 Cross-check with Real Data"):
        print(f"[DEBUG] Cross-check clicked")
        st.session_state.crosscheck_done = True

# ==================================================
# Section 5 — Cross-check Result
# ==================================================
if st.session_state.crosscheck_done:

    st.subheader("📊 Cross-check — Predicted vs Actual (30%)")

    if st.session_state.model_mode == "price":
        predicted_prices = st.session_state.predicted_output
        real_prices      = hidden_real_data["Close"].values
        real_dates       = hidden_real_data["Date"].values

        print(f"[DEBUG] CC predicted (first 5): {predicted_prices[:5]}")
        print(f"[DEBUG] CC actual    (first 5): {list(real_prices[:5])}")

        fig2, ax2 = plt.subplots(figsize=(12, 4))
        x_known = range(len(visible_data))
        x_pred  = range(len(visible_data), len(visible_data) + k)
        ax2.plot(x_known, visible_data["Close"], label="Known (70%)",        color="steelblue", marker="o",  markersize=3)
        ax2.plot(x_pred,  predicted_prices,      label="Predicted",          color="orange",    linestyle="--", marker="o", markersize=4)
        ax2.plot(x_pred,  real_prices,           label="Actual (Real 30%)",  color="green",     linestyle="-",  marker="x", markersize=5)

        all_x      = list(x_known) + list(x_pred)
        all_labels = list(visible_data["Date"]) + list(real_dates)
        step2 = max(1, len(all_x) // 12)
        ax2.set_xticks(all_x[::step2])
        ax2.set_xticklabels(all_labels[::step2], rotation=45, ha="right")
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=8))
        ax2.set_title("Predicted vs Actual — 30% Window")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Price")
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        st.pyplot(fig2)

        compare_df = pd.DataFrame({
            "Date":            real_dates,
            "Predicted Price": [round(p, 4) for p in predicted_prices],
            "Actual Price":    [round(p, 4) for p in real_prices],
            "Difference":      [round(abs(p - a), 4) for p, a in zip(predicted_prices, real_prices)]
        })
        with st.expander("📄 View Price Cross-check Table"):
            st.dataframe(compare_df)

    elif st.session_state.model_mode == "trend":
        predicted_dirs = st.session_state.predicted_output
        actual_dirs    = []
        prev_price     = visible_data["Close"].iloc[-1]

        for real_price in hidden_real_data["Close"].values:
            actual_dirs.append(1 if real_price > prev_price else 0)
            prev_price = real_price

        accuracy = sum(p == a for p, a in zip(predicted_dirs, actual_dirs)) / len(predicted_dirs)
        print(f"[DEBUG] Accuracy: {accuracy:.2%}")

        trend_compare_df = pd.DataFrame({
            "Day":             [f"Day +{i+1}" for i in range(k)],
            "Predicted Trend": ["UP 📈" if d == 1 else "DOWN 📉" for d in predicted_dirs],
            "Actual Trend":    ["UP 📈" if d == 1 else "DOWN 📉" for d in actual_dirs],
            "Match":           ["✅" if p == a else "❌" for p, a in zip(predicted_dirs, actual_dirs)]
        })

        with st.expander("📄 View Trend Cross-check Table"):
            st.dataframe(trend_compare_df)
        st.success(f"Trend Prediction Accuracy: {accuracy * 100:.2f}%")

# ==================================================
# ARIMA SECTION
# ==================================================
if model_choice.startswith("ARIMA"):
    st.subheader("🔮 ARIMA Time-Series Forecast")
    st.caption("AutoRegressive Integrated Moving Average — trained on the same 70% visible data")

    arima_selection = f"{active_label}__arima"
    if st.session_state.arima_last_selection != arima_selection:
        print(f"[DEBUG] ARIMA stock changed → reset ARIMA state")
        st.session_state.arima_forecast_done    = False
        st.session_state.arima_crosscheck_done  = False
        st.session_state.arima_predicted_output = None
        st.session_state.arima_best_order       = None
        st.session_state.arima_last_selection   = arima_selection

    if st.button("📡 Run ARIMA Forecast"):
        print(f"[DEBUG] ARIMA forecast clicked | k={k}")
        st.session_state.arima_crosscheck_done = False

        with st.spinner("Running ARIMA — finding best (p,d,q) order via AIC grid search..."):
            try:
                arima_prices = train_and_predict_arima(visible_data, days=k)
                st.session_state.arima_predicted_output = arima_prices
                st.session_state.arima_forecast_done    = True
                print(f"[DEBUG] ARIMA forecast done | first 5: {arima_prices[:5]}")
                st.success(f"✅ ARIMA forecast complete — {k} days predicted.")
            except Exception as e:
                print(f"[DEBUG] ARIMA ERROR: {e}")
                st.error(f"❌ ARIMA failed: {e}")

    if st.session_state.arima_forecast_done:

        arima_prices  = st.session_state.arima_predicted_output
        future_labels = [f"Day +{i+1}" for i in range(k)]

        st.subheader("📈 ARIMA Forecast Result (Predicted 30%)")

        arima_price_df = pd.DataFrame({
            "Day":             future_labels,
            "Predicted Price": [round(p, 4) for p in arima_prices]
        })
        with st.expander("📄 View ARIMA Forecast Table"):
            st.dataframe(arima_price_df)

        fig_a1, ax_a1 = plt.subplots(figsize=(12, 4))
        x_known    = range(len(visible_data))
        x_forecast = range(len(visible_data), len(visible_data) + k)

        ax_a1.plot(x_known,    visible_data["Close"], label="Known Data (70%)",      color="steelblue", linewidth=1.8)
        ax_a1.plot(x_forecast, arima_prices,          label="ARIMA Predicted (30%)", color="orange",    linestyle="--", linewidth=2)

        all_x      = list(x_known) + list(x_forecast)
        all_labels = list(visible_data["Date"]) + future_labels
        step_a1 = max(1, len(all_x) // 12)
        ax_a1.set_xticks(all_x[::step_a1])
        ax_a1.set_xticklabels(all_labels[::step_a1], rotation=45, ha="right")
        ax_a1.yaxis.set_major_locator(MaxNLocator(nbins=8))
        ax_a1.set_title(f"{active_label} — ARIMA Forecast")
        ax_a1.set_xlabel("Time")
        ax_a1.set_ylabel("Price")
        ax_a1.legend()
        ax_a1.grid(True)
        plt.tight_layout()
        st.pyplot(fig_a1)

        print(f"[DEBUG] ARIMA forecast chart rendered")

        st.markdown("---")
        if st.button("🔍 ARIMA Cross-check with Real Data"):
            print(f"[DEBUG] ARIMA cross-check clicked")
            st.session_state.arima_crosscheck_done = True

    if st.session_state.arima_crosscheck_done:

        arima_prices = st.session_state.arima_predicted_output
        real_prices  = hidden_real_data["Close"].values
        real_dates   = hidden_real_data["Date"].values

        st.subheader("📊 ARIMA Cross-check — Predicted vs Actual (30%)")

        fig_a2, ax_a2 = plt.subplots(figsize=(12, 4))
        x_known = range(len(visible_data))
        x_pred  = range(len(visible_data), len(visible_data) + k)

        ax_a2.plot(x_known, visible_data["Close"], label="Known (70%)",       color="steelblue", linewidth=1.8)
        ax_a2.plot(x_pred,  arima_prices,          label="ARIMA Predicted",   color="orange",    linestyle="--", linewidth=2)
        ax_a2.plot(x_pred,  real_prices,           label="Actual (Real 30%)", color="green",     linewidth=2)

        all_x      = list(x_known) + list(x_pred)
        all_labels = list(visible_data["Date"]) + list(real_dates)
        step_a2 = max(1, len(all_x) // 12)
        ax_a2.set_xticks(all_x[::step_a2])
        ax_a2.set_xticklabels(all_labels[::step_a2], rotation=45, ha="right")
        ax_a2.yaxis.set_major_locator(MaxNLocator(nbins=8))
        ax_a2.set_title(f"{active_label} — ARIMA Predicted vs Actual")
        ax_a2.set_xlabel("Time")
        ax_a2.set_ylabel("Price")
        ax_a2.legend()
        ax_a2.grid(True)
        plt.tight_layout()
        st.pyplot(fig_a2)

        min_len = min(len(arima_prices), len(real_prices))
        arima_compare_df = pd.DataFrame({
            "Date":            real_dates[:min_len],
            "ARIMA Predicted": [round(p, 4) for p in arima_prices[:min_len]],
            "Actual Price":    [round(p, 4) for p in real_prices[:min_len]],
            "Difference":      [round(abs(p - a), 4) for p, a in zip(arima_prices[:min_len], real_prices[:min_len])]
        })
        with st.expander("📄 View ARIMA Cross-check Table"):
            st.dataframe(arima_compare_df)

        mae = sum(abs(p - a) for p, a in zip(arima_prices[:min_len], real_prices[:min_len])) / min_len
        print(f"[DEBUG] ARIMA MAE: {mae:.4f}")
        st.info(f"📉 Mean Absolute Error (MAE): **{mae:.4f}**")

# ==================================================
# LSTM SECTION — Walk-Forward Deep Learning Forecast
# ==================================================
if model_choice.startswith("LSTM"):
    st.subheader("🧠 LSTM Deep Learning Forecast")
    st.caption("Long Short-Term Memory — 60-day lookback, walk-forward prediction on 30% window")

    lstm_selection = f"{active_label}__lstm"
    if st.session_state.lstm_last_selection != lstm_selection:
        print(f"[DEBUG] LSTM stock changed → reset LSTM state")
        st.session_state.lstm_forecast_done    = False
        st.session_state.lstm_crosscheck_done  = False
        st.session_state.lstm_predicted_output = None
        st.session_state.lstm_last_selection   = lstm_selection

    if st.button("🚀 Run LSTM Forecast"):
        print(f"[DEBUG] LSTM forecast clicked | k={k}")
        st.session_state.lstm_crosscheck_done = False

        with st.spinner("🧠 Training LSTM — this may take 30–60 seconds..."):
            try:
                # Walk-forward: pass hidden_real_data so real prices feed the window
                lstm_prices = train_and_predict_lstm(
                    visible_df     = visible_data,
                    days           = k,
                    hidden_real_df = hidden_real_data   # ← walk-forward mode
                )
                st.session_state.lstm_predicted_output = lstm_prices
                st.session_state.lstm_forecast_done    = True
                print(f"[DEBUG] LSTM forecast done | first 5: {lstm_prices[:5]}")
                st.success(f"✅ LSTM forecast complete — {k} days predicted.")
            except Exception as e:
                print(f"[DEBUG] LSTM ERROR: {e}")
                st.error(f"❌ LSTM failed: {e}")

    if st.session_state.lstm_forecast_done:

        lstm_prices   = st.session_state.lstm_predicted_output
        future_labels = [f"Day +{i+1}" for i in range(k)]

        st.subheader("📈 LSTM Forecast Result (Predicted 30%)")

        lstm_price_df = pd.DataFrame({
            "Day":             future_labels,
            "Predicted Price": [round(p, 4) for p in lstm_prices]
        })
        with st.expander("📄 View LSTM Forecast Table"):
            st.dataframe(lstm_price_df)

        fig_l1, ax_l1 = plt.subplots(figsize=(12, 4))
        x_known    = range(len(visible_data))
        x_forecast = range(len(visible_data), len(visible_data) + k)

        ax_l1.plot(x_known,    visible_data["Close"], label="Known Data (70%)",     color="steelblue", linewidth=1.8)
        ax_l1.plot(x_forecast, lstm_prices,           label="LSTM Predicted (30%)", color="#d62728",   linestyle="--", linewidth=2)

        all_x      = list(x_known) + list(x_forecast)
        all_labels = list(visible_data["Date"]) + future_labels
        step_l1 = max(1, len(all_x) // 12)
        ax_l1.set_xticks(all_x[::step_l1])
        ax_l1.set_xticklabels(all_labels[::step_l1], rotation=45, ha="right")
        ax_l1.yaxis.set_major_locator(MaxNLocator(nbins=8))
        ax_l1.set_title(f"{active_label} — LSTM Forecast")
        ax_l1.set_xlabel("Time")
        ax_l1.set_ylabel("Price")
        ax_l1.legend()
        ax_l1.grid(True)
        plt.tight_layout()
        st.pyplot(fig_l1)

        print(f"[DEBUG] LSTM forecast chart rendered")

        st.markdown("---")
        if st.button("🔍 LSTM Cross-check with Real Data"):
            print(f"[DEBUG] LSTM cross-check clicked")
            st.session_state.lstm_crosscheck_done = True

    if st.session_state.lstm_crosscheck_done:

        lstm_prices = st.session_state.lstm_predicted_output
        real_prices = hidden_real_data["Close"].values
        real_dates  = hidden_real_data["Date"].values

        print(f"[DEBUG] LSTM CC predicted (first 5): {lstm_prices[:5]}")
        print(f"[DEBUG] LSTM CC actual    (first 5): {list(real_prices[:5])}")

        st.subheader("📊 LSTM Cross-check — Predicted vs Actual (30%)")

        fig_l2, ax_l2 = plt.subplots(figsize=(12, 4))
        x_known = range(len(visible_data))
        x_pred  = range(len(visible_data), len(visible_data) + k)

        ax_l2.plot(x_known, visible_data["Close"], label="Known (70%)",       color="steelblue", linewidth=1.8)
        ax_l2.plot(x_pred,  lstm_prices,           label="LSTM Predicted",    color="#d62728",   linestyle="--", linewidth=2)
        ax_l2.plot(x_pred,  real_prices,           label="Actual (Real 30%)", color="green",     linewidth=2)

        all_x      = list(x_known) + list(x_pred)
        all_labels = list(visible_data["Date"]) + list(real_dates)
        step_l2 = max(1, len(all_x) // 12)
        ax_l2.set_xticks(all_x[::step_l2])
        ax_l2.set_xticklabels(all_labels[::step_l2], rotation=45, ha="right")
        ax_l2.yaxis.set_major_locator(MaxNLocator(nbins=8))
        ax_l2.set_title(f"{active_label} — LSTM Predicted vs Actual")
        ax_l2.set_xlabel("Time")
        ax_l2.set_ylabel("Price")
        ax_l2.legend()
        ax_l2.grid(True)
        plt.tight_layout()
        st.pyplot(fig_l2)

        print(f"[DEBUG] LSTM cross-check chart rendered")

        min_len = min(len(lstm_prices), len(real_prices))
        lstm_compare_df = pd.DataFrame({
            "Date":           real_dates[:min_len],
            "LSTM Predicted": [round(p, 4) for p in lstm_prices[:min_len]],
            "Actual Price":   [round(p, 4) for p in real_prices[:min_len]],
            "Difference":     [round(abs(p - a), 4) for p, a in zip(lstm_prices[:min_len], real_prices[:min_len])]
        })
        with st.expander("📄 View LSTM Cross-check Table"):
            st.dataframe(lstm_compare_df)

        mae = sum(abs(p - a) for p, a in zip(lstm_prices[:min_len], real_prices[:min_len])) / min_len
        print(f"[DEBUG] LSTM MAE: {mae:.4f}")
        st.info(f"📉 Mean Absolute Error (MAE): **{mae:.4f}**")