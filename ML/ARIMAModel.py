import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# ARIMAModel.py
# ML/ARIMAModel.py
# --------------------------------------------------

def find_best_arima_order(series):
    """
    Grid search over (p, d, q) combinations.
    Picks the order with the lowest AIC score.
    p: AR terms (0-3)
    d: differencing (0-2)
    q: MA terms (0-3)
    """
    print(f"[DEBUG] ARIMA grid search starting | series length: {len(series)}")

    best_aic   = np.inf
    best_order = (1, 1, 1)

    for p in range(0, 4):
        for d in range(0, 3):
            for q in range(0, 4):
                try:
                    model = ARIMA(series, order=(p, d, q))
                    result = model.fit()
                    aic = result.aic
                    print(f"[DEBUG] ARIMA({p},{d},{q}) AIC={aic:.2f}")
                    if aic < best_aic:
                        best_aic   = aic
                        best_order = (p, d, q)
                except Exception as e:
                    print(f"[DEBUG] ARIMA({p},{d},{q}) failed: {e}")
                    continue

    print(f"[DEBUG] Best order: {best_order} | AIC: {best_aic:.2f}")
    return best_order


def train_and_predict_arima(visible_data, days):
    """
    Walk-forward ARIMA forecast.

    Instead of predicting all `days` at once (which produces a flat line),
    we predict 1 day at a time and append that prediction to the history
    before predicting the next day. This produces a dynamic, curvy output.

    Args:
        visible_data (pd.DataFrame): Training data with 'Close' column
        days (int): Number of days to forecast

    Returns:
        list: Forecasted price values (length = days)
    """
    print(f"[DEBUG] ARIMA walk-forward start | rows: {len(visible_data)} | forecast days: {days}")

    series = list(visible_data["Close"].astype(float).values)
    print(f"[DEBUG] Series head: {series[:5]} | tail: {series[-5:]}")

    # Find best (p,d,q) order using full visible series
    best_order = find_best_arima_order(series)
    print(f"[DEBUG] Using order {best_order} for walk-forward loop")

    forecast_list = []

    # Walk-forward: predict 1 step, append to history, repeat
    for i in range(days):
        try:
            model  = ARIMA(series, order=best_order)
            result = model.fit()
            next_pred = result.forecast(steps=1)[0]
            forecast_list.append(float(next_pred))

            # Append prediction to history so next step sees it
            series.append(float(next_pred))

            print(f"[DEBUG] Day +{i+1} predicted: {next_pred:.4f} | history length now: {len(series)}")

        except Exception as e:
            # If a step fails, carry forward the last value
            fallback = forecast_list[-1] if forecast_list else series[-1]
            forecast_list.append(fallback)
            series.append(fallback)
            print(f"[DEBUG] Day +{i+1} FAILED ({e}) | using fallback: {fallback:.4f}")

    print(f"[DEBUG] Walk-forward complete | total predicted: {len(forecast_list)}")
    print(f"[DEBUG] Forecast (first 5): {forecast_list[:5]}")
    print(f"[DEBUG] Forecast (last 5):  {forecast_list[-5:]}")

    return forecast_list