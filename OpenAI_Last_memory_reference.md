# Memory Archive

> Extracted: 2026-03-05
> Source: Full conversation history

---

## Projects & Code Work

### Stock Trend Analyzer

* **Date first mentioned:** [date unknown]
* **Description:** A model-driven stock forecasting system that allows users to analyze stock price movements and predict future trends. The system hides future data points (`k` records) to simulate real-world forecasting and prevent data leakage. Users forecast future prices or directions using machine learning models and then cross-check predictions against the hidden real data.
* **Tech stack:** Python, Streamlit, pandas, numpy, matplotlib, yfinance
* **Current status:** Implemented with Linear Regression and Logistic Regression models, offline dataset support via pre-downloaded Yahoo Finance data, dynamic stock selection from sidebar, forecast visualization, and cross-check functionality.
* **Key decisions / bugs / fixes:**

  * Implemented **data leakage prevention** by splitting dataset into:

    * visible data (`n - k`)
    * hidden future data (`k`)
  * `k` automatically computed as:

    ```python
    k = min(3, n // 4) if n >= 4 else 1
    ```
  * Implemented **session state management**:

    * `forecast_done`
    * `crosscheck_done`
    * `model_mode`
    * `predicted_output`
  * Built **manual Linear Regression implementation** for price prediction using slope-intercept formula.
  * Built **manual Logistic Regression implementation** using gradient descent and sigmoid activation.
  * Logistic model predicts **direction (UP/DOWN)** rather than price.
  * Recursive prediction used for forecasting multiple days.
  * Cross-check system compares predicted values against hidden real data.
  * Trend prediction accuracy calculated by comparing predicted directions to actual movement.
  * Original dataset contained a `Company` column with multiple companies in a single CSV; later architecture switched to **one CSV per stock**, removing the `Company` column.
  * Encountered `KeyError: 'Company'` because the new CSV format did not include the column; resolved by replacing:

    ```python
    company_data = df[df["Company"] == selected_company]
    ```

    with:

    ```python
    company_data = df.reset_index(drop=True)
    ```
  * Encountered `FileNotFoundError: data/mock_stock_data.csv` after removing mock dataset; fixed by updating loader to read from:

    ```
    data/predownloaded_live_cache/
    ```
  * CSV files downloaded from yfinance contained an extra header row (`AAPL,AAPL,AAPL...`) due to multi-index columns; discussed fixes:

    * cleaning columns during download
    * or skipping the extra row during read.
  * X-axis labels overlapped due to large number of dates; solved by sampling date labels using a step size.
  * Y-axis label crowding solved by using:

    ```python
    from matplotlib.ticker import MaxNLocator
    ax0.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ```
  * Added `plt.tight_layout()` to avoid label clipping.
* **Files / components mentioned:**

  ```
  app.py
  ML/LinearRegressionModel.py
  ML/LogisticRegressionModel.py
  requirements.txt
  PROJECT_PROGRESS.md
  FILE_STRUCT.md
  data/predownloaded_live_cache/
  ```
* **Outstanding issues / TODOs:**

  * Model performance comparison dashboard
  * Confusion matrix visualization
  * ARIMA time-series model implementation
  * Model performance leaderboard
  * Historic data prediction using ARIMA

---

### Yahoo Finance Data Pre-download System

* **Date first mentioned:** [date unknown]
* **Description:** A system created to download historical stock market data from Yahoo Finance using the `yfinance` library and store it locally as CSV files so the Streamlit application can operate fully offline.
* **Tech stack:** Python, yfinance, pandas
* **Current status:** Implemented. Stock data downloaded and saved under `data/predownloaded_live_cache/`.
* **Key decisions / bugs / fixes:**

  * Initial misunderstanding where `.py` files were created per stock; corrected to store CSV files instead.
  * Created scripts to download stocks and save them as CSV files.
  * Data files created for:

    ```
    AAPL
    RELIANCE.NS
    TCS.NS
    INFY.NS
    NVDA
    GLD
    ```
  * Files saved as:

    ```
    AAPL.csv
    RELIANCE_NS.csv
    TCS_NS.csv
    INFY_NS.csv
    NVDA.csv
    GLD.csv
    ```
  * Folder structure:

    ```
    data/predownloaded_live_cache/
    ```
  * Converted date index to column using:

    ```python
    df.reset_index(inplace=True)
    ```
* **Files / components mentioned:**

  ```
  data/predownloaded_live_cache/AAPL.csv
  data/predownloaded_live_cache/GLD.csv
  data/predownloaded_live_cache/INFY_NS.csv
  data/predownloaded_live_cache/NVDA.csv
  data/predownloaded_live_cache/RELIANCE_NS.csv
  data/predownloaded_live_cache/TCS_NS.csv
  ```
* **Outstanding issues / TODOs:**

  * Optional hybrid system allowing both live and offline data modes.

---

## Instructions & Rules Given by User
* [2026-03-05] The project roadmap for the Stock Trend Analyzer must follow this exact order: 

  1. Yahoo Finance live data integration
  2. Model performance comparison dashboard
  3. Confusion matrix visualization
  4. ARIMA Time-Series Model
  5. Model performance leaderboard
     Bonus: Historic data prediction using ARIMA
* [2026-03-05] The ensemble model feature should be treated lightly because it was **not covered in the project presentation (PPT)**.
