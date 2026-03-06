import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------
# Constants
# --------------------------------------------------
LOOKBACK    = 60    # past days used to predict next day
EPOCHS      = 20    # training epochs
BATCH_SIZE  = 32    # samples per gradient update
HIDDEN_SIZE = 100   # LSTM hidden units per layer
NUM_LAYERS  = 2     # stacked LSTM layers
DROPOUT     = 0.2   # dropout rate between layers


# --------------------------------------------------
# PyTorch LSTM Model Definition
# --------------------------------------------------
class LSTMNet(nn.Module):
    """
    2-layer stacked LSTM followed by a fully connected output layer.
    Input shape  : (batch, sequence_length, 1)
    Output shape : (batch, 1)
    """
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super(LSTMNet, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,           # input: (batch, seq, feature)
            dropout=dropout             # dropout between stacked layers
        )
        self.fc = nn.Linear(hidden_size, 1)

        print(f"[DEBUG] LSTMNet built | hidden={hidden_size} | layers={num_layers} | dropout={dropout}")

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take output from last time step only
        last_out = lstm_out[:, -1, :]   # (batch, hidden_size)
        out = self.fc(last_out)          # (batch, 1)
        return out


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def prepare_sequences(prices_scaled, lookback=LOOKBACK):
    """
    Build (X, y) numpy arrays from scaled 1D price array.
    X[i] : prices[i : i+lookback]  → shape (lookback,)
    y[i] : prices[i+lookback]       → scalar
    """
    X, y = [], []
    for i in range(lookback, len(prices_scaled)):
        X.append(prices_scaled[i - lookback:i])
        y.append(prices_scaled[i])
    X = np.array(X, dtype=np.float32)   # (samples, lookback)
    y = np.array(y, dtype=np.float32)   # (samples,)
    print(f"[DEBUG] Sequences | X: {X.shape} | y: {y.shape}")
    return X, y


def create_batches(X, y, batch_size=BATCH_SIZE):
    """Yield (X_batch, y_batch) tensors of size batch_size."""
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        idx = indices[start:start + batch_size]
        X_batch = torch.tensor(X[idx]).unsqueeze(-1)   # (batch, seq, 1)
        y_batch = torch.tensor(y[idx]).unsqueeze(-1)   # (batch, 1)
        yield X_batch, y_batch


# --------------------------------------------------
# Main entry point
# --------------------------------------------------
def train_and_predict_lstm(visible_df, days, hidden_real_df=None):
    """
    Train LSTM on visible_df['Close'] and forecast `days` steps ahead.

    Walk-forward mode (when hidden_real_df is provided):
        At each step, the real price from hidden_real_df is fed into
        the sliding window — one real day at a time.
        This produces a wavy, realistic-looking prediction curve.

    Args:
        visible_df     : DataFrame with 'Close' column (70% training data)
        days           : number of future days to forecast (30% window)
        hidden_real_df : DataFrame with 'Close' column (30% real data)
                         If provided → walk-forward mode
                         If None     → recursive mode (pure future forecast)

    Returns:
        predicted_prices : list of float (inverse-transformed real prices)
    """
    print(f"[DEBUG] LSTM start | rows: {len(visible_df)} | forecast days: {days}")
    mode = "walk-forward" if hidden_real_df is not None else "recursive"
    print(f"[DEBUG] Forecast mode: {mode}")

    # ── 1. Extract and scale prices using visible data only ──────────────
    prices = visible_df["Close"].values.reshape(-1, 1).astype(np.float32)
    print(f"[DEBUG] Raw price range: {prices.min():.2f} → {prices.max():.2f}")

    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices).flatten()
    print(f"[DEBUG] Scaled range: {prices_scaled.min():.4f} → {prices_scaled.max():.4f}")

    # ── 2. Guard: need at least LOOKBACK + 1 rows ────────────────────────
    if len(prices_scaled) <= LOOKBACK:
        raise ValueError(
            f"Not enough data for LSTM. Need > {LOOKBACK} rows, got {len(prices_scaled)}."
        )

    # ── 3. Build sequences ───────────────────────────────────────────────
    X_train, y_train = prepare_sequences(prices_scaled, LOOKBACK)

    # ── 4. Build model + optimizer + loss ────────────────────────────────
    device    = torch.device("cpu")
    model     = LSTMNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ── 5. Training loop ─────────────────────────────────────────────────
    model.train()
    print(f"[DEBUG] Training | epochs={EPOCHS} | batch={BATCH_SIZE}")

    for epoch in range(EPOCHS):
        epoch_loss  = 0.0
        batch_count = 0

        for X_batch, y_batch in create_batches(X_train, y_train, BATCH_SIZE):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            loss   = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss  += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        if (epoch + 1) % 5 == 0:
            print(f"[DEBUG] Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f}")

    print(f"[DEBUG] Training complete")

    # ── 6. Forecast ───────────────────────────────────────────────────────
    model.eval()

    # Full price history available to the sliding window
    # In walk-forward mode: we append real hidden prices after each prediction
    # In recursive mode:    we append the predicted price after each prediction
    full_scaled = list(prices_scaled)

    if hidden_real_df is not None:
        # Scale hidden real prices using the SAME scaler fitted on visible data
        hidden_prices = hidden_real_df["Close"].values.reshape(-1, 1).astype(np.float32)
        hidden_scaled = scaler.transform(hidden_prices).flatten().tolist()
        print(f"[DEBUG] Hidden prices scaled | first 5: {hidden_scaled[:5]}")

    predicted_scaled = []

    with torch.no_grad():
        for step in range(days):
            # Always take last LOOKBACK points from full_scaled as input
            input_seq    = np.array(full_scaled[-LOOKBACK:], dtype=np.float32)
            input_tensor = torch.tensor(input_seq).unsqueeze(0).unsqueeze(-1).to(device)
            # shape: (1, LOOKBACK, 1)

            next_scaled = model(input_tensor).item()
            predicted_scaled.append(next_scaled)

            if hidden_real_df is not None and step < len(hidden_scaled):
                # Walk-forward: append the REAL next price to the window
                full_scaled.append(hidden_scaled[step])
            else:
                # Recursive fallback: append the predicted price
                full_scaled.append(next_scaled)

    print(f"[DEBUG] Forecast done | mode: {mode} | steps: {days}")
    print(f"[DEBUG] Predicted scaled (first 5): {predicted_scaled[:5]}")

    # ── 7. Inverse transform → real prices ──────────────────────────────
    predicted_array  = np.array(predicted_scaled, dtype=np.float32).reshape(-1, 1)
    predicted_prices = scaler.inverse_transform(predicted_array).flatten().tolist()

    print(f"[DEBUG] Predicted prices (first 5): {[round(p, 4) for p in predicted_prices[:5]]}")
    return predicted_prices