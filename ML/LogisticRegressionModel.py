import numpy as np

class LogisticRegressionManual:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = 0.0
        self.b = 0.0

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        for _ in range(self.epochs):
            z = self.w * X + self.b
            y_pred = self.sigmoid(z)

            dw = np.mean((y_pred - y) * X)
            db = np.mean(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        X = np.array(X)
        probs = self.sigmoid(self.w * X + self.b)
        return (probs >= 0.5).astype(int)


def prepare_data(close_prices):
    close_prices = np.array(close_prices)

    X = []
    y = []

    mean = np.mean(close_prices)
    std = np.std(close_prices)

    close_prices = (close_prices - mean) / std

    for i in range(len(close_prices) - 1):
        X.append(close_prices[i])
        y.append(1 if close_prices[i + 1] > close_prices[i] else 0)
    

    return np.array(X), np.array(y)


def train_and_predict_direction(df, days):
    prices = df["Close"].values

    X, y = prepare_data(prices)

    model = LogisticRegressionManual()
    model.fit(X, y)

    predictions = []
    last_price = prices[-1]

    for _ in range(days):
        direction = model.predict([last_price])[0]
        predictions.append(direction)

        # simulate next price movement for chaining
        last_price = last_price * (1.01 if direction == 1 else 0.99)

    return predictions
