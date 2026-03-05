import numpy as np

class LinearRegressionModel:
    def __init__(self):
        self.m = 0
        self.b = 0
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        y_mean = np.mean(y)
        x_mean = np.mean(X)

        numerator = np.sum((X - x_mean) * (y - y_mean))
        denominator = np.sum((X - x_mean) ** 2)

        self.m = numerator / denominator
        self.b = y_mean - self.m * x_mean
    
    def predict(self, X):

        X = np.array(X)

        return self.m * X + self.b

def prepare_data(close_prices):
    X = close_prices[:-1]
    y = close_prices[1:]
    return X , y

def train_and_predict(df, days=3):
    close_prices = df["Close"].values

    X, y = prepare_data(close_prices)

    model = LinearRegressionModel()
    model.fit(X, y)


    predictions = []
    last_price = close_prices[-1]

    for _ in range(days):
        next_price = float(model.predict([last_price])[0])
        predictions.append(round(next_price, 3))
        last_price = next_price
    
    return predictions
