import numpy as np 

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)   # start at 0
        self.bias = 0

        for _ in range(self.epochs):
            y_pred = sigmoid(X @ self.weights + self.bias)

            # gradients (from MSE loss)
            dw = (1/n_samples) * X.T @ (y_pred - y)
            db = (1/n_samples) * np.sum(y_pred - y)

            # update step
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        probs = sigmoid(X @ self.weights + self.bias)
        return (probs >= 0.5).astype(int)
