import copy
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
class LinearRegression():
    def __init__(self, lr: int = 0.01, n_iters: int = 2000) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape     
        self.weights = np.random.rand(num_features) 
        self.bias = 0

        for i in range(self.n_iters):

            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1 / num_samples) * np.dot(X.T, y_pred - y)
            db = (1 / num_samples) * np.sum(y_pred - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

        return self
    def r_squared(self, y,y_pred):
        y_mean = np.full((len(y)), np.mean(y))
        err_reg = sum((y - y_pred)**2)
        err_y_mean = sum((y - y_mean)**2)
        return (1 - (err_reg/err_y_mean))
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    