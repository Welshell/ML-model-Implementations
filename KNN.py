import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        # KNN doesn't "train" - it just remembers the data!
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict_one(x) for x in X]
        return np.array(predictions)

    def _predict_one(self, x):
        # Step 1: distance from x to every training point
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))

        # Step 2: get indices of k smallest distances
        k_indices = np.argsort(distances)[:self.k]

        # Step 3: get labels of those k neighbors
        k_labels = self.y_train[k_indices]

        # Step 4: return most common label
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]
