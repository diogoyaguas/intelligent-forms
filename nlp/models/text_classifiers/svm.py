"""SVM classifier module."""

import numpy as np
from sklearn.svm import LinearSVC


class SVMClassifier(LinearSVC):
    """Sklearn SVM classifier."""

    def predict_proba(self, x):
        """Predict sample with a associated probability.

        Args:
            x: np.ndarray

        Returns: probs

        """
        f = np.vectorize(self.__platt_func)
        raw_predictions = self.decision_function(x)
        platt_predictions = f(raw_predictions)
        try:
            probs = platt_predictions / platt_predictions.sum(axis=1)[:, None]
        except np.AxisError:
            probs = np.asarray(list(map(lambda prob1: [1-prob1, prob1], platt_predictions)))
        return probs

    @staticmethod
    def __platt_func(x):
        return 1/(1+np.exp(-x))
