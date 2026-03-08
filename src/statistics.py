import numpy as np


class StatisticsEngine:

    def __init__(self, data: np.ndarray):
        self.data = data

    def mean(self):
        return np.mean(self.data, axis=0)

    def std(self):
        return np.std(self.data, axis=0)

    def variance(self):
        return np.var(self.data, axis=0)

    def minimum(self):
        return np.min(self.data, axis=0)

    def maximum(self):
        return np.max(self.data, axis=0)

    def median(self):
        return np.median(self.data, axis=0)

    def total_sum(self):
        return np.sum(self.data, axis=0)