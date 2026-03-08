import numpy as np

class DataScaler:

    def __init__(self, data: np.ndarray):
        self.data = data

    def min_max_scale(self) -> np.ndarray:
        """Return scaled data using min-max scaling"""

        min_value = np.min(self.data, axis=0)
        max_value = np.max(self.data, axis=0)

        denom = max_value - min_value
        denom[denom == 0] = 1

        scaled = (self.data - min_value) / denom
        return scaled

    def standard_scale(self) -> np.ndarray:
        """Return scaled data using standard scaling"""

        mean = self.data.mean(axis=0)
        std = self.data.std(axis=0)

        std[std == 0] = 1

        scaled = (self.data - mean) / std
        return scaled