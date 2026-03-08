import numpy as np

class DataAnalyzer:

    def __init__(self, data: np.ndarray):
        self.data = data

    def correlation_matrix(self):
        """
        Compute correlation matrix between columns
        """
        correlation = np.corrcoef(self.data, rowvar=False)
        return correlation

    def detect_outlier(self, threshold=3):
        """
        Detect outliers using Z-score method
        """

        mean = np.mean(self.data, axis=0)
        std = np.std(self.data, axis=0)

        z_score = (self.data - mean) / std

        mask = np.abs(z_score) > threshold

        if not mask.any():
            print("No outliers detected")
            return None

        return np.where(mask)