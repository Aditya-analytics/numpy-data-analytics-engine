import os
import numpy as np


class DatasetLoader:

    def __init__(self, file_path: str):
        self.file_path = file_path

    def validate_file(self):
        """Check if dataset file exists"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset not found: {self.file_path}")

    def load_dataset(self):
        """Load CSV dataset and convert to NumPy array"""

        self.validate_file()

        try:
            data = np.genfromtxt(
                self.file_path,
                delimiter=",",
                skip_header=1
            )

            return data

        except Exception as e:
            raise RuntimeError(f"Error loading dataset: {e}")