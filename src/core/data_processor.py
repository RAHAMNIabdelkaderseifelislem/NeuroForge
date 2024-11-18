from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class DataProcessor(ABC):
    @abstractmethod
    def process(self, data):
        pass

class PandasDataProcessor(DataProcessor):
    def process(self, data):
        return pd.DataFrame(data)