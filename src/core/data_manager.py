import pandas as pd
from sklearn.model_selection import train_test_split

class DataManager:
    def load_data(self, file_path):
        return pd.read_csv(file_path)

    def preprocess_data(self, df):
        # Example preprocessing steps, can be expanded later
        return df.dropna()

    def split_data(self, df, target_column):
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return train_test_split(X, y, test_size=0.2)
