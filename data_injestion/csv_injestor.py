import pandas as pd
from .injestor import DataInjestor

class CSVInjestor(DataInjestor):
    def ingest_data(self, source: str) -> pd.DataFrame:
        return pd.read_csv(source)