from abc import ABC, abstractmethod
import pandas as pd


class DataInjestor(ABC):
    @abstractmethod
    def ingest_data(self, source: str) -> pd.DataFrame:
        pass