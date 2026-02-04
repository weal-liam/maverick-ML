import pandas as pd
from .injestor import DataInjestor
from .encoding import detect_encoding
from .encoding import to_utf8
from .encoding import handle_file_encoding
import zipfile

class ZippedDataInjestor(DataInjestor):
    def ingest_data(self, source: str) -> pd.DataFrame:
        # Implementation for ingesting zipped data
        with zipfile.ZipFile(source, 'r') as zip_ref:
            zip_ref.extractall(path='extracted_data')

        # Handle file encoding for the extracted file
        handle_file_encoding(f'extracted_data/{zip_ref.namelist()[0]}')

        return pd.read_csv(f'extracted_data/{zip_ref.namelist()[0]}')