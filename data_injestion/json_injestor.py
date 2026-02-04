import pandas as pd
import json
from .injestor import DataInjestor
from .encoding import handle_file_encoding

class JSONInjestor(DataInjestor):
    def ingest_data(self, source: str) -> pd.DataFrame:
        # Validate JSON format
        with open(source, 'r') as file:
            content = file.read(1000)  # Read first 1000 characters to check format
            if not content.strip().startswith('{') and not content.strip().startswith('['):
                raise ValueError("Invalid JSON format")
            
        # Handle file encoding for the JSON file
        handle_file_encoding(source)

        # Load JSON data
        with open(source, 'r') as file:
            raw_data = json.load(file)

        return pd.json_normalize(raw_data, sep='_')