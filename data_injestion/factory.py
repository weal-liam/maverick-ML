from .csv_injestor import CSVInjestor
from .json_injestor import JSONInjestor
from .zipped_data_injestor import ZippedDataInjestor

class InjestionFactory():
    @staticmethod
    def set_injestor(injestor_type: str):
        if injestor_type == "csv":
            return CSVInjestor()
        elif injestor_type == "json":
            return JSONInjestor()
        elif injestor_type == "zip":
            return ZippedDataInjestor()
        else:
            raise ValueError(f"Unknown injestor type: {injestor_type}")