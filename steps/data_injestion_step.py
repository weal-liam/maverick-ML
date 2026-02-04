import pandas as pd
from data_injestion.factory import InjestionFactory
from zenml import step

@step
def data_injestion_step( source: str) -> pd.DataFrame:
    injestor_type = source.split('.')[-1]
    injestor = InjestionFactory.set_injestor(injestor_type)
    data = injestor.ingest_data(source)
    return data