from zenml import step
from typing import Dict, Hashable, Any
import bentoml
import pandas as pd
import logging

@step
def predict(service:str, data: dict) -> Dict[Hashable, Any]:
    logging.info("Making predictions...")
    
    @bentoml.service(name=f"{service}-service")
    class PredService():
        def __init__(self):
            self.model = bentoml.sklearn.get(service).to_runner()
            self.model.init_local()

        @bentoml.api
        def predict(self, input) -> Dict[Hashable, Any]:
            input_data = input["data"]
            cols = input["cols"]
            input_df = pd.DataFrame(input_data, columns=cols)
            return {"predictions": self.model.run(input_df).tolist()}

    predictions = PredService().predict(data)["predictions"]

    logging.info("Predictions made successfully.") 
    logging.info(f"Predictions: {predictions}") 
    return {"predictions": predictions}


