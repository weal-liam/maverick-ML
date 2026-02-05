from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from zenml import step, ArtifactConfig
from zenml.enums import ArtifactType
from typing import Annotated, Tuple, Union
from .mode_choice import ModelSelector
from sklearn.base import BaseEstimator
import logging
import mlflow, bentoml, pandas as pd

def get_model(required_model_type: str) -> BaseEstimator:
    logging.info(f"Using {required_model_type} model")

    model_selector = ModelSelector(required_model_type)
    model = model_selector.get_model()
    
    logging.info("Obtained model successfully!!")
    return model
    
@step
def train_model(model: GridSearchCV, data, target) -> Tuple[Annotated[BaseEstimator | Pipeline, ArtifactConfig(artifact_type=ArtifactType.MODEL)], str]:
    mlflow.sklearn.autolog()
    
    model.fit(data, target)
    
    results = model.cv_results_
    table_1 = pd.DataFrame(results, columns=list(results.keys()))

    model_attr = model.best_params_
    table_2 = pd.Series(model_attr)

    training_score = model.best_score_
    model = model.best_estimator_

    model_info = mlflow.sklearn.log_model(model, artifact_path="model")

    logging.info(f"Model performance(RMSE/F1): {training_score} \nBest parameters: \n {table_2} \n Results: {table_1}")

    model_uri = model_info.model_uri

    return model, model_uri

@step
def register_model(model_uri, model_name) -> str:
    logging.info("Registering model...")

    model = mlflow.sklearn.load_model(model_uri)

    model = bentoml.sklearn.save_model(
        name=model_name,
        model=model
    )
    
    model_label = model.info.name

    logging.info(f"Registered {model_label} model")
    return model_label