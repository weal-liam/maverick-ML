from sklearn.model_selection import GridSearchCV
from zenml import step, log_metadata, ArtifactConfig
from zenml.enums import ArtifactType
from typing import Annotated, Tuple
from .mode_choice import ModelSelector
from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline
import logging
import mlflow, bentoml

def get_model(required_model_type: str) -> RegressorMixin:
    logging.info(f"Using {required_model_type} model")

    model_selector = ModelSelector(required_model_type)
    model = model_selector.get_model()
    
    logging.info("Obtained model successfully!!")
    return model
    
@step
def train_model(model: GridSearchCV, data, target) -> Tuple[Annotated[Pipeline, ArtifactConfig(artifact_type=ArtifactType.MODEL)], str]:
    mlflow.sklearn.autolog()
    
    model.fit(data, target)
    model = model.best_estimator_
    model_info = mlflow.sklearn.log_model(model, artifact_path="model")

    logging.info(f"Model r2 at {model.score(data, target)}")

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