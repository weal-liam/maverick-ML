from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, f1_score
from sklearn.pipeline import Pipeline
from zenml import step
import logging

logging.basicConfig(level=logging.INFO)

@step
def evaluate_model(model : BaseEstimator | Pipeline, X_test, y_test) -> bool:
    should_deploy : bool = False
    predictions = model.predict(X_test)

    if isinstance(model, TransformedTargetRegressor):
        logging.info(f"R2 Score: {r2_score(y_test, predictions)}")
        if r2_score(y_test, predictions) < 0.8:
            logging.warning("The model is performing poorly")
        else:
            should_deploy = True
    else:
        logging.info(f"F1 Score: {f1_score(y_test, predictions)}")
        if f1_score(y_test, predictions) < 0.8:
            logging.warning("The model is performing poorly")
        else:
            should_deploy = True

    return should_deploy
