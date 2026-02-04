from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from zenml import step
import logging

logging.basicConfig(level=logging.INFO)

@step
def evaluate_model(model : Pipeline, X_test, y_test) -> bool:
    should_deploy : bool = False
    predictions = model.predict(X_test)

    logging.info(f"R2 Score: {r2_score(y_test, predictions)}")

    if r2_score(y_test, predictions) < 0.8:
        logging.warning("The model is performing poorly")
    else:
        should_deploy = True

    return should_deploy
