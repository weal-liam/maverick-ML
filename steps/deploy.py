from zenml import step
import bentoml
import logging

logging.basicConfig(level=logging.INFO)

@step
def deploy_model(deploy: bool) -> None:
    if deploy:
        model_ref = bentoml.sklearn.get("mav-pred:latest").to_runner()

        logging.info("Deploying the model...")
        #Deploy here
        logging.info(f"Service is up!!")
        return 
    return None

