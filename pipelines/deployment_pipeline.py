from zenml import pipeline
from steps.deploy import deploy_model
import logging

@pipeline(enable_cache=False)
def deployment_pipeline(deploy: bool):
    logging.info("Starting deployment pipeline...")
    deploy_model(deploy=deploy)
    logging.info("Deployment pipeline finished.")