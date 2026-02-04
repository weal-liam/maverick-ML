from zenml import pipeline
from steps.predict import predict
import logging

@pipeline(enable_cache=False)
def inference_pipeline(service, data):
    logging.info("Starting inference pipeline...")
    predict(service, data)
    logging.info("Inference pipeline finished.")