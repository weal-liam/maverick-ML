from pipelines.training_pipeline import training_pipeline
from pipelines.deployment_pipeline import deployment_pipeline
from pipelines.inference_pipeline import inference_pipeline
from zenml.client import Client

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
  # Run the training pipeline
  # The training pipeline will ingest the data, perform data analysis, preprocess the data, train a model, evaluate it and register the model in the model registry.
  #for bike count as target, ["bike_count","hour", "temp", "humidity", "wind", "visibility", "dew_pt_temp", "radiation", "rain", "snow","seasons", "holiday", "functioning_day"]
  #for radiation class, ["fLength", "fWidth", "fSize", "fConc", "fConcl", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"] 
  run = training_pipeline(model_name="model",source='file-path',target='',desired_cols=[], undesired_cols=None, preferred_cat=None)

  #evaluation = Client().get_pipeline("training_pipeline").last_successful_run.steps["evaluate_model"].output.load()
  #model_label = Client().get_pipeline("training_pipeline").last_successful_run.steps["register_model"].output.load()

  #deployment_pipeline(deploy=evaluation)

  #inference_pipeline(service=model_label, data={
  #  "data": [[10, 15.7, 20, 0.5, 4000, 1.0, 10.0, 0.0, 0.0, "Summer", "Holiday", "Yes"]], 
  #  "cols" :["hour", "temp", "humidity", "wind", "visibility", "dew_pt_temp", "radiation", "rain", "snow","seasons", "holiday",
  #             "functioning_day"]})

