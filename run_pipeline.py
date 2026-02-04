from pipelines.training_pipeline import training_pipeline
from pipelines.deployment_pipeline import deployment_pipeline
from pipelines.inference_pipeline import inference_pipeline
from zenml.client import Client

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
  #run = training_pipeline("mav-pred","/home/surface_laptop_3/projects/prices-predictor-system/maverick-predictor-system/data/seoul+bike+sharing+demand.zip","bike_count",["bike_count", "hour", "temp", "humidity", "wind", "visibility", "dew_pt_temp", "radiation", "rain", "snow","seasons", "holiday", "functioning_day"],["Date"])

  evaluation = Client().get_pipeline("training_pipeline").last_successful_run.steps["evaluate_model"].output.load()
  model_label = Client().get_pipeline("training_pipeline").last_successful_run.steps["register_model"].output.load()

  #deployment_pipeline(deploy=evaluation)

  inference_pipeline(service=model_label, data={
    "data": [[10, 15.7, 20, 0.5, 4000, 1.0, 10.0, 0.0, 0.0, "Summer", "Holiday", "Yes"]], 
    "cols" :["hour", "temp", "humidity", "wind", "visibility", "dew_pt_temp", "radiation", "rain", "snow","seasons", "holiday",
               "functioning_day"]})

