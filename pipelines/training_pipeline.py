from typing import Tuple
from zenml import pipeline
from data_analysis.cross_val import cross_validation_step
from data_models.model_train import get_model, train_model, register_model
from data_preprocessing.preprocess import preprocess_data
from steps.data_analysis_step import data_analysis_step
from steps.data_injestion_step import data_injestion_step
from data_model_evaluation.model_evaluate import evaluate_model

@pipeline(enable_cache=False)
def training_pipeline(model_name: str, source: str, target:str, desired_cols: list, undesired_cols: list):
    df = data_injestion_step(source=source)

    analysis_1, analysis_2, cat_cols, num_cols, df_new = data_analysis_step(df, target, desired_cols, undesired_cols)

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df_new, target, analysis_1,cat_cols, num_cols)

    model = cross_validation_step(analysis_1, preprocessor)

    trained_model, model_uri = train_model(model=model, data=X_train, target=y_train)

    evaluate_model(trained_model, X_test, y_test)

    register_model(model_uri,model_name)




