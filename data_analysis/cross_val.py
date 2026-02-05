import logging
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from zenml import step
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from data_models.model_params import model_params
from data_models.model_train import get_model

def inverse_func(y):
            #inverse the log-transformation
            y = np.expm1(y)
            
            #handle possible zero predictions
            y = np.maximum(0, y)

            #return a rounded off integer
            return np.round(y).astype(int)

@step
def cross_validation_step(analysis: dict, preprocessor: ColumnTransformer) -> GridSearchCV:
    logging.info("Initiating cross-validation step...")

    #Based on the analysis, determine the model to use and its corresponding parameter grid for tuning
    model_list = list(analysis['analysis']["problem"].values()) if len(list(analysis['analysis']["problem"].values())[0]) > 0 else None
    model = get_model(model_list[0][0]) if model_list else None  
    logging.info(f"Selected model: {model.__name__} based on analysis.")

    #multiple parameter grids for different models
    param_grid = model_params.get(f"{model.__name__}_param_grid", dict())
    logging.info(f"Using parameter grid: {param_grid} for model: {model.__name__}")

    #Create a pipeline with the preprocessor and the model
    pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", model())
    ])

    #set up estimator with pipeline as default
    estimator = pipe

    if 'Regressor' in str(model.__name__):
        #handle logarithmic transformation suggestion for target variable if any
        target_suggestion = analysis["analysis"].get("target_suggestion", None)
        
        #Handle forward and inverse finctions for the transformer
        fwd_func = np.log1p if target_suggestion == 'log-transform' else None
        logging.info("Applying logarithmic transformation to the target variable as suggested by analysis.") if fwd_func else None

        #Instantiate Transformer
        tt_regressor = TransformedTargetRegressor(regressor=pipe, func=fwd_func, inverse_func=inverse_func if target_suggestion == 'log-transform' else None)
        
        #Update the estimator
        estimator = tt_regressor

        param_grid = {f"regressor__{key}": value for key, value in param_grid.items()}
        logging.info(f"Updated parameter grid for TransformedTargetRegressor: {param_grid}")

    #Set up K-Fold cross-validation with n splits, shuffling, and a fixed random state for reproducibility
    kf = KFold(n_splits=6, shuffle=True, random_state=42)
    
    #Set up GridSearchCV with the pipeline, parameter grid, K-Fold cross-validation, and a relevant scoring metric
    search = GridSearchCV(
        estimator,
        param_grid,
        cv=kf,
        scoring="neg_mean_squared_error" if 'Regressor' in str(model.__name__) else "f1"
    )

    logging.info("Cross-validation step completed.")
    return search
