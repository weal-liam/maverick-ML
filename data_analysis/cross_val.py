from zenml import step
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from data_models.model_train import get_model
import numpy as np
from typing import Any

@step
def cross_validation_step(analysis, preprocessor) -> GridSearchCV:
    model = analysis['analysis']["model_suggestions"][0]
    model = get_model(model)      
    
    pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", model())
    ])

    #future, design function for multiple parameter grids for different models
    param_grid = {
        "model__n_estimators" : [200, 300, 500, 700],
        "model__learning_rate" :[0.05],
        "model__random_state" :[42]
    }

    kf = KFold(n_splits=6, shuffle=True, random_state=42)
    
    search = GridSearchCV(
        pipe,
        param_grid,
        cv=kf,
        scoring="neg_mean_squared_error"
    )

    return search
