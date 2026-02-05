model_params ={
    "LinearRegression_param_grid": {
        "model__fit_intercept": [True, False],
        "model__normalize": [False],
        "model__copy_X": [True],
    },
    "LogisticRegression_param_grid": {
        "model__penalty": ["l2", "none"],
        "model__C": [0.1, 1.0, 10.0],
        "model__solver": ["lbfgs", "saga"],
        "model__max_iter": [100, 200],
    },
    "DecisionTreeRegressor_param_grid": {
        "model__criterion": ["squared_error", "absolute_error"],
        "model__splitter": ["best", "random"],
        "model__max_depth": [None, 5, 10],
    },
    "DecisionTreeClassifier_param_grid": {
        "model__criterion": ["gini", "entropy"],
        "model__splitter": ["best", "random"],
        "model__max_depth": [None, 5, 10],
    },
    "RandomForestRegressor_param_grid": {
        "model__n_estimators": [100, 200],
        "model__criterion": ["squared_error", "absolute_error"],
        "model__max_depth": [None, 5, 10],
    },
    "RandomForestClassifier_param_grid": {
        "model__n_estimators": [100, 200],
        "model__criterion": ["gini", "entropy"],
        "model__max_depth": [None, 5, 10],
    },
    "XGBRegressor_param_grid": {
        "model__n_estimators": [100, 200, 300, 400, 500],
        "model__learning_rate": [0.01, 0.05],
        "model__max_depth": [3, 5, 7],
        "model__random_state": [42]
    },
    "XGBClassifier_param_grid": {
        "model__n_estimators": [100, 200, 300, 400, 500],
        "model__learning_rate": [0.01, 0.05],
        "model__max_depth": [3, 5, 7],
        "model__random_state": [42]
    },
    "LGBMRegressor_param_grid": {
        "model__n_estimators": [100, 200, 300, 400, 500],
        "model__learning_rate": [0.01, 0.05],
        "model__max_depth": [3, 5, 7],
        "model__random_state": [42]
    },
    "LGBMClassifier_param_grid": {
        "model__n_estimators": [100, 200, 300, 400, 500],
        "model__learning_rate": [0.01, 0.05],
        "model__max_depth": [3, 5, 7],
        "model__random_state": [42]
    },
    "Ridge_param_grid": {
        "model__alpha": [0.1, 1.0, 10.0],
        "model__fit_intercept": [True, False],
        "model__solver": ["auto", "svd"],
    },
    "Lasso_param_grid": {
        "model__alpha": [0.1, 1.0, 10.0],
        "model__fit_intercept": [True, False],
        "model__selection": ["cyclic", "random"],
    },
    "KNeighborsClassifier_param_grid": {
        "model__n_neighbors": [3, 5, 7, 9],
        "model__weights": ["uniform", "distance"],
        "model__algorithm": ["auto", "ball_tree", "kd_tree"],
    },
}