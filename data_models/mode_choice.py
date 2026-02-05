from abc import ABC, abstractmethod
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

class model_choice(ABC):
    @abstractmethod
    def choose_model(self):
        pass

class LinearRegressionModelChoice(model_choice):
    def choose_model(self) -> LinearRegression:
        model = LinearRegression
        return model  

class DecisionTreeModelChoice(model_choice):
    def choose_model(self) -> DecisionTreeRegressor:
        model = DecisionTreeRegressor
        return model

class RandomForestModelChoice(model_choice):
    def choose_model(self) -> RandomForestRegressor:
        model = RandomForestRegressor
        return model

class XGBoostModelChoice(model_choice):
    def choose_model(self) -> XGBRegressor:
        model = XGBRegressor
        return model

class LightGBMModelChoice(model_choice):
    def choose_model(self) -> LGBMRegressor:
        model = LGBMRegressor
        return model 

class RidgeModelChoice(model_choice):
    def choose_model(self) -> Ridge:
        model = Ridge
        return model
    
class LassoModelChoice(model_choice):
    def choose_model(self) -> Lasso:
        model = Lasso
        return model

class KNNModelChoice(model_choice):
    def choose_model(self) -> KNeighborsClassifier:
        model = KNeighborsClassifier
        return model

class LogisticRegressionModelChoice(model_choice):
    def choose_model(self) -> LogisticRegression:
        model = LogisticRegression
        return model

class DecisionTreeClassifierChoice(model_choice):
    def choose_model(self) -> DecisionTreeClassifier:
        model = DecisionTreeClassifier
        return model
    
class RandomForestClassifierChoice(model_choice):
    def choose_model(self) -> RandomForestClassifier:
        model = RandomForestClassifier
        return model
    
class LightGBMClassifierChoice(model_choice):
    def choose_model(self) -> LGBMClassifier:
        model = LGBMClassifier
        return model
    
class XGBoostClassifierChoice(model_choice):
    def choose_model(self) -> XGBClassifier:
        model = XGBClassifier
        return model

class ModelSelector:
    def __init__(self , required_model_type: str):
        self.model_choice = None
        self.required_model_type = required_model_type.lower()

    def get_model(self):
        if self.required_model_type == "linear_regression":
            self.model_choice = LinearRegressionModelChoice()
        elif self.required_model_type == "decision_tree":
            self.model_choice = DecisionTreeModelChoice()
        elif self.required_model_type == "random_forest":
            self.model_choice = RandomForestModelChoice()
        elif self.required_model_type == "xgboost":
            self.model_choice = XGBoostModelChoice()
        elif self.required_model_type == "lightgbm":
            self.model_choice = LightGBMModelChoice()
        elif self.required_model_type == "logistic_regression":
            self.model_choice = LogisticRegressionModelChoice()
        elif self.required_model_type == "decision_tree_classifier":
            self.model_choice = DecisionTreeClassifierChoice()
        elif self.required_model_type == "random_forest_classifier":
            self.model_choice = RandomForestClassifierChoice()
        elif self.required_model_type == "lightgbm_classifier":
            self.model_choice = LightGBMClassifierChoice()
        elif self.required_model_type == "xgboost_classifier":
            self.model_choice = XGBoostClassifierChoice()
        elif self.required_model_type == "knn":
            self.model_choice = KNNModelChoice()
        elif self.required_model_type == "ridge":
            self.model_choice = RidgeModelChoice()
        elif self.required_model_type == "lasso":
            self.model_choice = LassoModelChoice()
        else:
            raise ValueError(f"Unsupported model type: {self.required_model_type}")

        return self.model_choice.choose_model()