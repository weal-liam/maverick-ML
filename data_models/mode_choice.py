from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

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
        else:
            raise ValueError(f"Unsupported model type: {self.required_model_type}")

        return self.model_choice.choose_model()