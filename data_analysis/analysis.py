from abc import ABC, abstractmethod
from typing import Dict, Hashable, Any, Tuple, List
from scipy.stats import skew, pearsonr, spearmanr, f_oneway
import pandas as pd

class AnalysisStrategy(ABC):
    @abstractmethod
    def analyse(self, df: pd.DataFrame):
        pass

class BasicAnalysisStrategy(AnalysisStrategy):
    def analyse(self, df: pd.DataFrame) -> None:
        return df.info()

class SimpleStatAnalysisStrategy(AnalysisStrategy):
    def analyse(self, df: pd.DataFrame) -> Dict[Hashable, Any]:
        analysis = df.describe()
        stat = analysis.to_dict()
        return stat

class MissingValueAnalysisStrategy(AnalysisStrategy):
    def analyse(self, df: pd.DataFrame) -> Dict[Hashable, Any] | None:
        analysis = df.isna().sum().sort_values()
        missing = dict[str]
        for key, value in analysis.to_dict().items():
            if value > 0:
                missing[key] = value

        return missing if len(list(missing)) == 0 else None
    
class SingleVarAnalysisStrategy(AnalysisStrategy):
    def analyse(self, feature: pd.Series) -> Dict[Hashable, Any]:
        analysis = feature.describe()
        return analysis.to_dict()
    
class LinearRelationAnalysisStrategy(AnalysisStrategy):
    def analyse(self, feature_1, feature_2) -> Tuple[float ,float]:
        corr, p = pearsonr(feature_1, feature_2)
        return corr ,p
    
class NonLinearRelationAnalysisStrategy(AnalysisStrategy):
    def analyse(self, feature_1, feature_2) -> Tuple[float ,float]:
        corr, p = spearmanr(feature_1, feature_2)
        return corr ,p
    
class CategoricalEffectAnalysisStrategy(AnalysisStrategy):
    def analyse(self, df: pd.DataFrame, category, feature) -> Dict[Hashable, Any]:
        analysis = df.groupby(category)[feature].agg(["mean", "std", "count"])
        return analysis.to_dict()

class CategoricalSignificanceAnalysisStrategy(AnalysisStrategy):
    def analyse(self, df: pd.DataFrame, category, feature) -> float:
        groups = [df[df[category] == value ][feature] for value in df[category].unique()]
        f_stat, p = f_oneway(*groups)
        return p

class FeatureTypeIdentifierAnalysisStrategy(AnalysisStrategy):
    def analyse(self, df: pd.DataFrame) -> Tuple[List[str],List[str]]:
        categorical_cols = list()
        numerical_cols = list()
    
        for col in df.columns:
            if str(df[col].dtype) == 'object' or 2 <= df[col].nunique() <= 23:
                if col not in ['hour', 'day', 'month']:
                    categorical_cols.append(col) 
            else:
                numerical_cols.append(col)

        return categorical_cols, numerical_cols

class DataAnalyser():
    def __init__(self, strategy: AnalysisStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: AnalysisStrategy):
        self._strategy = strategy

    def inspect(self, df: pd.DataFrame) -> Any:
        return self._strategy.analyse(df)

class UniVarAnalyser():
    def __init__(self, target, strategy: AnalysisStrategy):
        self.target = target
        self.strategy = strategy

    def set_target(self, target):
        self.target = target

    def set_strategy(self, strategy: AnalysisStrategy):
        self.strategy = strategy

    def inspect(self) -> Any:
        return self.strategy.analyse(self.target)

class BiVarAnalyser():
    def __init__(self, feature_1, feature_2, strategy: AnalysisStrategy):
        self.feature_1 = feature_1
        self.feature_2 = feature_2
        self.strategy = strategy

    def set_target(self, feature_1, feature_2):
        self.feature_1 = feature_1
        self.feature_2 = feature_2

    def set_strategy(self, strategy: AnalysisStrategy):
        self.strategy = strategy

    def inspect(self) -> Any:
        return self.strategy.analyse(self.feature_1, self.feature_2)

    def inspect(self, df: pd.DataFrame) -> Any:
        return self.strategy.analyse(df,self.feature_1, self.feature_2)