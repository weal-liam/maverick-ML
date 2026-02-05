from zenml import step
from typing import Tuple, List, Dict, Hashable, Any
import pandas as pd
from data_analysis.analysis import *
import logging

from data_analysis.deep_analysis import deep_analysis

@step
def data_analysis_step(df: pd.DataFrame, target: str, desired_cols: list, undesired_cols:list) -> Tuple[Dict[Hashable, Any] | None, Dict[Hashable, Any] | None, List[str], List[str], pd.DataFrame]:    
    #Drop less significant columns if any
    df = df.drop(undesired_cols, axis=1)

    #Rename features if necessary
    df.columns = desired_cols if desired_cols else df.columns

    logging.info("Initiating analysis...")
    analyser = DataAnalyser(BasicAnalysisStrategy())

    logging.info("Basic analysis...")
    analyser.inspect(df)

    analyser.set_strategy(SimpleStatAnalysisStrategy())
    logging.info("Simple stat analysis...")
    general_stats = analyser.inspect(df)

    analyser.set_strategy(MissingValueAnalysisStrategy())
    logging.info("Missing value analysis...")
    missing_vals = analyser.inspect(df)

    analyser.set_strategy(FeatureTypeIdentifierAnalysisStrategy())
    logging.info("Feature type analysis...")
    cat_cols, num_cols = analyser.inspect(df)

    linear_relations = df.drop([*cat_cols], axis=1).corr(method="pearson")[target].to_dict() if target not in cat_cols else df.drop([*cat_cols], axis=1).corr(method="pearson").to_dict()

    skewness = df.drop(columns=cat_cols).skew().to_dict()

    final_analysis = deep_analysis(general_stats, missing_vals, linear_relations, skewness, target, cat_cols, num_cols)

    return final_analysis , None, cat_cols, num_cols, df