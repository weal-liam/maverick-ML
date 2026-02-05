from typing import Any, Dict, Hashable
from .analysis import *
import logging

def deep_analysis(dict_one: Dict[Hashable, Any], dict_two: Dict[Hashable, Any] | None, linear_relations: Dict[Hashable, Any], skewness: Dict[Hashable, Any], target:str, cat_cols: list, num_cols: list) -> Dict[Hashable, Any]:
    """
    Deep analysis function takes on analysis from a function, makes intense 
    statistical analysis on the data, collects the deductions and measures, and returns
    a report

    ======================================================================================
    parameter: dict_one    type: Dictionary\n
    parameter: dict_two    type: Dictionary or None\n
    parameter: linear_relations    type: Dictionary\n
    parameter: skewness    type: Dictionary\n
    parameter: target      type: string\n
    parameter: cat_cols    type: list\n
    parameter: num_cols    type: list\n
    ======================================================================================
    
    return        
        rtype: Dictionary
        
    """

    #Instantiate dictionaries for analysis deductions
    report = dict()
    analysis = dict()

    #Get the possible models that could be used
    analysis["problem"] = {
        'linear':['linear_regression'],
        'classification':['knn','logistic_regression', 'RandomForestClassifier', 'DecisionTreeClassifier', 'XGBClassifier', 'LGBMClassifier'],
        'regularization':['ridge', 'lasso'],
        'tree_based':['random_forest', 'decision_tree', 'xgboost', 'lightgbm'],
    }

    #Target variable analysis
    if target in cat_cols:
        logging.info(f"Target variable '{target}' is categorical.")
        analysis["problem"].pop('linear') if 'linear' in analysis["problem"] else None
        analysis["problem"].pop('regularization') if 'regularization' in analysis["problem"] else None
    elif target in num_cols:
        logging.info(f"Target variable '{target}' is numerical.")
        analysis["problem"].pop('classification') if 'classification' in analysis["problem"] else None
    else:
        logging.warning(f"Target variable '{target}' not found in either categorical or numerical columns.")

    #Distribution by skewness    
    for key, value in linear_relations.items():
        if target in num_cols:
            if str(key) in num_cols:
                logging.info(f"==== {key} ====")

                skew_ratio = abs(skewness[key]) 
                #from skew ratio, determine skew magnitude
                if skew_ratio < 0.5:
                    logging.info("approximately symmetric")
                    skewness[f'{key}_skew_category'] = "approximately symmetric"
                elif 0.5 <= skew_ratio <= 1.0:
                    logging.info("mild skew")
                    skewness[f'{key}_skew_category'] = "mild skew"
                else:
                    logging.info("strong skew")
                    skewness[f'{key}_skew_category'] = "strong skew"

                    #if the feature is the target, add a necessary model suggestion since target largely influences model choice
                    if str(key).lower().startswith(target):
                        analysis["target_suggestion"] = "log-transform"
                    
                    if str(key).lower().startswith(target):
                        analysis["problem"].pop('linear') if 'linear' in analysis["problem"] else None
                        analysis["problem"].pop('regularization') if 'regularization' in analysis["problem"] else None

    #add skew analysis to final report
    report[f"skew"] = skewness
    
    #feature and linear relation summary
    feature_summary = dict()
    feature_summary["num_features"] = len(num_cols)
    feature_summary["cat_features"] = len(cat_cols)
    
    dominantly_categorical: bool = feature_summary["cat_features"] > feature_summary["num_features"]
    feature_summary["dominantly_categorical"] = dominantly_categorical

    if target in num_cols:
        dominantly_linear: bool = all(abs(corr) > 0.5 for corr in linear_relations.values()) 
        feature_summary["dominantly_linear"] = dominantly_linear
        if dominantly_linear:
            analysis["problem"].pop('tree_based')
        else:
            analysis["problem"].pop('linear') if 'linear' in analysis["problem"] else None
            analysis["problem"].pop('regularization') if 'regularization' in analysis["problem"] else None

    analysis["feature_summary"] = feature_summary

    #Inconsistent data
    if isinstance(dict_two, dict):
        if len(list(dict_two)) > 0:
            analysis["feature_suggestion"] = "Impute for missing values"

    #Add analysis to final report
    report[f"analysis"] = analysis

    logging.info(report)
    return report
        

    
