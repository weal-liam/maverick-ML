from typing import Any, Dict, Hashable
import logging

def deep_analysis(dict_one: Dict[Hashable, Any], dict_two: Dict[Hashable, Any] | None, target:str, cat_cols: list, num_cols: list) -> Dict[Hashable, Any]:
    """
    Deep analysis function takes on analysis from a function, makes intense 
    statistical analysis on the data, collects the deductions and measures, and returns
    a report
    ======================================================================================
    parameter: dict_one    type: Dictionary
    parameter: dict_two    type: Dictionary or None
    parameter: target      type: string
    parameter: cat_cols    type: list
    parameter: num_cols    type: list
    ======================================================================================
    return        
        rtype: Dictionary
        
    """

    #Instantiate dictionaries for analysis deductions
    report = dict()
    analysis = dict()

    #Get the possible models that could be used
    analysis["model_suggestions"] = ["Linear_regression","lightgbm", "xgboost", "random_forest", "decision_tree"]

    #Distribution by skewness
    skew = dict() #Instantiate a dictionary to hold skew analysis [skew direction, skew intensity, skew comment]
    
    for key, value in dict_one.items():
        if str(key) in num_cols:
            logging.info(f"==== {key} ====")

            #for each feature, store the skew analysis
            skew[key] = dict() 
            mean = value["mean"]
            median = value["50%"] if value["50%"] != 0 else 1

            #Compare means and median to hint at skewness(Non visual analysis)
            if mean > median:
                logging.info("right skewed")

                #add skew analysis to skew dictionary
                skew[key][f"skew_direction"] = "right skewed"

                #if the feature is the target, add a necessary decision
                if str(key).lower().startswith(target):
                    analysis["target_suggestion"] = "log-transform"

            elif mean < median:
                logging.info("left skewed")

                #add skew analysis to skew dictionary
                skew[key][f"skew_direction"] = "left skewed"
            
            #Calculate skew Intensity
            skew_ratio = abs(mean - median)/median
            logging.info(skew_ratio)

            #add skew analysis to skew dictionary
            skew[key][f"skew_ratio"] = skew_ratio

            #from skew ratio, determine skew magnitude
            if skew_ratio < 0.05:
                logging.info("approximately symmetric")
            elif 0.05 <= skew_ratio <= 0.15:
                logging.info("mild skew")
            else:
                logging.info("strong skew")

                #if the feature is the target, add a necessary model suggestion since target largely influences model choice
                if str(key).lower().startswith(target):
                    analysis["model_suggestions"].pop(analysis["model_suggestions"].index("Linear_regression"))
        
    #Add analysis to final report
    report[f"skew"] = skew
    report[f"analysis"] = analysis

    #Inconsistent data
    if isinstance(dict_two, dict):
        if len(list(dict_two)) > 0:
            analysis["feature_suggestion"] == "Impute for missing values"

    logging.info(report)
    return report
        

    
