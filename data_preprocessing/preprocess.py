from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from zenml import step
from sklearn.model_selection import train_test_split
from typing import Tuple, Annotated
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

@step
def preprocess_data(df : pd.DataFrame, target:str, analysis, cat_cols: list, num_cols: list, preferred_cat:str) -> Tuple[Annotated[pd.DataFrame, "X_train"], Annotated[pd.DataFrame, "X_test"], Annotated[pd.Series, "y_train"], Annotated[pd.Series, "y_test"], Annotated[ColumnTransformer, "preprocessor"]]:     
    """
    Preprocess data step
    ====================================
    param df -> type: pandas DataFrame

    param target -> type: string

    param analysis -> type: a dictionary

    param cat_cols -> type: list

    param num_cols -> type: list
    ====================================
    return: 
    rtype: Tuple[DataFrame, DataFrame, Series[Any], Series[Any], ColumnTransformer]
    """
    
    
    
    logging.info(f'Beginning preprocessing...')
    
    #get column types for feature differentiation logic
    #instantiate a list
    col_types = list() 

    #find the feature types from dataframe, and add each to the col_types list 
    for t in df.dtypes.to_dict().values():
        col_types.append(str(t))

    logging.info(f'Dataframe shape before preprocessing: {df.shape} \n {num_cols} \n {cat_cols} \n {df.head()}')
    logging.info(f'Columns in dataframe before preprocessing: {df.columns.tolist()}')

    #Split features and target
    X = df.drop([target], axis=1)
    y = df[target]

    #Target Preprocess, checks whether any analysis suggestion were given for the target
    #target_analysis = analysis["analysis"]["target_suggestion"]

    #Binary transformation
    if str(y.dtype) == 'object':
       y = (y == preferred_cat).astype(int)

    #Remove Target from these columns to avoid future pipeline errors
    if target in cat_cols:
        cat_cols.remove(target)
    if target in num_cols:
        num_cols.remove(target)

    #Split into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    #Feature Preprocess to handle missing values and object type categories
    if "object" in col_types:
        X_cat = cat_cols
        X_num = num_cols

    #Instantiate preprocessor
        preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), X_num),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
            ]), X_cat)
        ]
        )
    else:
    #else for already refined data but handle for missing values just incase
        X_cat = cat_cols
        X_num = num_cols

        preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), X_num),
            ("cat", SimpleImputer(strategy="most_frequent"), X_cat),
        ]
        )

    #return training data, testing data, and the preprocessor for following steps
    return pd.DataFrame(X_train), pd.DataFrame(X_test), pd.Series(y_train), pd.Series(y_test), preprocessor