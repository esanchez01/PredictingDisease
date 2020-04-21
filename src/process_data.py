""" Data Processing

process_data.py [TODO: description]

"""

# Importing libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



def build_model(df):
    """
    Builds a logistic regression model for 
    genetic SNP data
    
    :param df: Dataframe containing SNP data
    :returns: Dictionary with classes as keys
              and mean probability as value
    """
    
    # Creating target column
    # NOTE: Temporary, will remove when data is available
    df['has_disease'] = np.random.choice([0,1], size=len(df))
    
    # Creating training and test set
    X = df.drop('has_disease', axis=1)
    y = df['has_disease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    
    # Building model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Making predictions
    preds = model.predict_proba(X_test)
    
    # Creating output dictionary
    result_df = pd.DataFrame({'True': y_test, 'Predictions': preds[:,1]})
    result_dict = result_df.groupby('True')['Predictions'].mean()
    
    return result_dict
