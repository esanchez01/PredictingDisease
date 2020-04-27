""" Data Processing

process_data.py [TODO: description]

"""

# Importing libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



def build_model(fp):
    """
    Builds a logistic regression model for 
    genetic SNP data
    
    :param fp: Filepath to csv file
    :returns: Model accuracy
    """
    
    df = pd.read_csv(fp)  
    
    # Creating training and test set
    X = df.drop(['Class', 'PRS'], axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    
    # Building model
    model = LogisticRegression(solver='liblinear', multi_class='auto')
    model.fit(X_train, y_train)
    
    # Making predictions
    preds = model.predict(X_test)
    
    # Determining accuracy
    accuracy = np.mean(preds==y_test)
    
    num_correct = int(len(df)*accuracy)
    prompt = ('{} out of {} individuals correctly classified ({} percent)'
              .format(num_correct, len(df), np.round(accuracy*100, 2)))
    print(prompt)
    
    return accuracy
