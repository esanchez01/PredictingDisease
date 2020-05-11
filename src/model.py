""" Data Processing

process_data.py [TODO: description]

"""

# Importing libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



def build_model(sim_fp, model_fp):
    """
    Builds a logistic regression model for 
    genetic SNP data
    
    :param fp: Filepath to simulated data
    :param fp: Filepath to model GWAS data
    :param cols: SNP column names to keep
    :returns: Model accuracy
    """
    
    print('Building and testing model..')
    
    label_cols = ['Class', 'PRS']
    
    # Loading in model GWAS data
    model_data = pd.read_csv(model_fp, usecols=['variant_id'])
    model_snps = model_data['variant_id'].unique()
    
    # Loading in simulated data and filtering to model SNPs
    df = pd.read_csv(sim_fp)
    sim_snps = df.drop(label_cols, axis=1).columns
    keep_cols = list(set(sim_snps).intersection(model_snps))
    df = df[keep_cols+label_cols]
    
    # Creating training and test set
    X = df.drop(label_cols, axis=1)
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
