""" Data Processing

process_data.py: Library code to construct a Support Vector 
Machine model on population SNP data and generate/save results 
for the model's performance.

"""

# Importing libraries
import pandas as pd
import numpy as np
import json
# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# Model utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Importing scripts
import visualize_data as vd



def build_model(sim_fp, model_fp, outpath):
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
    
    model_params = json.load(open('config/model-params.json', 'r'))
    results = pd.DataFrame()
    # Iterate through models
    modelNames = {
        "LogisticRegression": LogisticRegression,
        "KNeighborsClassifier": KNeighborsClassifier,
        "SVC": SVC,
        "GaussianNB": GaussianNB,
        "RandomForestClassifier": RandomForestClassifier,
        "DecisionTreeClassifier": DecisionTreeClassifier
    }
    
    finalModels = {}
    for name, model in modelNames.items():
        model_fit, model_results = get_model_results(model_params[name], name,
                                          model, X_train, X_test, y_train, y_test)
        model_results['Model Type'] = name
        results = results.append(model_results)
        finalModels[name] = model_fit
    
    
    
    
    # Saving model plots for SVM
    roc = vd.plot_multiclass_roc('SVM', finalModels['SVC'], X_test, y_test, 3, (10, 6))
    roc.savefig(outpath+'/SVM_ROC_plot.png')
    
    pr = vd.plot_precision_recall('SVM', finalModels['SVC'], X_test, y_test, 3)
    pr.savefig(outpath+'/SVM_PR_plot.png')
    
    results.to_csv(outpath+'/results.csv')
    print('\nFull model results saved at {}'.format(outpath))
    
    

def get_model_results(model_params, model_name, sklearn_model, X_train, X_test, y_train, y_test):
    """
    Fits the given model and returns the fit model and results
    
    :param model_params: parameters to use for the model
    :param model_name: name of the model type
    :param sklearn_model: pointer to sklearn model class
    :param X_train: X training set
    :param X_test: X test set
    :param y_train: y training set
    :param y_test: y test set
    :returns: tuple of the model and the model results
    """
    model = sklearn_model(**model_params)
    model.fit(X_train, y_train)
    
    # Making predictions
    preds = model.predict(X_test)
    
    # Saving classification report
    target_names = ['Low Risk', 'Medium Risk', 'High Risk']
    return (model, pd.DataFrame(classification_report(y_test, preds, 
                                     target_names=target_names, 
                                     output_dict=True)))
