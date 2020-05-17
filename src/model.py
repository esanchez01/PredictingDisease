""" Data Processing

process_data.py: Library code to construct a Support Vector 
Machine model on population SNP data and generate/save results 
for the model's performance.

"""

# Importing libraries
import pandas as pd
import numpy as np
from sklearn.svm import SVC
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
    
    # Building model
    model = SVC(C=10, tol=0.1, gamma='auto')
    model.fit(X_train, y_train)
    
    # Making predictions
    preds = model.predict(X_test)
    
    # Determining accuracy
    accuracy = np.mean(preds==y_test)
    print('\nSummary Results:')
    
    # Printing immediate results to user
    num_correct = int(len(df)*accuracy)
    rounded_accuracy = np.round(accuracy*100, 2)
    prompt = ('{}: {} out of {} individuals correctly classified ({} percent)'
              .format('Accuracy', num_correct, len(df), rounded_accuracy))
    print(prompt)
    
    # Saving model plots
    roc = vd.plot_multiclass_roc('SVM', model, X_test, y_test, 3, (10, 6))
    roc.savefig(outpath+'/ROC_plot.png')
    
    pr = vd.plot_precision_recall('SVM', model, X_test, y_test, 3)
    pr.savefig(outpath+'/PR_plot.png')
    
    # Saving classification report
    target_names = ['Low Risk', 'Medium Risk', 'High Risk']
    c_report = pd.DataFrame(classification_report(y_test, preds, 
                                     target_names=target_names, 
                                     output_dict=True))
    c_report.to_csv(outpath+'/report.csv', index=True)
    
    print('\nFull model results saved at {}'.format(outpath))
    
    
    return accuracy
