""" Data Processing

visualize_data.py: Library code to create data visualizations,
such as KDE plots, histograms, and scatter plots.

"""

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Libraries for plotting curves
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from itertools import cycle



def plot_polygenic_risk_scores(fp):
    """
    Creates a histogram of the polygenic risk scores.
    
    :param fp: Filepath to csv file
    """
    
    simulated_df = pd.read_csv(fp)
    
    plt.hist(simulated_df['PRS'], bins=20)
    plt.title('Polygenic Risk Scores for Unbiased Population')
    plt.xlabel('Polygenic Risk Score')
    plt.ylabel('Frequency');
    
    

def plot_risk_across_classes(fp):
    """
    Creates a histogram of the polygenic risk scores.
    
    :param fp: Filepath to csv file
    """
    
    simulated_bias_df = pd.read_csv(fp)
    
    sns.kdeplot(simulated_bias_df[simulated_bias_df['Class'] == 0]['PRS'], label="Low Risk")
    sns.kdeplot(simulated_bias_df[simulated_bias_df['Class'] == 1]['PRS'], label="Medium Risk")
    sns.kdeplot(simulated_bias_df[simulated_bias_df['Class'] == 2]['PRS'], label="High Risk")
    plt.title('Distribution of PRS Across Classes')
    plt.xlabel('Polygenic Risk Score')
    plt.ylabel('Normalized Frequency');
    

    
def plot_multiclass_roc(clf_name, clf, X_test, y_test, n_classes, figsize=(17, 6)):
    """
    Plots the multiclass version of the Receiver Operating Characteristic (ROC) 
    curve, which shows the connection/trade-off between 
    the true positive rate and false positive rate
    
    :param clf_name: String of the model's name
    :param clf: The trained model
    :param X_test: Test data of the features
    :param y_test: Test data of the labels
    :param n_classes: Number of classes
    :param figsize: Size of the ROC curve plot
    """
    
    # try to run decision_function(), which is contained in 
    # all classifiers we used except for KNN
    try:
        y_score = clf.decision_function(X_test)
    
    # except: run predict_proba() for KNN
    except:
        y_score = clf.predict_proba(X_test)

    # dictionaries to hold false positive rate (fpr), 
    # true positive rate (tpr), ROC area under the curve (roc_auc),
    # and the classes
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    classes_dict = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}

    # one-hot encode labels to determine ROC curve
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # plot of the ROC for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve for ' + clf_name)
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %s' % (roc_auc[i], classes_dict[i]))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()
    
    return fig

    
    
def plot_precision_recall(clf_name, clf, X_test, y_test, n_classes, figsize=(7, 8)):
    """
    Plots the multiclass version of the Precision-Recall (P-R) 
    curve, which shows the tradeoff between precision and recall 
    for different threshold and is a useful measure of success 
    of prediction when the classes are very imbalanced.
    
    :param clf_name: String of the model's name
    :param clf: The trained model
    :param X_test: Test data of the features
    :param y_test: Test data of the labels
    :param n_classes: Number of classes
    :param figsize: Size of the ROC curve plot
    """
    
    # try to run decision_function(), which is contained in 
    # all classifiers we used except for KNN
    try:
        y_score = clf.decision_function(X_test)
    
    # except: run predict_proba() for KNN
    except:
        y_score = clf.predict_proba(X_test)

    # dictionaries to hold precision, 
    # recall, average precision,
    # and the classes
    precision = dict()
    recall = dict()
    average_precision = dict()
    classes_dict = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}

    # one-hot encode labels to determine ROC curve
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_dummies[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test_dummies[:, i], y_score[:, i])
        
    # A "micro-average": quantifying score on all classes jointly
    # Micro-averaging is plotting a precision-recall curve by considering 
    # each element of the label indicator matrix as a binary prediction
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_dummies.ravel(),
        y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test_dummies, y_score,
                                                         average="micro")
    
    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    plt.figure(figsize=figsize)
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
    
    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))
    
    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(classes_dict[i], average_precision[i]))
    
    # plot of the P-R curve for each class
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall Curve to Multi-Class for ' + clf_name)
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    plt.show()
    
    return fig
