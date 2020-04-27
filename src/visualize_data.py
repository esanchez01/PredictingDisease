""" Data Processing

visualize_data.py: Library code to create data visualizations,
such as KDE plots, histograms, and scatter plots.

"""

# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



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
