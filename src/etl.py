""" Data Ingestion & Wrangling

etl.py: Library code to ingest data from the GWAS Catalog 
(https://www.ebi.ac.uk/gwas/), creates simulated populations 
and prepares for model building.

"""


# Importing libraries
import pandas as pd
import numpy as np
import os
import re
import gzip
import shutil
import subprocess as sp
import requests
import json



def get_gwas_trait(simulated_data, model_data, outpath):
    """
    Gets the gwas data for the given trait ID
    
    :param simulated_data: Dictionary containing ID of GWAS data for 
                           simulation and max p value for the data
    :param model_data: Dictionary containing ID of GWAS data for 
                       model building and max p value for the data
    :param outpath: Path to save data
    """
    print('Collecting GWAS data..')
    
    def add_snps_to_dfdict(snps, dfdict):
        """Function to add each SNP data to dfdict"""
        for snp in snps.values():
            for col in dfdict.keys():
                dfdict[col].append(snp[col])
                
    # Check whether the specified path exists or not
    path_exists = os.path.exists(outpath) 
    if not path_exists:
        os.makedirs(outpath)
                
    for data in [simulated_data, model_data]:
        
        # dictionary to contain data and build DataFrame
        dfdict = {'variant_id': [], 'beta': [], 
                  'p_value': [], 'effect_allele_frequency': [], 
                  'effect_allele': [], 'other_allele': []}

        # Counters
        offset, page = 1000, 0

        # First page
        ID = data["gwas"]
        p_upper = data["max_p_value"]
        req = requests.get(f'https://www.ebi.ac.uk/gwas/summary-statistics/api/traits/{ID}' +
                           f'/associations?start={page}&size={offset}&p_upper={p_upper}').content
        snps = json.loads(req)['_embedded']['associations']
        add_snps_to_dfdict(snps, dfdict)

        # Continually query pages from the API until last page is reached
        while len(snps) == offset:
            page += offset
            req = requests.get(f'https://www.ebi.ac.uk/gwas/summary-statistics/api/traits/{ID}' +
                               f'/associations?start={page}&size={offset}&p_upper={p_upper}').content
            try:
                snps = json.loads(req)['_embedded']['associations']
                add_snps_to_dfdict(snps, dfdict)
            except:
                break

        # Saving data
        snp_df = pd.DataFrame(dfdict)
        snp_df.to_csv(outpath+'/{}.csv'.format(ID), index=False)
        
        print('- {} collected.'.format(ID))



def simulate_data(outpath, gwas_fp, n_samples):
    """
    Simulates a data set of individuals at different disease 
    risk levels
    
    :param outpath: File path to save simulated data
    :param gwas_fp: Filepath to GWAS TSV file
    :param n_samples: Number of individuals to simulate
    :returns: Simulated data filepath
    """
    print('Simulating population..')
    # Reading and cleaning GWAS data
    gwas = pd.read_csv(gwas_fp)
    gwas = gwas.dropna(subset=['beta', 'effect_allele_frequency'], axis=0)

    # Creating labels that will be associated with disease risk
    # 0=Low   1=Mid   2=High
    risk_labels = [0, 1, 2]

    # Defining the probability of being a label
    risk_prob = [.6, .3, .1]

    # Defining values to scale probability of having SNP
    low_risk_bias = np.arange(.1, .5, .05)
    medium_risk_bias = np.arange(.5, .76, .05)
    high_risk_bias = np.arange(.75, 1, .05)
    risk_bias = [low_risk_bias, medium_risk_bias, high_risk_bias]

    # Simulating
    indiv_rows_bias = []
    indiv_class = []
    N = n_samples
    for _ in range(N):
        label = np.random.choice(a=risk_labels, p=risk_prob)
        bias = np.random.choice(risk_bias[label])
        has_snps = (gwas['effect_allele_frequency']
                    .apply(lambda x: 
                           np.random.choice(a=[0,1], p=[1-(x*bias), (x*bias)])))
        indiv_rows_bias.append(has_snps.values)
        indiv_class.append(label)
    
    # Creating dataframe
    simulated_df = pd.DataFrame(indiv_rows_bias, columns=gwas['variant_id'])
    simulated_df.columns.name = ''
    
    # Calculating polygenic risk score
    beta_dict = gwas.set_index('variant_id')['beta'].to_dict()
    beta_values = np.array([beta_dict.get(x) for x in simulated_df.columns])
    prs = simulated_df.apply(lambda x: (x*beta_values).sum(), axis=1)
    simulated_df['PRS'] = prs
    
    # Creating label
    simulated_df['Class'] = indiv_class
    
    # Saving simulated data to outpath
    sim_fp = outpath+'/simulated_data.csv'
    simulated_df.to_csv(sim_fp, index=False)
    
    return sim_fp

    
    
# ---------------------------------------------------------------------
# Driver Function
# ---------------------------------------------------------------------

def get_data(simulated_data, model_data, outpath, test=False):
    """
    Reads in the desired data in test-params.json 
    and uses the configuration to download 
    the various file types and corresponding CSV files.

    :param simulated_data: Dictionary containing ID of GWAS data for 
                           simulation, max p value for the data, and
                           the number of samples to simulate
    :param model_data: Dictionary containing ID of GWAS data for 
                       model building and max p value for the data
    :param test: Whether function call is for testing
    :outpath: Path to save GWAS data
    :returns: File paths to simulated data and model GWAS data
    """
    
    # Check whether the specified path exists or not
    path_exists = os.path.exists(outpath) 
    if not path_exists:
        os.makedirs(outpath)
    
    if not test:
        # Collecting GWAS data
        get_gwas_trait(simulated_data, model_data, outpath)
    
    # Creating simulated data
    if not test:
        gwas_fp = outpath+'/{}.csv'.format(simulated_data['gwas'])
    else:
        gwas_fp = simulated_data['gwas']
    simulate_data(outpath, gwas_fp, simulated_data['n_samples'])
    
    # Determining simulated data filepath
    sim_fp = outpath+'/simulated_data.csv'
    
    # Determining model data filepath
    if not test:
        model_fp = outpath+'/{}.csv'.format(model_data['gwas'])
    else:
        model_fp = model_data['gwas']
    
    return sim_fp, model_fp
