""" 
download_data.py: Downloads files from FTP server to data directory
"""
import numpy as np
import pandas as pd
import requests
import json
import os
import re
from ftplib import FTP

def download_ftp_data(url, path_to_folder, files, output_folder):
    """
    Downloads all specified files from folder inside the FTP server
    to ./data/[output_folder]

    :param url: url to FTP server
    :param path_to_folder: Relative path to folder from FTP server root
    :param files: List of files to download from the FTP folder
    :param output_folder: Subdirectory in ./data/ to download files to
    """
    # Make data folder
    data_folder_path = './data'
    if not os.path.exists(data_folder_path):
        print('making data_folder_path')
        os.mkdir(data_folder_path)
    
    # Make output folder
    output_folder = output_folder + "/" if output_folder[-1] != "/" else output_folder
    output_folder_path = './data/' + output_folder
    if not os.path.exists(output_folder_path):
        print('making output_folder')
        os.mkdir(output_folder_path)
        
    # Connect to FTP server
    f = FTP(url)
    f.login()
    f.cwd(path_to_folder)
    
    # Download each file from FTP server
    for file in files:
        with open(output_folder_path + file, 'wb') as fp:
            f.retrbinary('RETR ' + file, fp.write)
        with open(output_folder_path + file + '.tbi', 'wb') as fp:
            f.retrbinary('RETR ' + file + '.tbi', fp.write)


def get_gwas_trait(ID, p_upper):
    """
    Gets the gwas data for the given trait ID
    
    :param ID: ID of the trait in GWAS
    :param p_upper: Upper p value threshold of SNPs to include
    """
    # dictionary to contain data and build DataFrame
    dfdict = {'variant_id': [], 'beta': [], 'p_value': [], 
              'effect_allele_frequency': [], 'effect_allele': [], 'other_allele': []}
    
    # Simple function to add each SNP data to dfdict
    def add_snps_to_dfdict(snps, dfdict):
        for snp in snps.values():
            for col in dfdict.keys():
                dfdict[col].append(snp[col])
    # Counters
    offset = 1000
    page = 0
    
    # First page
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
    return pd.DataFrame(dfdict)

