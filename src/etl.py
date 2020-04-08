""" Data Ingestion & Wrangling

etl.py [description]

"""

# Importing libraries
import pandas as pd
import numpy as np

# Importing scripts
import read_data as rd



def prepare_vcf(fp):
    """
    Transforms a VCF file into a machine learning
    read dataframe. Rows represent a specific sample
    and contain binary values signifying whether
    the sample has a particular SNP.
    
    :param fp: Filepath to VCF file
    :returns: Dataframe
    """
    
    # Reading VCF file into dataframe
    vcf = rd.read_vcf(fp)
    
    # Creating identifier column
    vcf['ID'] = vcf.apply(lambda x: x['#CHROM']+':'+str(x['POS']), axis=1)

    # Dropiing unnecessary columns and transposing
    drop_cols = ['#CHROM', 'POS', 'REF', 'ALT', 
                 'QUAL', 'FILTER', 'INFO', 'FORMAT']
    vcf = vcf.drop(drop_cols, axis=1).T

    # Creating mappings for reference/alternate pairs
    variant_map = {'0|0': 0, '1|0': 1, '0|1':2, '1|1':3}

    # Wrangling data
    vcf.columns = vcf.loc['ID', :]
    vcf = vcf.drop('ID', axis=0).reset_index()
    vcf.columns.name = None
    vcf = vcf.rename({'index': 'Sample'}, axis=1).replace(variant_map)

    return vcf


