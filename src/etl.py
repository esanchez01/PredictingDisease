""" Data Ingestion & Wrangling

etl.py [TODO: description]

"""

# Importing libraries
import pandas as pd
import numpy as np
import os
import gzip
import shutil
import subprocess as sp

# Importing scripts
from src import read_data as rd



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
    # NOTE: Temporary, will be replaced by rsID
    vcf['ID'] = vcf.apply(lambda x: x['#CHROM']+':'+str(x['POS']), axis=1)

    # Dropping unnecessary columns and transposing
    drop_cols = ['#CHROM', 'POS', 'REF', 'ALT', 
                 'QUAL', 'FILTER', 'INFO', 'FORMAT']
    vcf = vcf.drop(drop_cols, axis=1).T

    # Creating mappings for reference/alternate pairs
    variant_map = {'0|0': 0, '1|0': 1, '0|1':1, '1|1':2}

    # Wrangling data
    vcf.columns = vcf.loc['ID', :]
    vcf = vcf.drop('ID', axis=0).reset_index(drop=True)
    vcf.columns.name = None
    vcf = vcf.replace(variant_map)

    return vcf



def get_table_vcf_test(chromosome, outpath, file_type):
    """
    Gets the VCF from the given testdata directory,
    reads in the VCF as a DataFrame and converts it to a CSV file,
    and saves the VCF and CSV files to a specified directory.

    :param chromosome: The chromosome number
    :param outpath: The path to save data
    :param file_type: The file type of the genetic file
    """
    
    # given test data input and corresponding output
    in_fn  = './testdata/vcf/chr22_test.vcf.gz'
    out_fn = './data/raw/vcf/chr22_test.vcf'
    
    # Check whether the specified path exists or not
    path_exists = os.path.exists(outpath) 
    if not path_exists:
        print('making outpath:', outpath)
        os.mkdir(outpath)
    
    # Check whether .vcf.gz file is already unzipped
    unzip_file = os.path.exists(out_fn)
    print('Does chr22_test.vcf exist?', unzip_file)
    if not unzip_file:
        print('converting .vcf.gz file to .vcf file')
        with gzip.open(in_fn, 'rb') as f_in, open(out_fn, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def filter_vcf(vcf_path, maf, geno, mind, tsv_path, **kwargs):
    """
    Runs script shell file to run plink2 commands to
    filter the VCF file.
    
    :param vcf_path: The file path to the input VCF file
    :param maf: The minor allele frequency
    :param geno: The value to filter variants with missing call rates exceeding its value
    :param mind: The value to filter samples with missing call rates exceeding its value
    :param kwargs: Extra key word arguments
    :param tsv_path: The file path to the file containing relevant SNPs from the GWAS
    :returns: output of script
    """
    
    # calls helper functions to get the relevant SNPs as a string
    vcf = rd.read_vcf(vcf_path)
    snps_str = ', '.join(pd.read_csv(tsv_path, sep='\t')['SNPS'])
    
    # opens script shell file
    print('opening script')    
    cmd_str = ("./src/filter_snps/filter_snps.sh " + 
               vcf_path + " " + snps_str + " " + 
               str(maf) + " " + str(geno) + " " + str(mind))
    proc = sp.Popen(cmd_str, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
    
    # runs script shell file
    print('running script')
    out_tuple = proc.communicate()
    print('script finished running')
    return out_tuple



# ---------------------------------------------------------------------
# Driver Function
# ---------------------------------------------------------------------
def get_data_test(chromosomes, samples, outpath, file_types, **kwargs):
    """
    Reads in the desired data in test-params.json 
    and uses the configuration to download 
    the various file types and corresponding CSV files.

    :param chromosomes: The chromosome numbers
    :param samples: The sample numbers
    :param outpath: The directory to which to save the data
    :param file_types: The genetic file types
    :param kwargs: Extra keyword arguments
    """
    
    # Check whether the specified path exists or not
    data_folder_path = './data'
    data_folder_path_exists = os.path.exists(data_folder_path) 
    if not data_folder_path_exists:
        print('making data_folder_path')
        os.mkdir(data_folder_path)
    
    # Check whether the specified path exists or not
    path_exists = os.path.exists(outpath) 
    if not path_exists:
        print('making outpath:', outpath)
        os.mkdir(outpath)
        
    # loop through each file type to create a saved data directory
    for file_type in file_types:
        # Check whether the specified savedir exists or not
        savedir = outpath + '/' + file_type
        savedir_exists = os.path.exists(savedir) 
        if not savedir_exists:
            print('making savedir:', savedir)
            os.mkdir(savedir)
    
    # loop through each file type and chromosome to get table
    for file_type in file_types:
        print('get_data_test() - file_type:', file_type)
        savedir = outpath + '/' + file_type
        if file_type == 'vcf':
            for chromosome in chromosomes:
                get_table_vcf_test(chromosome, savedir, file_type)
        elif file_type == 'bam':
            for sample in samples:
                get_table_bam_test(sample, savedir, file_type)
        elif file_type == 'fastq':
            for sample in samples:
                get_table_fastq_test(sample, savedir, file_type)
                get_table_fasta(sample, savedir, file_type)
