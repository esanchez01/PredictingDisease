""" Data Ingestion & Wrangling

etl.py [TODO: description]

"""

# NOTE: The below code is written to work with UK10K data. Unfortunately,
#       the data has not become available and it's looking like it won't 
#       be coming in time. Therefore, we decided to simulate the data –
#       the function simulate_data() does this. This ultimately removes
#       the need for the rest of the functions, including the driver function.
#       We apologize for the confusion and the disorganization. We do not want 
#       to discard everything until we know the UK10K will for sure not be 
#       coming in time. This will be fixed by the next checkpoint.


# Importing libraries
import pandas as pd
import numpy as np
import os
import re
import gzip
import shutil
import subprocess as sp

# Importing scripts
from src import read_data as rd



def simulate_data(gwas_fp, maf_fp, n_samples):
    """
    Simulates a data set of individuals at different disease 
    risk levels
    
    :param gwas_fp: Filepath to GWAS TSV file
    :param maf_fp: Filepath to SNP minor allele frequency file
    :param n_samples: Number of individuals to simulate
    :returns: Simulated data frame
    """
    
    # Reading and cleaning GWAS data
    gwas = pd.read_csv(gwas_fp, sep='\t')
    gwas = gwas.dropna(subset=['OR or BETA'], axis=0)
    
    # Reading MAF data
    maf_cols = ['Variation ID', 'Minor Allele Global Frequency']
    maf = pd.read_csv(maf_fp, sep='\t', usecols=maf_cols)
    
    # Cleaning MAF data
    maf = maf.drop_duplicates(subset=maf_cols)
    maf = maf[maf['Minor Allele Global Frequency'] != 'None']
    maf['Minor Allele Global Frequency'] = (maf['Minor Allele Global Frequency']
                                               .astype(float))

    # Creating labels that will be associated with disease risk
    # 0=Low   1=Mid   2=High
    risk_labels = [0, 1, 2]

    # Defining the probability of being a label
    # Low=55%   Mid=30%   High=15%
    risk_prob = [.55, .3, .15]

    # Defining values to increase probability of having SNP
    # Low=0%   Mid=50%   High=100%
    risk_bias = [1, 1.5, 2]

    # Simulating
    indiv_rows_bias = []
    indiv_class = []
    N = n_samples
    for _ in range(N):
        label = np.random.choice(a=risk_labels, p=risk_prob)
        bias = risk_bias[label]
        has_snps = (maf['Minor Allele Global Frequency']
                    .apply(lambda x: np.random.choice(a=[0,1], p=[1-(x*bias), (x*bias)])))
        indiv_rows_bias.append(has_snps.values)
        indiv_class.append(label)
    
    # Creating dataframe
    simulated_df = pd.DataFrame(indiv_rows_bias, columns=maf['Variation ID'])
    simulated_df.columns.name = ''
    
    # Calculating polygenic risk score
    beta_dict = gwas.set_index('SNPS')['OR or BETA'].to_dict()
    beta_values = np.array([beta_dict.get(x) for x in simulated_df.columns])
    prs = simulated_df.apply(lambda x: (x*beta_values).sum(), axis=1)
    simulated_df['PRS'] = prs
    
    # Creating label
    simulated_df['Class'] = indiv_class
    
    return simulated_df

    
    
    
# ---------------------------------------------------------------------
# Driver Function
# ---------------------------------------------------------------------
def get_data_test_simulated(gwas_fp, maf_fp, n_samples, outpath):
    """
    Reads in the desired data in test-params.json 
    and uses the configuration to download 
    the various file types and corresponding CSV files.

    :param gwas_fp: Filepath to GWAS TSV file
    :param maf_fp: Filepath to SNP minor allele frequency file
    :param n_samples: Number of individuals to simulate
    :outpath: Path to save data frame
    """
    
    # Check whether the specified path exists or not
    data_folder_path = './data'
    data_folder_path_exists = os.path.exists(data_folder_path) 
    if not data_folder_path_exists:
        os.mkdir(data_folder_path)
    
    # Check whether the specified path exists or not
    path_exists = os.path.exists(outpath) 
    if not path_exists:
        os.mkdir(outpath)
        
    # Creating simulated data
    df = simulate_data(gwas_fp, maf_fp, n_samples)
    
    # Saving dataframe to outpath
    fp = outpath+'/simulated_data.csv'
    df.to_csv(fp, index=False)
    
    return fp

    
    

### CODE BELOW IS FOR UK10K, PLEASE READ NOTE ABOVE ###


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
    vcf['ID'] = vcf.apply(lambda x: str(x['#CHROM'])+':'+str(x['POS']), axis=1)

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
    snps_str = ', '.join(pd.read_csv(tsv_path, sep='\t')['SNPS'])
    
    # opens script shell file
    print('opening script')
    cmd_str = ("./src/scripts/filter_snps.sh " +
               vcf_path + " " + snps_str + " " +
               str(maf) + " " + str(geno) + " " + str(mind))
    proc = sp.Popen(cmd_str, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
    
    # runs script shell file
    print('running script')
    out_tuple = proc.communicate()
    print('script finished running')
    return out_tuple



def filter_merge_by_chr(folder_path, vcf_files_dict, tsv_path, **kwargs):
    """
    Runs script shell file to run plink2 commands to filter each VCF file
    in specified folder. Filters SNP IDs for the chromosome number each file
    corresponds to. After filtering each file, merges all VCFs into one
    file named merged.vcf in the target folder

    :param folder_path: Folder containing the vcf files, and where the merged VCF file will be saved
    :param vcf_files_dict: Dictionary mapping each chromosome number (1-22) to a VCF file path
    :param tsv_path: The file path to the file containing relevant SNPs from the GWAS
    :param kwargs: Extra key word arguments
    :returns: output of script
    """

    # Read snps
    snps = pd.read_csv(tsv_path, sep='\t')

    # Make temporary folder to store filtered VCFs
    folder_path = folder_path + "/" if folder_path[-1] != "/" else folder_path
    temporary_path = folder_path + "temp_vcfs/"
    if not os.path.exists(temporary_path):
        os.mkdir(temporary_path)

    # Filter each VCF and output filtered VCF into temporary folder
    for chr_id, vcf_path in vcf_files_dict.items():
        done = False
        cur_snps = snps[snps['CHR_ID'] == chr_id]['SNPS'].unique().tolist()
        while not done:
            snps_str = ', '.join(cur_snps)
            cmd_str = "plink2 --vcf " + folder_path + vcf_path + " --max-alleles 2 --make-bed --snps " + snps_str +\
                      " --recode vcf --out " + temporary_path + "chr_" + chr_id
            proc = sp.Popen(cmd_str, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
            output = proc.communicate()
            if output[1]:
                missing_snp = re.findall('rs[0-9]+', str(output[1]))
                if missing_snp:
                    cur_snps.remove(missing_snp[0])
                    print("Removed " + missing_snp[0])
            else:
                done = True


    # Merge vcfs
    merge_vcfs(temporary_path)

    # Delete temporary folder
    shutil.rmtree(temporary_path, ignore_errors=True)



def merge_vcfs(folder_path):
    """
    Runs script shell file to merge vcfs into one vcf, creates
    a file named merged.vcf.gz in the target folder

    :param folder_path: Path to folder containing the vcf files
    """
    proc = sp.Popen('./src/scripts/merge_vcfs.sh ' + folder_path, shell=True)
    out_tuple = proc.communicate()
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
