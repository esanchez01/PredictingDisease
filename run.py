#!/usr/bin/env python

import os
import sys
import json
import shutil

sys.path.insert(0, 'src')
from etl import *
from model import *

TEST_PARAMS = 'config/test-params.json'
DATA_PARAMS = 'config/data-params.json'



def load_params(fp):
    with open(fp) as fh:
        param = json.load(fh)

    return param

def create_fp(fp, fname):
    return fp + f'/{fname}.csv'
    

def main(targets):

    # make the clean target
    if 'clean' in targets:
        shutil.rmtree('data/', ignore_errors=True)
        

    # make the data target
    if 'data' in targets:
        cfg = load_params(DATA_PARAMS)
        if not os.path.exists(cfg['outpath']):
            os.makedirs(cfg['outpath'])
        for disease, params in cfg['diseases'].items():
            diseasepath = cfg['outpath'] + f'/{disease}'
            get_gwas_trait(params['train_data'], params['test_data'], cfg['max_p_value'], diseasepath)
        
        
    # make the simulate target
    if 'simulate' in targets:
        cfg = load_params(DATA_PARAMS)
        outpath = cfg['outpath']
        n_samples = cfg['n_samples']
        for disease, params in cfg['diseases'].items():
            diseasepath = cfg['outpath'] + f'/{disease}'
            gwas_fp = create_fp(diseasepath, params['train_data'])
            simulate_data(diseasepath, gwas_fp, n_samples)
        
 
    # make the model target
    if 'model' in targets:
        cfg = load_params(DATA_PARAMS)
        outpath = cfg['outpath']
        simulate_name = 'simulated_data'
        for disease, params in cfg['diseases'].items():
            diseasepath = cfg['outpath'] + f'/{disease}'
            sim_fp = create_fp(diseasepath, simulate_name)
            model_fp = create_fp(diseasepath, params['test_data'])
            build_model(sim_fp, model_fp, diseasepath)
        
        
    # make the test-project target
    if 'test-project' in targets:
        cfg = load_params(TEST_PARAMS)
        if not os.path.exists(cfg['outpath']):
            os.makedirs(cfg['outpath'])
        for disease, params in cfg['diseases'].items():
            diseasepath = cfg['outpath'] + f'/{disease}'
            fps = get_data(params['train_data'], params['test_data'], diseasepath,
                           cfg['max_p_value'], cfg['n_samples'])
            build_model(fps[0], fps[1], diseasepath)
        
        
    # make the run-project target
    if 'run-project' in targets:
        cfg = load_params(DATA_PARAMS)
        if not os.path.exists(cfg['outpath']):
            os.makedirs(cfg['outpath'])
        for disease, params in cfg['diseases'].items():
            diseasepath = cfg['outpath'] + f'/{disease}'
            fps = get_data(params['train_data'], params['test_data'], diseasepath,
                           cfg['max_p_value'], cfg['n_samples'])
            build_model(fps[0], fps[1], diseasepath)

    return


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)