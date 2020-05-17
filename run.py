#!/usr/bin/env python

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
    return fp+'/{}.csv'.format(fname)
    

def main(targets):

    # make the clean target
    if 'clean' in targets:
        shutil.rmtree('data/',ignore_errors=True)
        

    # make the data target
    if 'data' in targets:
        cfg = load_params(DATA_PARAMS)
        get_gwas_trait(**cfg)
        
        
    # make the simulate target
    if 'simulate' in targets:
        cfg = load_params(DATA_PARAMS)
        outpath = cfg['outpath']
        n_samples = cfg['simulated_data']['n_samples']
        gwas = cfg['simulated_data']['gwas']
        gwas_fp = create_fp(outpath, gwas)
        simulate_data(outpath, gwas_fp, n_samples)
        
 
    # make the model target
    if 'model' in targets:
        cfg = load_params(DATA_PARAMS)
        outpath = cfg['outpath']
        simulate_name = 'simulated_data'
        sim_fp = create_fp(outpath, simulate_name)
        model_name = cfg['model_data']['gwas']
        model_fp = create_fp(outpath, model_name)
        build_model(sim_fp, model_fp, cfg['outpath'])
        
        
    # make the test-project target
    if 'test-project' in targets:
        cfg = load_params(TEST_PARAMS)
        fps = get_data(**cfg, test=True)
        
        build_model(fps[0], fps[1], cfg['outpath'])
        
        
    # make the test-project target
    if 'run-project' in targets:
        cfg = load_params(DATA_PARAMS)
        fps = get_data(**cfg)
        
        build_model(fps[0], fps[1], cfg['outpath'])

    return


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)