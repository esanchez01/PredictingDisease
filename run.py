#!/usr/bin/env python

import sys
import json
import shutil

sys.path.insert(0, 'src') # add library code to path
from etl import *
# from model import driver


# DATA_PARAMS = 'config/01-data.json'
# CLEAN_PARAMS = 'config/02-clean.json'
# MODEL_PARAMS = 'config/03-model.json'

TEST_DATA_PARAMS = 'config/test-01-data.json'
# TEST_CLEAN_PARAMS = 'config/test-02-clean.json'


def load_params(fp):
    with open(fp) as fh:
        param = json.load(fh)

    return param


def main(targets):

    # make the clean target
    if 'clean' in targets:
        shutil.rmtree('data/raw', ignore_errors=True)
        shutil.rmtree('data/cleaned', ignore_errors=True)
        shutil.rmtree('data/out', ignore_errors=True)
        shutil.rmtree('data/test', ignore_errors=True)

    # make the data target
    if 'data' in targets:
        print('running data target')
        cfg = load_params(TEST_DATA_PARAMS)
        get_data_test(**cfg)
        
    # make the process target
    if 'process' in targets:
        print('running process target')
        cfg = load_params(TEST_DATA_PARAMS)
        filter_vcf(**cfg)
        
    # make the test-project target
    if 'test-project' in targets:
        print('running test-project target')
        cfg = load_params(TEST_DATA_PARAMS)
        get_data_test(**cfg)
        filter_vcf(**cfg)

    # make the test target
#     if 'test' in targets:
#         cfg = load_params(TEST_DATA_PARAMS)
#         [_ for _ in get_seasons(**cfg)]

#         cfg = load_params(TEST_CLEAN_PARAMS)
#         [_ for _ in clean_seasons(**cfg)]

    # make the model target
#     if 'model' in targets:
#         cfg = load_params(MODEL_PARAMS)
#         driver(**cfg)

    return


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)