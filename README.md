# DSC180B
Data Science Capstone Project: Predicting Disease Risk Through Machine Learning

## Usage Instructions

In order to use the different components of this project, please run `python run.py` along with a target of your choice:

* `clean`: Cleans the data directory
* `data`: Downloads the data from GWAS Catalog according to data-params.json
* `simulate-one`: Simulates a SNP population for the training GWAS
* `simulate-both`: Simulates a SNP population for both the training GWAS and the test GWAS
* `model`: 
   * If there is no simulated data for the test GWAS: 
          Splits the training GWAS simulated data into a train and test subset. Model is trained on the training subset, filtered to only contain SNPs also present in the test GWAS, in order to simulate sampling. The model is then tested on the test subset and results are reported (and saved).
   * If there is simulated data for the test GWAS (run via `simulate-both` target):
          Model is trained on simulated data (filtered to contain SNPs present in both GWAS's) from the training GWAS. Model is tested on simulated data from the test GWAS and results are reported (and saved).
* `test-project`: Tests project using test data
* `run-project`: Runs entire project according to cofig files

## Description of Contents

The project consists of these portions:
```
PROJECT
├── config
│   ├── data-params.json
│   ├── env.json
│   ├── model-params.json
│   └── test-params.json
├── notebooks
│   ├── Build_Model.ipynb
│   ├── Simulate_Data.ipynb
│   └── Visualize_Data.ipynb
├── src
│   ├── etl.py
│   ├── model.py
│   └── visualize_data.py
├── testdata
│   ├── alzheimer's
│   ├── coronary_artery
│   └── diabetes_type1_melittus
├── .gitignore
├── README.md
├── requirements.txt
└── run.py
```

### `root`

* `run.py`: Python script to run main command.

### `src`

* `etl.py`: Library code that executes tasks useful for getting data and transforming it into a machine-learning-ready format.

* `model.py`: Library code that builds and tests multiple models, and reports the results.

* `visualize_data.py`: Library code that generates a variety of visualization that are useful for analysis.

### `config`

* `data-params.json`: Parameters for downloading data from the GWAS Catalog and preparing for model building

* `env.json`: Environment information

* `model-params.json`: Contains the (sklearn) models that are tested, and the parameters to use for each model.

* `test-params.json`: Parameters for preparing test data for model building

### `testdata`

This directory contains two summary data files from the GWAS catalog for different diseases, one is used for building a training set and the other is use for a test set. Which is which can be found in the `data-params.json` configuration file (and changed).

* `alzheimer's`: Contains two summary statistics CSV's from GWAS studies on Alzheimer's Disease.

* `coronary_artery`: Contains two summary statistics CSV's from GWAS studies on Coronary Artery Disease.

* `diabetes_type1_melittus`: Contains two summary statistics CSV's from GWAS studies on Diabetes Type I.
