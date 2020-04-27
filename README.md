# DSC180B
Data Science Senior Project: Predicting Disease From Genetic Variation

## Usage Instructions

* Description of targets and using `run.py`

## Description of Contents

The project consists of these portions:
```
PROJECT
├── .env
├── .gitignore
├── README.md
├── config
│   ├── download-1000-genomes-data.json
│   ├── filter-merge-1000-genomes-data.json
│   ├── test-1000-genomes-data.json
│   └── test-01-data.json
├── references
│   └── .gitkeep
├── requirements.txt
├── run.py
├── src
│   ├── download_data.py
│   ├── etl.py
│   ├── process_data.py
└── └── read_data.py
```

### `root`

* `run.py`: Python script to run main command, with the following targets:
    * `clean`: Cleans the data directory
    * `data`: Gets the test data
    * `process`: Filters the data to only contain SNPs for the specified disease
    * `test-project`: Gets the test data and filters it, to prepare for machine learning
    * `download-1000-genomes`: Downloads, filters, and merges the 1000 genomes data into data/1000_genomes/
    * `test-1000-genomes`: Prepares VCF for machine learning and builds a proof-of-concept Logistic Regressor on it

### `src`

* `download_data.py`: Library code that downloads data from an FTP server.

* `etl.py`: Library code that executes tasks useful for getting data and transforming it into a machine-learning-ready format.

* `process_data.py`: Library code that builds a Logistic Regressor given data.

* `read_data.py`: Library code that reads VCF data into a DataFrame to prepare for machine learning.

### `config`

* `download-1000-genomes-data.json`: Parameters for downloading data from the 1000 genomes data portal, downloads into data/1000_genomes/

* `filter-merge-1000-genomes-data.json`: Parameters for filtering 1000 genomes VCF files and merging them into one

* `test-1000-genomes-data.json`: Parameters for generating a randomized model on the 1000 genomes data (as proof of concept)
  
* `test-params.json`: parameters for running small process on small
  test data.

### `references`

* Data Dictionaries, references to external sources
