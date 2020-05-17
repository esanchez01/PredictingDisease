# DSC180B
Data Science Senior Project: Predicting Disease From Genetic Variation

## Usage Instructions

* Description of targets and using `run.py`

## Description of Contents

The project consists of these portions:
```
PROJECT
├── config
│   ├── data-params.json
│   ├── test-params.json
│   └── env.json
├── notebooks
│   ├── Build_Model.ipynb
│   ├── Simulate_Data.ipynb
│   └── Visualize_Data.ipynb
├── src
│   ├── etl.py
│   ├── model.py
|   └── visualize_data.py
├── testdata
|   └── gwas
├── .gitignore
├── README.md
├── requirements.txt
└── run.py
```

### `root`

* `run.py`: Python script to run main command, with the following targets:
    * `clean`: Cleans the data directory
    * `data`: Gets the data from GWAS Catalog according to data-params.json
    * `simulate`: Simulates SNP population using ingested GWAS Catalog data
    * `model`: Constructs and tests model on ingesting and wrangled data
    * `test-project`: Tests project using test data
    * `run-project`: Runs entire project according to cofig files

### `src`

* `etl.py`: Library code that executes tasks useful for getting data and transforming it into a machine-learning-ready format.

* `model.py`: Library code that builds and tests a Support Vector Machine given data.

* `visualize_data.py`: Library code that generates a variety of visualization that are useful for analysis.

### `config`

* `data-params.json`: Parameters for downloading data from the GWAS Catalog and preparing for model building

* `test-params.json`: Parameters for preparing test data for model building

* `env.json`: Environment information

### `references`

* Data Dictionaries, references to external sources
