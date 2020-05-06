FROM python:3

USER root

# Change working directory
WORKDIR /usr/src/app

# Copy requirements.txt
COPY requirements.txt ./

# Install conda
RUN apt-get install conda

# Install PLINK2
RUN conda install -c bioconda plink2

# Install requirements.txt
RUN pip install -r requirements.txt

# Copy application code
COPY . .

USER $NB_UID