FROM python:3

USER root

# Install conda
RUN apt-get install conda

# Install PLINK2
RUN conda install -c bioconda plink2

# Install requirements.txt
RUN pip install -r requirements.txt

USER $NB_UID