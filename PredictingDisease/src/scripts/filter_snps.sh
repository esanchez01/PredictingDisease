#!/bin/sh

# --------------------- #
# Docker
# run using volume
# path will be to forked on your computer
# docker run -it --name genetics2 -v /Users/shannonellis/Desktop/Teaching/DSC180A/Genetic-Variation/project/testdata:/data shanellis/dsc180a-genetics:0.2

# --------------------- #
# establish file structure
mkdir -p ./data/cleaned
mkdir -p ./data/out
mkdir -p ./data/test

# --------------------- #
# filter the VCF file
plink2 \
  --vcf "$1" \
  --make-bed \
  --snps "$2" \
  --maf "$3" \
  --geno "$4" \
  --mind "$5" \
  --recode vcf \
  --out ./data/cleaned/only-gwas-snps

# --------------------- #

"$@"