#!/bin/bash
bcftools concat "$1"*.vcf -o "$1"merged.vcf.gz -O z
bcftools index -f -t "$1"merged.vcf.gz