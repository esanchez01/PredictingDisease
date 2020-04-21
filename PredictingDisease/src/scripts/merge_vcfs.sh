#!/bin/bash
bcftools concat "$1"*.vcf -o "$1"../merged.vcf -O v