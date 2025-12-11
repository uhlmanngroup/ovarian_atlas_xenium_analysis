#!/bin/bash
#SBATCH --job-name=nf_sd
#SBATCH --output=nextflow/logs/output.log
#SBATCH --error=nextflow/logs/error.log
#SBATCH --time=24:00:00            
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

source /etc/profile

module load nextflow

SAMPLE_SHEET=workflow/nextflow/sample_sheet.csv

nextflow run workflow/nextflow/spatial_data_pipeline.nf \
    -c workflow/nextflow/nextflow.config \
    --samplesheet $SAMPLE_SHEET \
    --outdir ovarian/final_spatial_data \
    -with-singularity "container.simg"