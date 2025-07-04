#!/bin/bash
#SBATCH --job-name=roberta_extraction
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --output=roberta_extraction.out
#SBATCH --error=roberta_extraction.err
#SBATCH --partition=data-science
#SBATCH --account=pregrado
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hector.nieto@utec.edu.pe
srun python3 feature_extraction/roberta_extraction.py