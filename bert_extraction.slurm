#!/bin/bash
#SBATCH --job-name=bert_extraction
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --output=bert_extraction.out
#SBATCH --error=bert_extraction.err
#SBATCH --partition=data-science
#SBATCH --account=pregrado
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hector.nieto@utec.edu.pe
srun python3 feature_extraction/bert_extraction.py