#!/bin/bash
#SBATCH --job-name=bert_features
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --output=bert.out
#SBATCH --error=bert.err
#SBATCH --partition=data-science
#SBATCH --account=pregrado
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hector.nieto@utec.edu.pe
srun python3 bert_extraction.py