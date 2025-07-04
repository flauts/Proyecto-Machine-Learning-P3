#!/bin/bash
#SBATCH --job-name=tfidf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=tfidf.out
#SBATCH --error=tfidf.err
#SBATCH --partition=big-mem
#SBATCH --account=pregrado
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hector.nieto@utec.edu.pe
srun python3 feature_extraction/tfidf_extraction.py