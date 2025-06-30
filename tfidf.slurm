#!/bin/bash
#SBATCH --job-name=tfidf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=98G
#SBATCH --output=tfidf.out
#SBATCH --error=tfidf.err
#SBATCH --partition=big-mem
#SBATCH --account=pregrado
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hector.nieto@utec.edu.pe
srun python3 tfidf.py