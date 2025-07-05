#!/bin/bash
#SBATCH --job-name=tfidf_experiment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=tfidf_experiment.out
#SBATCH --error=tfidf_experiment.err
#SBATCH --partition=big-mem
#SBATCH --account=pregrado
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hector.nieto@utec.edu.pe
srun python3 experiment/tfidf_experiment.py