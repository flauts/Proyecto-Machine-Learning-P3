#!/bin/bash
#SBATCH --job-name=bert_experiment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=98G
#SBATCH --output=bert_experiment.out
#SBATCH --error=bert_experiment.err
#SBATCH --partition=data-science
#SBATCH --account=pregrado
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hector.nieto@utec.edu.pe
srun python3 experiment/bert_experiment.py