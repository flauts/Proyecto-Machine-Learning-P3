#!/bin/bash
#SBATCH --job-name=bert
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=98G
#SBATCH --output=bert.out
#SBATCH --error=bert.err
#SBATCH --partition=big-mem
#SBATCH --account=pregrado
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jose.galvez.p@utec.edu.pe
srun python3 proyecto_bert.py