#!/bin/bash

# Fail on error
set -e

# 1. Create and activate Python virtual environment
echo "Creating virtual environment..."
python3 -m venv env
source env/bin/activate

# 2. Upgrade pip and install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Submit SLURM jobs
echo "Submitting tfidf.slurm job..."
sbatch tfidf_experiment.slurm

echo "Submitting bert.slurm job..."
sbatch bert_experiment.slurm

echo "All jobs submitted!"
