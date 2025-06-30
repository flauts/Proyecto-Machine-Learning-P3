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
pip install pandas transformers scikit-learn joblib
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. Submit SLURM jobs
echo "Submitting tfidf.slurm job..."
sbatch tfidf.slurm

echo "Submitting bert.slurm job..."
sbatch bert.slurm

echo "All jobs submitted!"
