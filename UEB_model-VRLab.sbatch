#!/bin/bash
#SBATCH --job-name=UEB_model
#SBATCH --output=UEB_model-%j.out
#SBATCH --error=UEB_model-%j.err
#SBATCH --nodes=1
#SBATCH --partition=inf
#SBATCH --tasks-per-node=1

# Load Python 3.9.7
module load Python/3.9.7

# Run Python script
srun python3 main.py
