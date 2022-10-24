#!/bin/bash
#SBATCH --job-name=UEB_model
#SBATCH --output=UEB_model-%j.out
#SBATCH --error=UEB_model-%j.err
#SBATCH --nodes=1
#SBATCH --partition=small
#SBATCH --tasks-per-node=1

# Load Python 3.9.7
#module load Python/3.9.7
source /home/rosalie/venv_UEB_model/bin/activate

export UEB_MODEL_INPUT_DIR= '/net/labdata/rosalie/UEB_model'
#export UEB_MODEL_INPUT_DIR_knmi = /net/labdata/rosalie/UEB_model
# Run Python script
srun python3 main.py
