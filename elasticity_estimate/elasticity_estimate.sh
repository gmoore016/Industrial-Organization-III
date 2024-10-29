#!/bin/bash
#SBATCH --job-name=movies
#SBATCH --output=elasticity.out
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=normal

ml load python/3.11

python code/estimate_elasticity.py