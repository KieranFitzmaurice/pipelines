#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32g
#SBATCH -t 12:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=kieranf@email.unc.edu

cd /work/users/k/i/kieranf/projects/pipelines
module purge
module load anaconda/2023.03
conda activate /proj/characklab/flooddata/NC/conda_environments/envs/flood_v1
python3.11 download_data.py