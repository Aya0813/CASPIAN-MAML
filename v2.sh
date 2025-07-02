#!/bin/bash
#SBATCH --job-name=v2-t2-eval


#SBATCH -o sbatch_output_files/job.%J.out
#SBATCH -e sbatch_output_files/job.%J.err

#SBATCH --time=120:00:00
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=100GB     # Optional: adjust if needed

# Env setup 
module purge

#Activating conda
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate tf-caspian-maml

#Your appication commands go here

echo "Running script at $(date)"
python -u v1_MAML_Performance_Evaluation.py
# python -u v2_opt_MAML_SFAD_Training.py
# python -u v2_MAML_SFAD_FSL.py

echo "Finished at $(date)"

