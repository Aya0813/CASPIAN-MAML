#!/bin/bash
#SBATCH --job-name=og_ft_eval

#SBATCH -o sbatch_output_files/job_June16_MAML_CASPIAN_og_ft_eval.%J.out
#SBATCH -e sbatch_output_files/job_June16_MAML_CASPIAN_og_ft_eval.%J.err

#SBATCH --time=47:59:59
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G     # Optional: adjust if needed

# Env setup 
module purge

#Activating conda
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate tf-caspian-maml

#Your application commands go here

echo "Running script at $(date)"
# python -u MAML_Performance_Evaluation.py
# python -u v2_MAML_SFAD_Training.py
python -u MAML_Performance_Evaluation.py

echo "Finished at $(date)"