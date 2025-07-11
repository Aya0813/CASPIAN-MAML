#!/bin/bash
#SBATCH --job-name=v2-val


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

# mkdir -p ./data/batched_data/SF/

#Your appication commands go here
echo "Running script at $(date)"

# python -u v2_opt_MAML_SFAD_Training.py
python -u v2_opt_MAML_SFAD_FSL.py
# python -u v1_MAML_Performance_Evaluation.py

echo "Finished at $(date)"


# Move .out file for persistent record
# mv sbatch_output_files/job.${SLURM_JOB_ID}.out ./data/batched_data/SF/run_log_${SLURM_JOB_ID}.txt

