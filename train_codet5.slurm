#!/bin/bash
#SBATCH --time=24:00:00              # Job run time (up to 24 hours)
#SBATCH --mem=600G                    # Memory for all cores (600GB)
#SBATCH --gres=gpu:v100:1            # Request 1 GPU (V100)
#SBATCH --cpus-per-task=8            # Number of CPU cores per task
#SBATCH --job-name=codeT5_training_Python_summarization   # Job name
#SBATCH --output=output_%j.txt       # Standard output log file (%j expands to jobID)
#SBATCH --mail-user=ramaalkhouli19@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=def-alkhouli  # Replace with your Compute Canada account

# Load required modules
module load python/3.12
module load cuda/10.1 cudnn/7.6

# Activate your virtual environment
source ~/myenv/bin/activate

# Run the Python training script
python fine_tune_codet5.py
