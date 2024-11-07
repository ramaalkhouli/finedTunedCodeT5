#!/bin/bash

# Load required modules
module load python/3.12
module load cuda/10.1 cudnn/7.6

# Activate your virtual environment
source ~/myenv/bin/activate

# Run the Python training script
python fined_tune_codet5.py
