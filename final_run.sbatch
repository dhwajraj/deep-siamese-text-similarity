#!/bin/bash
#SBATCH --partition=cosmos
#SBATCH --nodelist=blazar
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1,gpu_mem:7000
#SBATCH --mem=4000
#SBATCH --qos=intermediate
#SBATCH --output=sleep.out

source activate my_venv_2_7
#python eval.py
python train.py
