#!/bin/bash
#SBATCH --partition=cosmos
#SBATCH --nodelist=quasar
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:1,gpu_mem:7000
#SBATCH --mem=4000
#SBATCH --qos=intermediate

source activate my_venv_2_7
#python eval.py
#python train.py --hidden_dim 10
#python train.py --hidden_dim 20
#python train.py --hidden_dim 30
python train.py --hidden_dim 50

