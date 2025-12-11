#!/bin/bash
#SBATCH --job-name=cf_dqn_experiment
#SBATCH --output=logs/outputs/experiment-%A.%a.out
#SBATCH --error=logs/errors/experiment-%A.%a.err
#SBATCH --time=0-03:00:00
#SBATCH --nodes=1                       # number of nodes
#SBATCH --gpus-per-task=4               # number of gpus per node
#SBATCH --cpus-per-task=24              # number of cpus per gpu
#SBATCH --mem=0                         # all memory per node
#SBATCH --ntasks-per-node=1             # crucial - only 1 task per node!
#SBATCH --constraint=80gb               # constraints
#SBATCH --exclude=cn-k[001-002]
#SBATCH --partition=short-unkillable

cd /home/mila/b/baldelld/scratch/RL/CF-DQN

module load anaconda/3
conda activate $SCRATCH/RL/venv

python cleanrl/cf_dqn.py 