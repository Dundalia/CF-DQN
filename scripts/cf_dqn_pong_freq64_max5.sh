#!/bin/bash
#SBATCH --job-name=cf_dqn_pong_f64_w5
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

python cleanrl/cf_dqn_atari.py \
    --env-id PongNoFrameskip-v4 \
    --track \
    --wandb-project-name Pong \
    --exp-name cf_dqn_freq64_max5 \
    --n-frequencies 64 \
    --freq-max 5.0 \
    --collapse-max-w 2.0 \
    --total-timesteps 10000000
