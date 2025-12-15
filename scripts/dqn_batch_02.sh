#!/bin/bash
#SBATCH --job-name=dqn_batch_02
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

# GPU 0: Breakout, seed=2
CUDA_VISIBLE_DEVICES=0 python cleanrl/dqn_atari.py \
    --env-id BreakoutNoFrameskip-v4 \
    --track \
    --wandb-project-name Baselines-Breakout \
    --exp-name dqn_seed2 \
    --seed 2 \
    --total-timesteps 10000000 &

# GPU 1: Asterix, seed=2
CUDA_VISIBLE_DEVICES=1 python cleanrl/dqn_atari.py \
    --env-id AsterixNoFrameskip-v4 \
    --track \
    --wandb-project-name Baselines-Asterix \
    --exp-name dqn_seed2 \
    --seed 2 \
    --total-timesteps 10000000 &

# GPU 2: Seaquest, seed=2
CUDA_VISIBLE_DEVICES=2 python cleanrl/dqn_atari.py \
    --env-id SeaquestNoFrameskip-v4 \
    --track \
    --wandb-project-name Baselines-Seaquest \
    --exp-name dqn_seed2 \
    --seed 2 \
    --total-timesteps 10000000 &

# GPU 3: Qbert, seed=2
CUDA_VISIBLE_DEVICES=3 python cleanrl/dqn_atari.py \
    --env-id QbertNoFrameskip-v4 \
    --track \
    --wandb-project-name Baselines-Qbert \
    --exp-name dqn_seed2 \
    --seed 2 \
    --total-timesteps 10000000 &

wait
