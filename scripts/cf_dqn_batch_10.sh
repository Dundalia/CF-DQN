#!/bin/bash
#SBATCH --job-name=cf_dqn_batch_10
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

# GPU 0: Qbert, seed=1, n_freq=32, max=2
CUDA_VISIBLE_DEVICES=0 python cleanrl/cf_dqn_atari.py \
    --env-id QbertNoFrameskip-v4 \
    --track \
    --wandb-project-name CF-DQN-Qbert \
    --exp-name cf_dqn_freq32_max2_seed1 \
    --seed 1 \
    --n-frequencies 32 \
    --freq-max 2.0 \
    --collapse-max-w 1.0 \
    --total-timesteps 10000000 &

# GPU 1: Qbert, seed=1, n_freq=32, max=5
CUDA_VISIBLE_DEVICES=1 python cleanrl/cf_dqn_atari.py \
    --env-id QbertNoFrameskip-v4 \
    --track \
    --wandb-project-name CF-DQN-Qbert \
    --exp-name cf_dqn_freq32_max5_seed1 \
    --seed 1 \
    --n-frequencies 32 \
    --freq-max 5.0 \
    --collapse-max-w 1.0 \
    --total-timesteps 10000000 &

# GPU 2: Qbert, seed=1, n_freq=64, max=2
CUDA_VISIBLE_DEVICES=2 python cleanrl/cf_dqn_atari.py \
    --env-id QbertNoFrameskip-v4 \
    --track \
    --wandb-project-name CF-DQN-Qbert \
    --exp-name cf_dqn_freq64_max2_seed1 \
    --seed 1 \
    --n-frequencies 64 \
    --freq-max 2.0 \
    --collapse-max-w 1.0 \
    --total-timesteps 10000000 &

# GPU 3: Qbert, seed=1, n_freq=64, max=5
CUDA_VISIBLE_DEVICES=3 python cleanrl/cf_dqn_atari.py \
    --env-id QbertNoFrameskip-v4 \
    --track \
    --wandb-project-name CF-DQN-Qbert \
    --exp-name cf_dqn_freq64_max5_seed1 \
    --seed 1 \
    --n-frequencies 64 \
    --freq-max 5.0 \
    --collapse-max-w 1.0 \
    --total-timesteps 10000000 &

wait
