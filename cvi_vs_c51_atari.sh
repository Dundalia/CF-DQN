#!/bin/bash
#SBATCH --job-name=cvi-vs-c51-atari
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=4      # 4 CPUs per GPU: env stepping is CPU-bound, keeps GPU fed
#SBATCH --ntasks=1
#SBATCH --array=0-11%12       # 12 independent tasks, each gets its own GPU (12 GPUs in parallel)
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mem-per-cpu=4G      # 4 CPUs × 4G = 16G RAM per task (replay buffer ~7G + overhead)
#SBATCH --time=24:00:00

################################################################################
# CVI-DQN vs C51 Atari Benchmark
################################################################################
# 2 algorithms × 2 envs × 3 seeds = 12 total runs (all parallel)
#
# Array index mapping:
#   0- 5 → c51_atari.py     (0-2: Pong seeds 1-3 | 3-5: Breakout seeds 1-3)
#   6-11 → cvi_dqn_atari.py (6-8: Pong seeds 1-3 | 9-11: Breakout seeds 1-3)
#
# Submit: sbatch cvi_vs_c51_atari.sh
################################################################################

ALGOS=(
    "cleanrl/c51_atari.py"
    "cleanrl/c51_atari.py"
    "cleanrl/c51_atari.py"
    "cleanrl/c51_atari.py"
    "cleanrl/c51_atari.py"
    "cleanrl/c51_atari.py"
    "cleanrl/cvi_dqn_atari.py"
    "cleanrl/cvi_dqn_atari.py"
    "cleanrl/cvi_dqn_atari.py"
    "cleanrl/cvi_dqn_atari.py"
    "cleanrl/cvi_dqn_atari.py"
    "cleanrl/cvi_dqn_atari.py"
)

ENV_IDS=(
    "PongNoFrameskip-v4"
    "PongNoFrameskip-v4"
    "PongNoFrameskip-v4"
    "BreakoutNoFrameskip-v4"
    "BreakoutNoFrameskip-v4"
    "BreakoutNoFrameskip-v4"
    "PongNoFrameskip-v4"
    "PongNoFrameskip-v4"
    "PongNoFrameskip-v4"
    "BreakoutNoFrameskip-v4"
    "BreakoutNoFrameskip-v4"
    "BreakoutNoFrameskip-v4"
)

SEEDS=(1 2 3 1 2 3 1 2 3 1 2 3)

SCRIPT=${ALGOS[$SLURM_ARRAY_TASK_ID]}
ENV_ID=${ENV_IDS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "CVI-DQN vs C51 — Atari Benchmark"
echo "=========================================="
echo "Task:     ${SLURM_ARRAY_TASK_ID} / 11"
echo "Script:   ${SCRIPT}"
echo "Env:      ${ENV_ID}"
echo "Seed:     ${SEED}"
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Host:     $(hostname)"
echo "GPU:      ${CUDA_VISIBLE_DEVICES}"
echo "Start:    $(date)"
echo "=========================================="

# uv writes to the shared .venv — serialize setup across all array tasks to
# avoid race conditions (corrupted numpy, missing RECORD files, Remote I/O
# error 121 from scratch filesystem overload when 12 tasks write concurrently).
export UV_LINK_MODE=copy
SETUP_LOCK="${SLURM_SUBMIT_DIR}/.venv_setup.lock"
(
    flock -x 9
    echo "[Task ${SLURM_ARRAY_TASK_ID}] Acquiring lock — setting up environment..."
    uv sync --frozen --quiet
    uv pip install ".[atari]" --quiet
    echo "[Task ${SLURM_ARRAY_TASK_ID}] Environment ready — releasing lock."
) 9>"${SETUP_LOCK}"

# --no-sync: skip per-task re-sync, the locked step above already ensured
# the venv is correct. This prevents any concurrent writes to .venv at runtime.
srun uv run --no-sync python ${SCRIPT} \
    --env-id ${ENV_ID} \
    --seed ${SEED} \
    --total-timesteps 10000000 \
    --wandb-project-name CVI-DQN \
    --track

echo "Task ${SLURM_ARRAY_TASK_ID} completed"
echo "End:      $(date)"
echo "=========================================="
