# CF-DQN Setup Guide

This repository is a fork of [CleanRL](https://github.com/vwxyzjn/cleanrl) for implementing Characteristic Value Iteration (CVI) and, in control settings, CF-DQN. The goal is to model return distributions in the frequency domain using Characteristic Functions rather than directly in value space, avoiding the projection and discretization biases inherent to categorical methods like C51.

## Installation on Mila Cluster

```bash
# Load Anaconda module
module load anaconda/3
conda deactivate

# Create virtual environment
conda create --prefix=./venv python=3.10
conda activate ./venv 

# Install uv package manager and dependencies
pip install uv 
uv pip install .
uv pip install ".[atari]"
```

## Quick Start

### Run Existing Algorithms

```bash
# C51 on Atari Breakout
python cleanrl/c51_atari.py --env-id BreakoutNoFrameskip-v4

# DQN on Atari
python cleanrl/dqn_atari.py --env-id PongNoFrameskip-v4
```

### Capture Training Videos

```bash
python cleanrl/c51_atari.py --env-id BreakoutNoFrameskip-v4 --capture_video
```
Videos saved to `videos/{run_name}/`

### Visualize with TensorBoard

Training metrics are automatically logged to `runs/`:

```bash
tensorboard --logdir runs
```

Open `http://localhost:6006` in your browser.

**Key Metrics:**
- `charts/episodic_return` - Episode rewards (main performance indicator)
- `charts/SPS` - Training speed (steps per second)
- `losses/q_values` - Average Q-values
- `losses/loss` - Training loss

### Experiment Tracking with Weights & Biases

```bash
wandb login  # First time only
python cleanrl/c51_atari.py \
    --track \
    --wandb-project-name CF-DQN \
    --capture_video
```

## Developing CF-DQN

We'll implement CF-DQN by starting with the existing DQN implementation and modifying it to work with Characteristic Functions in the frequency domain.

**Suggested workflow:**

1. **Create a development branch:**
```bash
git checkout -b feature/your-name
```

2. **Start by editing the DQN script:**
```bash
# Base CF-DQN on the existing DQN implementation
cp cleanrl/dqn_atari.py cleanrl/cf_dqn_atari.py
```

3. **Implement the CF-DQN modifications:**

4. **Test and iterate:**
```bash
python cleanrl/cf_dqn_atari.py --env-id BreakoutNoFrameskip-v4 --total-timesteps 100000
```

## Resources

- **Original CleanRL Docs**: https://docs.cleanrl.dev/
- **Our Fork**: https://github.com/Dundalia/CF-DQN
- **Upstream CleanRL**: https://github.com/vwxyzjn/cleanrl
- **CleanRL Paper**: [JMLR 2022](http://jmlr.org/papers/v23/21-1342.html)