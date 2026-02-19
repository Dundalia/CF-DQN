#!/usr/bin/env python3
"""
CVI Validation Suite v2
=======================
Step-by-step tests that build confidence in the CVI pipeline.
Each test isolates one component, so failures are easy to diagnose.

Usage:
    uv run scripts/cvi_validate.py

Tests (in order of complexity):
  0. collapse_cf_to_mean on known CFs  — does Q-extraction work at all?
  1. Tabular CVI with direct assignment — does Bellman backup + interpolation + collapse work?
  2. Tabular CVI with gradient descent — does optimization converge?
  3. Neural CVI on stateless bandit   — does the neural network learn CFs?
  3b. Neural CVI on 2-state chain    — does neural bootstrapping work?
  4. Neural CVI on FrozenLake         — does it work on a real env?
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cleanrl_utils.cf import (
    make_omega_grid,
    interpolate_cf_polar,
    collapse_cf_to_mean,
    reward_cf,
    complex_mse_loss,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "validation_plots")
os.makedirs(OUT_DIR, exist_ok=True)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


# =====================================================================
# Test 0: collapse_cf_to_mean on known CFs
# =====================================================================
def test_0_collapse():
    """
    Create CFs analytically: phi(w) = exp(iwQ) for known Q.
    Verify collapse extracts Q accurately at different omega_max.
    """
    print("\n" + "="*60)
    print("Test 0: collapse_cf_to_mean on known CFs")
    print("="*60)

    test_cases = [
        # (Q, omega_max, K, collapse_max_w)
        (0.0,   2.0, 128, 2.0),
        (0.5,   2.0, 128, 2.0),
        (1.0,   2.0, 128, 2.0),
        (2.0,   2.0, 128, 2.0),
        (5.0,   2.0, 128, 2.0),
        (10.0,  2.0, 128, 2.0),   # phase at w=2 is 20 rad, wraps
        (1.0,   0.5, 128, 0.5),   # small omega_max
        (100.0, 0.01, 128, 0.01), # very small omega_max for large Q
    ]

    all_pass = True
    for Q, wmax, K, cw in test_cases:
        omegas = make_omega_grid(wmax, K)
        cf = torch.exp(1j * omegas * Q)
        # collapse expects [batch, n_actions, K] or similar
        q_est = collapse_cf_to_mean(omegas, cf.unsqueeze(0).unsqueeze(0), max_w=cw).item()
        err = abs(q_est - Q)
        rel_err = err / max(abs(Q), 1e-8) * 100
        ok = err < 0.1 * max(abs(Q), 0.01)
        status = PASS if ok else FAIL
        print(f"  Q={Q:>7.1f}, w_max={wmax}, cw={cw}: est={q_est:>8.4f}, err={err:.4f} ({rel_err:.1f}%)  {status}")
        if not ok:
            all_pass = False

    return all_pass


# =====================================================================
# Test 1: Tabular CVI with direct assignment (no optimization)
# =====================================================================
def test_1_tabular_direct():
    """
    3-state linear MDP with known Q-values.
    Directly set phi = Bellman target each iteration (no gradient descent).
    Tests: Bellman backup + interpolation + collapse.

    S0 --r=1--> S1 --r=1--> S2 (terminal, r=0)
    True Q: Q(S0)=1+gamma=1.99, Q(S1)=1, Q(S2)=0
    """
    print("\n" + "="*60)
    print("Test 1: Tabular CVI, direct assignment")
    print("="*60)

    gamma = 0.99
    # Use omega_max=1.0: max_phase = 1.0 * 1.99 = 1.99 < pi, safe
    K = 128
    freq_max = 1.0
    omegas = make_omega_grid(freq_max, K)

    transitions = {
        (0, 0): (1, 1.0, False),
        (1, 0): (2, 1.0, False),
        (2, 0): (2, 0.0, True),
    }
    true_q = {(0, 0): 1.0 + gamma * 1.0, (1, 0): 1.0, (2, 0): 0.0}

    # Init CFs to identity
    phi = {key: torch.ones(K, dtype=torch.complex64) for key in true_q}

    q_history = {key: [] for key in true_q}
    n_iters = 20  # should converge in ~3 iters for a 3-state chain

    for iteration in range(n_iters):
        for (s, a), (s_next, r, done) in transitions.items():
            cf_r = reward_cf(omegas, torch.tensor([r])).squeeze(0)
            if done:
                target = cf_r  # terminal: phi = exp(iwr) (here r=0, so identity)
            else:
                phi_next = phi[(s_next, 0)].detach()
                phi_next_scaled = interpolate_cf_polar(
                    gamma * omegas, omegas, phi_next.unsqueeze(0)
                ).squeeze(0)
                target = cf_r * phi_next_scaled

            # Direct assignment: set CF to exact target
            phi[(s, a)] = target.clone()
            # Hard enforce phi(0) = 1
            zero_idx = torch.argmin(torch.abs(omegas))
            phi[(s, a)][zero_idx] = 1.0 + 0j

        # Extract Q-values
        for key in true_q:
            q_est = collapse_cf_to_mean(
                omegas, phi[key].unsqueeze(0).unsqueeze(0), max_w=freq_max
            ).item()
            q_history[key].append(q_est)

    # Plot convergence
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for key, true_val in true_q.items():
        ax.plot(q_history[key], 'o-', label=f"Q{key} -> {true_val:.2f}")
        ax.axhline(y=true_val, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Q-value")
    ax.set_title("Tabular CVI, Direct Assignment (w_max=1.0)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "1_tabular_direct.png"), dpi=150)
    plt.close()
    print(f"  Plot saved: {OUT_DIR}/1_tabular_direct.png")

    # Check final values (should be nearly exact after 3+ iterations)
    all_pass = True
    for key, true_val in true_q.items():
        est = q_history[key][-1]
        err = abs(est - true_val)
        ok = err < 0.05 * max(abs(true_val), 0.01)
        status = PASS if ok else FAIL
        print(f"  Q{key}: est={est:.4f}, true={true_val:.4f}, err={err:.4f}  {status}")
        if not ok:
            all_pass = False

    return all_pass


# =====================================================================
# Test 2: Tabular CVI with gradient descent
# =====================================================================
def test_2_tabular_gradient():
    """
    Same 3-state MDP, but use gradient descent to learn CFs.
    Key: complex_mse_loss uses .mean() which divides gradient by K*batch.
    In the tabular case, each CF bin is independent, so the effective
    per-bin learning rate is lr/(K*batch). We compensate with high lr.

    With K=128 and batch=1:
      effective_lr_per_bin = lr / K
      For convergence in N iters: (1 - lr/K)^N << 1  =>  lr >> K/N

    With lr=5.0, K=128, N=500: (1 - 5/128)^500 = 2e-9. Fully converged.
    """
    print("\n" + "="*60)
    print("Test 2: Tabular CVI, gradient descent (unweighted MSE)")
    print("="*60)

    gamma = 0.99
    K = 128
    freq_max = 1.0
    omegas = make_omega_grid(freq_max, K)
    lr = 5.0  # High lr to compensate for 1/K in .mean()
    n_iters = 500

    transitions = {
        (0, 0): (1, 1.0, False),
        (1, 0): (2, 1.0, False),
        (2, 0): (2, 0.0, True),
    }
    true_q = {(0, 0): 1.0 + gamma * 1.0, (1, 0): 1.0, (2, 0): 0.0}

    phi = {key: torch.ones(K, dtype=torch.complex64) for key in true_q}
    q_history = {key: [] for key in true_q}
    loss_history = []

    for iteration in range(n_iters):
        total_loss = 0.0
        for (s, a), (s_next, r, done) in transitions.items():
            cf_r = reward_cf(omegas, torch.tensor([r])).squeeze(0)
            if done:
                target = cf_r
            else:
                phi_next = phi[(s_next, 0)].detach()
                phi_next_scaled = interpolate_cf_polar(
                    gamma * omegas, omegas, phi_next.unsqueeze(0)
                ).squeeze(0)
                target = cf_r * phi_next_scaled

            pred = phi[(s, a)].clone().requires_grad_(True)
            loss = complex_mse_loss(pred.unsqueeze(0), target.unsqueeze(0).detach())
            loss.backward()

            with torch.no_grad():
                phi[(s, a)] = pred - lr * pred.grad
                zero_idx = torch.argmin(torch.abs(omegas))
                phi[(s, a)][zero_idx] = 1.0 + 0j

            total_loss += loss.item()

        for key in true_q:
            q_est = collapse_cf_to_mean(
                omegas, phi[key].unsqueeze(0).unsqueeze(0), max_w=freq_max
            ).item()
            q_history[key].append(q_est)
        loss_history.append(total_loss)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for key, true_val in true_q.items():
        ax1.plot(q_history[key], label=f"Q{key} -> {true_val:.2f}")
        ax1.axhline(y=true_val, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Q-value")
    ax1.set_title("Q-value Convergence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(loss_history, color='red')
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Loss")
    ax2.set_title("Total Bellman Loss")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Tabular CVI, Gradient Descent (lr={lr}, w_max={freq_max})", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "2_tabular_gradient.png"), dpi=150)
    plt.close()
    print(f"  Plot saved: {OUT_DIR}/2_tabular_gradient.png")

    all_pass = True
    for key, true_val in true_q.items():
        est = q_history[key][-1]
        err = abs(est - true_val)
        ok = err < 0.1 * max(abs(true_val), 0.01)
        status = PASS if ok else FAIL
        print(f"  Q{key}: est={est:.4f}, true={true_val:.4f}, err={err:.4f}  {status}")
        if not ok:
            all_pass = False
    print(f"  Final loss: {loss_history[-1]:.6f}")

    return all_pass


# =====================================================================
# Test 3: Neural CVI on stateless bandit (no bootstrapping)
# =====================================================================
def test_3_neural_bandit():
    """
    Stateless bandit with 2 actions:
      Action 0: reward = 1 (deterministic)
      Action 1: reward = 3 (deterministic)

    True CFs:
      phi_0(w) = exp(iw*1)
      phi_1(w) = exp(iw*3)

    True Q-values: Q(a=0) = 1, Q(a=1) = 3.

    No bootstrapping (single-step episodes), dense reward.
    Isolates whether the neural network can learn CFs at all.
    """
    print("\n" + "="*60)
    print("Test 3: Neural CVI, stateless bandit")
    print("="*60)

    K = 64
    freq_max = 0.8  # max_phase = 0.8 * 3 = 2.4 < pi
    omegas = make_omega_grid(freq_max, K)
    zero_idx = torch.argmin(torch.abs(omegas)).item()
    n_actions = 2
    rewards = [1.0, 3.0]
    lr = 1e-3
    n_steps = 3000
    batch_size = 64

    class BanditCFNet(nn.Module):
        def __init__(self):
            super().__init__()
            # Directly learned CFs per action (no obs input)
            self.cf_params = nn.Parameter(torch.ones(n_actions, K, 2))
            with torch.no_grad():
                self.cf_params[:, :, 0] = 1.0
                self.cf_params[:, :, 1] = 0.0

        def forward(self):
            cf = torch.complex(self.cf_params[:, :, 0], self.cf_params[:, :, 1])
            cf[:, zero_idx] = 1.0 + 0j  # Hard phi(0)=1
            return cf  # [n_actions, K]

    net = BanditCFNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    loss_history = []
    q_history = {a: [] for a in range(n_actions)}

    for step in range(n_steps):
        # Sample random actions and generate rewards
        actions = torch.randint(0, n_actions, (batch_size,))
        rews = torch.tensor([rewards[a] for a in actions], dtype=torch.float32)

        # Target CFs: exp(iwr) (single-step, terminal)
        target_cf = reward_cf(omegas, rews)  # [batch, K]

        # Predicted CFs
        all_cf = net()  # [n_actions, K]
        pred_cf = all_cf[actions]  # [batch, K]

        loss = complex_mse_loss(pred_cf, target_cf.detach())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        optimizer.step()

        loss_history.append(loss.item())

        if step % 100 == 0:
            with torch.no_grad():
                cf = net()
                for a in range(n_actions):
                    q = collapse_cf_to_mean(
                        omegas, cf[a].unsqueeze(0).unsqueeze(0), max_w=freq_max
                    ).item()
                    q_history[a].append(q)

    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    steps_q = list(range(0, n_steps, 100))
    for a in range(n_actions):
        ax1.plot(steps_q, q_history[a], 'o-', label=f"Q(a={a}) -> {rewards[a]:.0f}")
        ax1.axhline(y=rewards[a], color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Q-value")
    ax1.set_title("Q-value Convergence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(loss_history, alpha=0.3, color='red')
    sm = np.convolve(loss_history, np.ones(50)/50, mode='valid')
    ax2.plot(sm, color='red')
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training Loss")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    # Plot learned CFs vs true CFs
    with torch.no_grad():
        cf = net()
    for a in range(n_actions):
        true_cf = torch.exp(1j * omegas * rewards[a])
        ax3.plot(omegas.numpy(), cf[a].real.numpy(), '-', label=f"Re(phi_{a}) learned")
        ax3.plot(omegas.numpy(), true_cf.real.numpy(), '--', alpha=0.5, label=f"Re(phi_{a}) true")
    ax3.set_xlabel("w")
    ax3.set_ylabel("Re(phi(w))")
    ax3.set_title("Learned vs True CFs (real part)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.suptitle("Neural CVI, Stateless Bandit (no bootstrapping)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "3_neural_bandit.png"), dpi=150)
    plt.close()
    print(f"  Plot saved: {OUT_DIR}/3_neural_bandit.png")

    all_pass = True
    for a in range(n_actions):
        est = q_history[a][-1]
        true_val = rewards[a]
        err = abs(est - true_val)
        ok = err < 0.2
        status = PASS if ok else FAIL
        print(f"  Q(a={a}): est={est:.4f}, true={true_val:.1f}, err={err:.4f}  {status}")
        if not ok:
            all_pass = False

    return all_pass


# =====================================================================
# Test 3b: Neural CVI on 2-state chain (bootstrapping test)
# =====================================================================
def test_3b_neural_chain():
    """
    2-state chain MDP, 1 action per state, one-hot input, neural CF network.
    Tests whether a neural network can propagate CFs through bootstrapping.

    S0 --r=1--> S1 --r=1--> terminal
    True Q: Q(S0)=1+gamma=1.99, Q(S1)=1

    Compared to the bandit (Test 3): this adds bootstrapping.
    Compared to FrozenLake (Test 4): no exploration problem, dense rewards.
    """
    print("\n" + "="*60)
    print("Test 3b: Neural CVI, 2-state chain (bootstrapping)")
    print("="*60)

    K = 32
    freq_max = 1.0  # max_phase = 1.0 * 1.99 = 1.99 < pi
    gamma = 0.99
    lr = 1e-3
    n_steps = 5000
    batch_size = 32
    n_states = 2
    n_actions = 1
    omegas = make_omega_grid(freq_max, K)
    zero_idx = torch.argmin(torch.abs(omegas)).item()

    true_q = {0: 1.0 + gamma * 1.0, 1: 1.0}

    class ChainCFNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_states, 32),
                nn.ReLU(),
                nn.Linear(32, n_actions * K * 2),
            )
            with torch.no_grad():
                last = self.net[-1]
                last.weight.fill_(0.0)
                bias = torch.zeros(n_actions * K * 2)
                bias[0::2] = 1.0
                last.bias.copy_(bias)

        def forward(self, x):
            out = self.net(x)
            out = out.view(-1, n_actions, K, 2)
            cf = torch.complex(out[..., 0], out[..., 1])
            cf[:, :, zero_idx] = 1.0 + 0j
            return cf

    def obs_oh(s_batch):
        """One-hot encode a batch of states."""
        x = torch.zeros(len(s_batch), n_states, dtype=torch.float32)
        for i, s in enumerate(s_batch):
            x[i, s] = 1.0
        return x

    net = ChainCFNet()
    target_net = ChainCFNet()
    target_net.load_state_dict(net.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    q_history = {s: [] for s in range(n_states)}
    loss_history = []

    for step in range(n_steps):
        # Generate batch: equal mix of both transitions
        half = batch_size // 2
        states = np.array([0]*half + [1]*half)
        rewards_arr = np.array([1.0]*batch_size, dtype=np.float32)
        next_states = np.array([1]*half + [0]*half)  # S1->terminal uses done flag
        dones = np.array([0.0]*half + [1.0]*half, dtype=np.float32)  # S1 is terminal

        obs_b = obs_oh(states)
        rew_b = torch.tensor(rewards_arr)
        next_b = obs_oh(next_states)
        done_b = torch.tensor(dones)

        with torch.no_grad():
            next_cf_all = target_net(next_b)  # [batch, 1, K]
            next_cf = next_cf_all[:, 0, :]     # [batch, K] (only 1 action)
            next_cf_scaled = interpolate_cf_polar(gamma * omegas, omegas, next_cf)
            cf_r = reward_cf(omegas, rew_b)
            ds = done_b.unsqueeze(-1)
            cf_future = next_cf_scaled * (1 - ds) + (1.0 + 0j) * ds
            target_cf = cf_r * cf_future

        pred_cf_all = net(obs_b)
        pred_cf = pred_cf_all[:, 0, :]  # [batch, K]

        loss = complex_mse_loss(pred_cf, target_cf)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        optimizer.step()

        loss_history.append(loss.item())

        # Hard target update every 50 steps
        if step % 50 == 0:
            target_net.load_state_dict(net.state_dict())

        # Log Q values
        if step % 50 == 0:
            with torch.no_grad():
                for s in range(n_states):
                    x = torch.zeros(1, n_states)
                    x[0, s] = 1.0
                    cf_s = net(x)
                    q = collapse_cf_to_mean(omegas, cf_s, max_w=freq_max).item()
                    q_history[s].append(q)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    steps_q = list(range(0, n_steps, 50))
    for s in range(n_states):
        ax1.plot(steps_q, q_history[s], label=f"Q(S{s}) -> {true_q[s]:.2f}")
        ax1.axhline(y=true_q[s], color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Q-value")
    ax1.set_title("Q-value Convergence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(loss_history, alpha=0.3, color='red')
    if len(loss_history) > 50:
        sm = np.convolve(loss_history, np.ones(50)/50, mode='valid')
        ax2.plot(sm, color='red')
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training Loss")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Neural CVI, 2-State Chain (bootstrapping test)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "3b_neural_chain.png"), dpi=150)
    plt.close()
    print(f"  Plot saved: {OUT_DIR}/3b_neural_chain.png")

    all_pass = True
    for s, true_val in true_q.items():
        est = q_history[s][-1]
        err = abs(est - true_val)
        ok = err < 0.15 * max(abs(true_val), 0.01)
        status = PASS if ok else FAIL
        print(f"  Q(S{s}): est={est:.4f}, true={true_val:.4f}, err={err:.4f}  {status}")
        if not ok:
            all_pass = False

    return all_pass


# =====================================================================
# Test 4: Neural CVI-DQN on FrozenLake-v1
# =====================================================================
def test_4_neural_frozenlake():
    """
    FrozenLake-v1, 4x4, is_slippery=False.
    Q_max = gamma^6 = 0.94 (optimal 6-step path), w_max=1.0 is safe.

    Uses smaller K=32 to reduce output dimensionality (4*32*2=256 vs 4*64*2=512),
    nn.Embedding for states, and 200k steps to allow sufficient exploration.
    """
    import gymnasium as gym

    print("\n" + "="*60)
    print("Test 4: Neural CVI-DQN, FrozenLake-v1 (4x4, no slip)")
    print("="*60)

    K = 32
    freq_max = 1.0  # max_phase = 1.0 * 0.94 = 0.94 < pi
    gamma = 0.99
    lr = 1e-3
    total_steps = 200_000
    learning_starts = 500
    batch_size = 64
    buffer_size = 20_000
    train_freq = 1
    target_update_freq = 200

    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
    n_obs = env.observation_space.n
    n_actions = env.action_space.n
    omegas = make_omega_grid(freq_max, K)
    zero_idx = torch.argmin(torch.abs(omegas)).item()

    class FLCFNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(n_obs, 32)
            self.net = nn.Sequential(
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, n_actions * K * 2),
            )
            with torch.no_grad():
                last = self.net[-1]
                last.weight.fill_(0.0)
                bias = torch.zeros(n_actions * K * 2)
                bias[0::2] = 1.0
                last.bias.copy_(bias)

        def forward(self, x):
            # x: [batch] of integer state indices
            h = self.embed(x)  # [batch, 32]
            out = self.net(h)
            out = out.view(-1, n_actions, K, 2)
            cf = torch.complex(out[..., 0], out[..., 1])
            cf[:, :, zero_idx] = 1.0 + 0j
            return cf

    net = FLCFNet()
    target_net = FLCFNet()
    target_net.load_state_dict(net.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # Simple replay buffer
    buf_obs = np.zeros(buffer_size, dtype=np.int64)
    buf_act = np.zeros(buffer_size, dtype=np.int64)
    buf_rew = np.zeros(buffer_size, dtype=np.float32)
    buf_next = np.zeros(buffer_size, dtype=np.int64)
    buf_done = np.zeros(buffer_size, dtype=np.float32)
    buf_ptr = 0
    buf_len = 0

    episode_returns = []
    losses = []
    ep_ret = 0.0

    obs, _ = env.reset()

    for step in range(total_steps):
        # epsilon-greedy
        eps = max(0.05, 1.0 - step / (total_steps * 0.5))
        if np.random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = torch.tensor([obs], dtype=torch.long)
                cf_all = net(obs_t)
                q_vals = collapse_cf_to_mean(omegas, cf_all, max_w=freq_max)
                action = q_vals.argmax(dim=1).item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store
        i = buf_ptr % buffer_size
        buf_obs[i] = obs
        buf_act[i] = action
        buf_rew[i] = reward
        buf_next[i] = next_obs
        buf_done[i] = float(terminated)
        buf_ptr += 1
        buf_len = min(buf_len + 1, buffer_size)

        ep_ret += reward

        if done:
            episode_returns.append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()
        else:
            obs = next_obs

        # Train
        if step > learning_starts and step % train_freq == 0 and buf_len >= batch_size:
            indices = np.random.randint(0, buf_len, size=batch_size)
            obs_b = torch.tensor(buf_obs[indices], dtype=torch.long)
            act_b = torch.tensor(buf_act[indices], dtype=torch.long)
            rew_b = torch.tensor(buf_rew[indices], dtype=torch.float32)
            next_b = torch.tensor(buf_next[indices], dtype=torch.long)
            done_b = torch.tensor(buf_done[indices], dtype=torch.float32)

            with torch.no_grad():
                next_cf_all = target_net(next_b)
                next_q = collapse_cf_to_mean(omegas, next_cf_all, max_w=freq_max)
                best_acts = next_q.argmax(dim=1)
                batch_idx = torch.arange(batch_size)
                next_cf = next_cf_all[batch_idx, best_acts]

                next_cf_scaled = interpolate_cf_polar(gamma * omegas, omegas, next_cf)
                cf_r = reward_cf(omegas, rew_b)

                dones = done_b.unsqueeze(-1)
                cf_future = next_cf_scaled * (1 - dones) + (1.0 + 0j) * dones
                target_cf = cf_r * cf_future

            pred_cf_all = net(obs_b)
            pred_cf = pred_cf_all[batch_idx, act_b]

            loss = complex_mse_loss(pred_cf, target_cf)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            optimizer.step()
            losses.append(loss.item())

        # Target network update (hard copy)
        if step > learning_starts and step % target_update_freq == 0:
            target_net.load_state_dict(net.state_dict())

    env.close()

    # Evaluate
    eval_env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
    eval_returns = []
    for _ in range(100):
        obs_e, _ = eval_env.reset()
        done_e = False
        ret_e = 0.0
        steps_e = 0
        while not done_e and steps_e < 100:
            with torch.no_grad():
                obs_t = torch.tensor([obs_e], dtype=torch.long)
                cf_all = net(obs_t)
                q_vals = collapse_cf_to_mean(omegas, cf_all, max_w=freq_max)
                act_e = q_vals.argmax(dim=1).item()
            obs_e, rew_e, term_e, trunc_e, _ = eval_env.step(act_e)
            done_e = term_e or trunc_e
            ret_e += rew_e
            steps_e += 1
        eval_returns.append(ret_e)
    eval_env.close()
    eval_success = np.mean(eval_returns)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Episode returns (smoothed)
    if len(episode_returns) > 100:
        smoothed = np.convolve(episode_returns, np.ones(100)/100, mode='valid')
        axes[0].plot(smoothed, color='blue')
    else:
        axes[0].plot(episode_returns, color='blue')
    axes[0].axhline(y=1.0, color='green', linestyle='--', label='Goal')
    axes[0].set_title(f"Episode Returns (eval: {eval_success:.0%})")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    if losses:
        sm_w = min(100, len(losses))
        axes[1].plot(losses, alpha=0.2, color='red')
        if len(losses) > sm_w:
            sm = np.convolve(losses, np.ones(sm_w)/sm_w, mode='valid')
            axes[1].plot(sm, color='red')
        axes[1].set_title("Training Loss")
        axes[1].set_xlabel("Training step")
        axes[1].set_ylabel("Complex MSE")
        if min(losses) > 0:
            axes[1].set_yscale("log")
        axes[1].grid(True, alpha=0.3)

    # Q-values at start
    with torch.no_grad():
        start_cf = net(torch.tensor([0], dtype=torch.long))
        q_start = collapse_cf_to_mean(omegas, start_cf, max_w=freq_max).squeeze().numpy()
    actions_names = ['Left', 'Down', 'Right', 'Up']
    # Optimal from (0,0) is Down
    colors = ['gray', 'blue', 'gray', 'gray']
    axes[2].bar(actions_names, q_start, color=colors)
    axes[2].set_title(f"Q(s0,a); Optimal: Down ~ {gamma**6:.3f}")
    axes[2].set_ylabel("Q-value")
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle(f"CVI-DQN on FrozenLake-v1 ({total_steps//1000}k steps, w_max={freq_max})", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "4_frozenlake_cvi.png"), dpi=150)
    plt.close()

    print(f"  Total episodes: {len(episode_returns)}")
    print(f"  Eval success rate (greedy): {eval_success:.0%}")
    print(f"  Q(s0): {dict(zip(actions_names, [f'{v:.4f}' for v in q_start]))}")
    print(f"  Plot saved: {OUT_DIR}/4_frozenlake_cvi.png")

    ok = eval_success > 0.5
    status = PASS if ok else FAIL
    print(f"  Eval success {eval_success:.0%} {'>' if ok else '<='} 50%:  {status}")
    return ok


# =====================================================================
# Summary & Environment Recommendations
# =====================================================================
def print_recommendations():
    print("\n" + "="*60)
    print("CVI-DQN: Environment Suitability Guide")
    print("="*60)
    print("""
  The KEY constraint for CVI-DQN:

    w_max * Q_max < pi    (guarantees convex loss landscape)

  Since collapse_cf_to_mean uses regression weighted by w^2,
  we need w_max large enough for numerically stable regression,
  but small enough to avoid phase wrapping.

  Rule of thumb: w_max ~ pi / (2 * Q_max)

  Environment               Q_max    w_max    Suitability
  -------------------------  ------   ------   -----------
  FrozenLake (no slip)       ~1       1.0      IDEAL
  FrozenLake (slippery)      ~1       1.0      IDEAL (stochastic!)
  Blackjack-v1               1        1.5      IDEAL (stochastic!)
  CliffWalking-v0            ~10      0.15     GOOD
  Taxi-v3                    ~20      0.08     GOOD
  Acrobot-v1                 ~100     0.015    MODERATE
  CartPole-v1                ~500     0.003    HARD
  MountainCar-v0             ~200     0.008    HARD

  RECOMMENDED PROGRESSION:
  1. FrozenLake-v1 (no slip)  -- validate basic convergence
  2. Blackjack-v1             -- CVI's stochastic advantage over C51
  3. Taxi-v3                  -- moderate complexity
  4. CartPole-v1              -- stress test (w_max=0.003)

  CVI's advantage over C51:
  - C51 discretizes return into N fixed atoms -> resolution limited
  - CVI represents the full CF -> infinite resolution in principle
  - Advantage is greatest when return distribution is complex/multimodal
  - Environments with stochastic transitions/rewards are ideal showcases
""")


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    results = {}

    results["Test 0: collapse"] = test_0_collapse()
    results["Test 1: tabular direct"] = test_1_tabular_direct()
    results["Test 2: tabular gradient"] = test_2_tabular_gradient()
    results["Test 3: neural bandit"] = test_3_neural_bandit()
    results["Test 3b: neural chain"] = test_3b_neural_chain()

    # Only run FrozenLake if earlier tests pass
    if all(results.values()):
        results["Test 4: FrozenLake"] = test_4_neural_frozenlake()
    else:
        print("\n  Skipping Test 4 (FrozenLake) -- earlier tests failed.")
        results["Test 4: FrozenLake"] = None

    print_recommendations()

    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    for name, passed in results.items():
        if passed is None:
            status = "SKIPPED"
        elif passed:
            status = PASS
        else:
            status = FAIL
        print(f"  {name}: {status}")
    print("="*60)

    n_passed = sum(1 for v in results.values() if v is True)
    n_total = sum(1 for v in results.values() if v is not None)
    print(f"  {n_passed}/{n_total} tests passed")
