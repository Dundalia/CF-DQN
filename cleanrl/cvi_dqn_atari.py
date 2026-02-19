# CVI-DQN for Atari: Characteristic Value Iteration with CNN feature extractor
# Distributional RL in the frequency domain using characteristic functions
#
# KEY DESIGN: freq_max and collapse_max_w are DECOUPLED.
#   - freq_max = 1.0  → CF grid spans [-1, 1], giving strong gradient signal
#   - collapse_max_w = 0.03 → Q extraction uses only |ω| ≤ 0.03 (no phase wrapping)
#
# Why this works:
#   - The loss trains the CF at ALL frequencies (rich signal: loss ≈ 0.3 vs 0.0001)
#   - Action selection uses ONLY low frequencies where phase doesn't wrap
#   - make_omega_grid concentrates 50% of bins in inner 10% → ~20 bins at |ω|<0.03
#   - Polyak averaging keeps target CFs close → high-ω phase differences stay small
#
# Phase safety: collapse_max_w × Q_max < π → 0.03 × 105 = 3.15 ≈ π
#   Safe for Q_max up to ~100 (Atari with clipped rewards + γ=0.99)
#
# Compare against: cleanrl/c51_atari.py, cleanrl/dqn_atari.py
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.cf import (
    make_omega_grid,
    interpolate_cf_polar,
    collapse_cf_to_mean,
    reward_cf,
    complex_mse_loss,
    complex_huber_loss,
    weighted_complex_mse_loss,
    weighted_complex_huber_loss,
)
from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from cleanrl_utils.buffers import ReplayBuffer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "CVI-DQN"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""

    # CVI-specific arguments
    n_frequencies: int = 128
    """number of frequency grid points (K)"""
    freq_max: float = 1.0
    """maximum frequency W for CF grid — controls gradient signal strength. DECOUPLED from collapse_max_w."""
    collapse_max_w: float = 0.03
    """max |omega| for Q-value extraction. MUST satisfy collapse_max_w * Q_max < π. 0.03*100=3.0<π"""
    sigma_w: float = 0.15
    """Gaussian weighting std for weighted loss types. Smaller → more focus on low-ω (Q-encoding). 0.15 gives strong weight at |ω|<0.3"""
    loss_type: str = "weighted_mse"
    """loss function: complex_mse | complex_huber | weighted_mse | weighted_huber. weighted_mse recommended for Atari."""
    max_grad_norm: float = 10.0
    """gradient clipping norm (0 to disable)"""

    # Target network — Polyak averaging is essential with decoupled freq_max
    # Hard copies cause discontinuous jumps in high-ω CFs
    tau: float = 0.005
    """Polyak averaging coefficient"""
    use_polyak: bool = True
    """whether to use Polyak averaging instead of hard updates (recommended for CVI)"""

    # Standard DQN arguments (matching C51 Atari defaults)
    buffer_size: int = 500000
    """the replay memory buffer size (500K keeps RAM under 16GB for 84x84x4 uint8 frames)"""
    gamma: float = 0.99
    """the discount factor gamma"""
    target_network_frequency: int = 1
    """the timesteps it takes to update the target network (1 = every step for Polyak)"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 80000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.action_space.seed(seed)
        return env
    return thunk


class QNetwork(nn.Module):
    """
    CNN that outputs characteristic functions φ(s,a,ω) for each action.

    Architecture: Nature DQN CNN → 512 → (n_actions × K × 2)
    Output reshaped to complex tensor [batch, n_actions, K].
    φ(0) = 1 is HARD-ENFORCED in the forward pass.
    """
    def __init__(self, env, n_frequencies=128, freq_max=1.0, collapse_max_w=0.03):
        super().__init__()
        self.env = env
        self.n_frequencies = n_frequencies
        self.freq_max = freq_max
        self.collapse_max_w = collapse_max_w
        self.n = env.single_action_space.n

        omegas = make_omega_grid(freq_max, n_frequencies)
        self.register_buffer("omegas", omegas)

        # Find index closest to ω=0 for hard enforcement
        self.zero_idx = torch.argmin(torch.abs(omegas)).item()

        # Nature DQN CNN architecture (same as C51/DQN)
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, self.n * n_frequencies * 2),
        )

        # Initialize last layer to output ~1+0j (identity CF)
        with torch.no_grad():
            last_layer = self.network[-1]
            last_layer.weight.fill_(0.0)
            bias = torch.zeros_like(last_layer.bias)
            bias[0::2] = 1.0  # Real parts = 1.0
            last_layer.bias.copy_(bias)
            # Small noise to break symmetry
            last_layer.weight.add_(torch.randn_like(last_layer.weight) * 1e-4)
            last_layer.bias.add_(torch.randn_like(last_layer.bias) * 1e-4)

    def forward(self, x):
        batch_size = x.shape[0]
        output = self.network(x / 255.0)  # Normalize pixels to [0, 1]
        output = output.view(batch_size, self.n, self.n_frequencies, 2)
        cf = torch.complex(output[..., 0], output[..., 1])

        # HARD ENFORCE φ(0) = 1+0j
        cf[:, :, self.zero_idx] = 1.0 + 0j

        return cf

    def get_action(self, x, action=None):
        cf_all = self.forward(x)
        q_values = collapse_cf_to_mean(self.omegas, cf_all, max_w=self.collapse_max_w)
        if action is None:
            action = torch.argmax(q_values, dim=1)
        batch_indices = torch.arange(len(x), device=x.device)
        cf_for_action = cf_all[batch_indices, action]
        return action, cf_for_action

    def get_all_cf(self, x):
        cf_all = self.forward(x)
        q_values = collapse_cf_to_mean(self.omegas, cf_all, max_w=self.collapse_max_w)
        return cf_all, q_values


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Print operating regime info
    safe_q_max_collapse = np.pi / args.collapse_max_w if args.collapse_max_w > 0 else float('inf')
    theoretical_q_max = 1 / (1 - args.gamma)
    print(f"\n{'='*60}")
    print(f"CVI-DQN Atari — Operating Regime")
    print(f"{'='*60}")
    print(f"  env_id         = {args.env_id}")
    print(f"  loss_type      = {args.loss_type}")
    print(f"  freq_max       = {args.freq_max}  (CF grid range, controls gradient signal)")
    print(f"  collapse_max_w = {args.collapse_max_w}  (Q extraction, controls phase safety)")
    print(f"  K              = {args.n_frequencies}")
    print(f"  Safe Q_max     = {safe_q_max_collapse:.1f}  (need collapse_max_w * Q_max < π)")
    print(f"  Theoretical Q  = {theoretical_q_max:.0f}  (1/(1-γ) with clipped rewards)")
    phase_ok = '✓' if args.collapse_max_w * theoretical_q_max < np.pi else '✗ DANGER'
    print(f"  Phase check    = {args.collapse_max_w} × {theoretical_q_max:.0f} = {args.collapse_max_w * theoretical_q_max:.2f} < π  {phase_ok}")
    print(f"  Polyak         = {args.use_polyak} (τ={args.tau})")
    print(f"{'='*60}\n")

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(
        envs,
        n_frequencies=args.n_frequencies,
        freq_max=args.freq_max,
        collapse_max_w=args.collapse_max_w,
    ).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)
    target_network = QNetwork(
        envs,
        n_frequencies=args.n_frequencies,
        freq_max=args.freq_max,
        collapse_max_w=args.collapse_max_w,
    ).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()
    episode_count = 0

    # Start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ε-greedy action selection
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _ = q_network.get_action(torch.Tensor(obs).to(device))
            actions = actions.cpu().numpy()

        # Execute action
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Record episodic returns
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episode_count += 1
                    ep_ret = info['episode']['r']
                    ep_len = info['episode']['l']
                    print(f"step={global_step}, ep={episode_count}, return={ep_ret}, len={ep_len}")
                    writer.add_scalar("charts/episodic_return", ep_ret, global_step)
                    writer.add_scalar("charts/episodic_length", ep_len, global_step)

        # Store transition
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        # Training
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)

                with torch.no_grad():
                    # Target CF computation
                    next_cf_all, next_q_values = target_network.get_all_cf(data.next_observations)
                    next_actions = torch.argmax(next_q_values, dim=1)
                    batch_indices = torch.arange(args.batch_size, device=device)
                    next_cf_greedy = next_cf_all[batch_indices, next_actions]

                    # Interpolate at γω (discount scaling in frequency domain)
                    omegas = target_network.omegas
                    next_cf_scaled = interpolate_cf_polar(args.gamma * omegas, omegas, next_cf_greedy)

                    # Reward CF: exp(iωr)
                    cf_r = reward_cf(omegas, data.rewards.flatten())

                    # Bellman target: φ_target = φ_r × [φ_future × (1-done) + 1 × done]
                    dones = data.dones.flatten().unsqueeze(-1)
                    cf_future = next_cf_scaled * (1 - dones) + (1.0 + 0j) * dones
                    target_cf = cf_r * cf_future

                # Predicted CF for taken actions
                _, pred_cf = q_network.get_action(data.observations, data.actions.flatten())

                # Loss computation
                if args.loss_type == "complex_mse":
                    loss = complex_mse_loss(pred_cf, target_cf)
                elif args.loss_type == "complex_huber":
                    loss = complex_huber_loss(pred_cf, target_cf)
                elif args.loss_type == "weighted_mse":
                    loss = weighted_complex_mse_loss(pred_cf, target_cf, omegas, sigma_w=args.sigma_w)
                elif args.loss_type == "weighted_huber":
                    loss = weighted_complex_huber_loss(pred_cf, target_cf, omegas, sigma_w=args.sigma_w)
                else:
                    raise ValueError(f"Unknown loss_type: {args.loss_type}")

                # Optimize with gradient clipping
                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
                optimizer.step()

                # Logging (every 100 steps)
                if global_step % 100 == 0:
                    with torch.no_grad():
                        all_cf, all_q_values = q_network.get_all_cf(data.observations)

                    writer.add_scalar("losses/loss", loss.item(), global_step)
                    writer.add_scalar("losses/q_values", all_q_values.mean().item(), global_step)
                    writer.add_scalar("losses/q_values_max", all_q_values.max().item(), global_step)

                    # Phase safety check
                    max_q = abs(all_q_values.max().item())
                    collapse_phase = args.collapse_max_w * max_q
                    writer.add_scalar("debug/collapse_phase", collapse_phase, global_step)
                    writer.add_scalar("debug/phase_safe", float(collapse_phase < np.pi), global_step)

                    # φ(0) check
                    phi0_dev = (torch.abs(pred_cf[:, q_network.zero_idx]) - 1.0).abs().mean().item()
                    writer.add_scalar("debug/phi0_deviation", phi0_dev, global_step)

                    # Gradient norm
                    total_grad_norm = sum(
                        p.grad.data.norm(2).item() ** 2
                        for p in q_network.parameters() if p.grad is not None
                    ) ** 0.5
                    writer.add_scalar("debug/gradient_norm", total_grad_norm, global_step)

                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    writer.add_scalar("charts/epsilon", epsilon, global_step)

                    # === Per-frequency-band diagnostics ===
                    with torch.no_grad():
                        omegas_diag = q_network.omegas
                        low_mask = (omegas_diag.abs() <= 0.1)
                        mid_mask = (omegas_diag.abs() > 0.1) & (omegas_diag.abs() <= 0.4)
                        high_mask = (omegas_diag.abs() > 0.4)

                        # Per-band loss breakdown
                        diff = pred_cf - target_cf
                        loss_per_freq = diff.real**2 + diff.imag**2
                        if low_mask.any():
                            writer.add_scalar("debug/loss_low_omega", loss_per_freq[:, low_mask].mean().item(), global_step)
                        if mid_mask.any():
                            writer.add_scalar("debug/loss_mid_omega", loss_per_freq[:, mid_mask].mean().item(), global_step)
                        if high_mask.any():
                            writer.add_scalar("debug/loss_high_omega", loss_per_freq[:, high_mask].mean().item(), global_step)

                        # CF magnitude by band (should decay at high ω for well-learned CFs)
                        pred_mag = torch.abs(pred_cf)
                        if low_mask.any():
                            writer.add_scalar("debug/cf_mag_low", pred_mag[:, low_mask].mean().item(), global_step)
                        if mid_mask.any():
                            writer.add_scalar("debug/cf_mag_mid", pred_mag[:, mid_mask].mean().item(), global_step)
                        if high_mask.any():
                            writer.add_scalar("debug/cf_mag_high", pred_mag[:, high_mask].mean().item(), global_step)

                        # Q-value spread across actions (key convergence signal)
                        q_spread = (all_q_values.max(dim=1)[0] - all_q_values.min(dim=1)[0]).mean().item()
                        writer.add_scalar("debug/q_spread_across_actions", q_spread, global_step)

                        # Per-action Q-value means
                        for a in range(min(all_q_values.shape[1], 6)):
                            writer.add_scalar(f"debug/q_action_{a}", all_q_values[:, a].mean().item(), global_step)

                        # Target CF statistics
                        target_mag = torch.abs(target_cf)
                        target_phase_at_collapse = torch.angle(target_cf[:, low_mask]).mean().item() if low_mask.any() else 0
                        writer.add_scalar("debug/target_cf_mag_mean", target_mag.mean().item(), global_step)
                        writer.add_scalar("debug/target_phase_low_omega", target_phase_at_collapse, global_step)

                    # === Detailed print every 10K steps ===
                    if global_step % 10000 == 0:
                        with torch.no_grad():
                            q_min_across = all_q_values.min(dim=1)[0].mean().item()
                            q_max_across = all_q_values.max(dim=1)[0].mean().item()
                            low_loss_val = loss_per_freq[:, low_mask].mean().item() if low_mask.any() else 0
                            mid_loss_val = loss_per_freq[:, mid_mask].mean().item() if mid_mask.any() else 0
                            high_loss_val = loss_per_freq[:, high_mask].mean().item() if high_mask.any() else 0
                            low_mag_val = pred_mag[:, low_mask].mean().item() if low_mask.any() else 0
                            mid_mag_val = pred_mag[:, mid_mask].mean().item() if mid_mask.any() else 0
                            high_mag_val = pred_mag[:, high_mask].mean().item() if high_mask.any() else 0
                            # Action selection diversity
                            chosen = all_q_values.argmax(dim=1)
                            acounts = torch.bincount(chosen, minlength=all_q_values.shape[1]).float()
                            aprobs = acounts / acounts.sum()
                            action_ent = -(aprobs * (aprobs + 1e-8).log()).sum().item()

                        # Memory usage
                        try:
                            import psutil
                            ram_gb = psutil.Process().memory_info().rss / 1e9
                        except ImportError:
                            ram_gb = -1
                        gpu_mb = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0

                        print(f"\n{'='*70}")
                        print(f"Step {global_step} / {args.total_timesteps}  (ep={episode_count})")
                        print(f"{'='*70}")
                        print(f"  Q-values:  mean={all_q_values.mean():.4f}  min={q_min_across:.4f}  max={q_max_across:.4f}  spread={q_spread:.5f}")
                        print(f"  Per-action Q: {[f'{all_q_values[:, a].mean():.4f}' for a in range(min(all_q_values.shape[1], 6))]}")
                        print(f"  Loss:  total={loss:.6f}  low_ω={low_loss_val:.6f}  mid_ω={mid_loss_val:.6f}  high_ω={high_loss_val:.6f}")
                        print(f"  CF mag:  low={low_mag_val:.4f}  mid={mid_mag_val:.4f}  high={high_mag_val:.4f}")
                        print(f"  Grad={total_grad_norm:.5f}  phase={collapse_phase:.4f}rad  ε={epsilon:.3f}")
                        print(f"  Action entropy={action_ent:.3f}  (uniform={np.log(all_q_values.shape[1]):.3f})")
                        print(f"  Memory: RAM={ram_gb:.1f}GB  GPU={gpu_mb:.0f}MB  Buffer={rb.size()}/{args.buffer_size}")
                        print(f"  SPS={int(global_step / (time.time() - start_time))}")
                        print(f"{'='*70}")

            # Target network update
            if global_step % args.target_network_frequency == 0:
                if args.use_polyak:
                    for param, target_param in zip(q_network.parameters(), target_network.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                else:
                    target_network.load_state_dict(q_network.state_dict())

    # Evaluation
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save({"model_weights": q_network.state_dict(), "args": vars(args)}, model_path)
        print(f"model saved to {model_path}")

        eval_episodes = 10
        eval_returns = []
        eval_env = gym.make(args.env_id)
        eval_env = NoopResetEnv(eval_env, noop_max=30)
        eval_env = MaxAndSkipEnv(eval_env, skip=4)
        eval_env = EpisodicLifeEnv(eval_env)
        if "FIRE" in eval_env.unwrapped.get_action_meanings():
            eval_env = FireResetEnv(eval_env)
        eval_env = gym.wrappers.ResizeObservation(eval_env, (84, 84))
        eval_env = gym.wrappers.GrayScaleObservation(eval_env)
        eval_env = gym.wrappers.FrameStack(eval_env, 4)

        for ep in range(eval_episodes):
            obs_eval, _ = eval_env.reset()
            done = False
            total_reward = 0.0
            while not done:
                with torch.no_grad():
                    obs_t = torch.Tensor(np.array(obs_eval)).unsqueeze(0).to(device)
                    action, _ = q_network.get_action(obs_t)
                    action = action.cpu().numpy()[0]
                obs_eval, reward, terminated, truncated, _ = eval_env.step(action)
                total_reward += reward
                done = terminated or truncated
            eval_returns.append(total_reward)
        eval_env.close()
        for i, er in enumerate(eval_returns):
            writer.add_scalar("eval/episodic_return", er, i)
        print(f"Eval: mean={np.mean(eval_returns):.2f}, std={np.std(eval_returns):.2f}")

    envs.close()
    writer.close()
