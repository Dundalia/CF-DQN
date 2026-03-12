import os
import math
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

from collections import deque

from cleanrl_utils.buffers import ReplayBuffer

from cleanrl.cvi_utils import create_three_density_grid, polar_interpolation, gaussian_collapse_q_values, safe_collapse_q_values, create_uniform_grid, ifft_collapse_q_values, get_cleaned_target_cf


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
    wandb_project_name: str = "cleanRL"
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
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    K: int = 128
    """the number of frequency grid points"""
    w: float = 1.0
    """the frequency range [-W, W] for the grid construction during training"""
    n_collapse_pairs: int = 1
    """number of innermost symmetric frequency pairs used to collapse CF → Q-value; MUST satisfy 2*omega_k*Q_max < pi for all pairs used"""
    q_max_bound: float = 200.0
    """upper bound on expected Q-values for phase-safety checks (CartPole gamma=0.99: true Q_max~100, 2x margin)"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """the soft update coefficient for Polyak target network updates"""
    target_network_frequency: int = 1000
    """the frequency at which the target network is hard-updated (0 to disable, use Polyak only)"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""
    max_grad_norm: float = 10.0
    """the maximum gradient norm for clipping"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk


class CF_QNetwork(nn.Module):
    def __init__(self, envs, actual_grid_size):
        super().__init__()
        self.action_dim = envs.single_action_space.n
        self.K = actual_grid_size 
        self.zero_idx = actual_grid_size // 2  # Center of the symmetric grid
        
        self.network = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
        )
        self.cf_head = nn.Linear(84, self.action_dim * self.K * 2)
        
    def forward(self, x):
        features = self.network(x)
        out = self.cf_head(features)
        
        out = out.view(out.shape[0], self.action_dim, self.K, 2)
        V_complex = torch.complex(out[..., 0], out[..., 1])
        
        # Hard normalization to ensure V(0) = 1+0j is always respected.
        # Divide by the value at omega=0 so that the CF is valid by construction.
        # This is mathematically exact: phi(0) = E[e^{i*0*G}] = 1.
        V_at_zero = V_complex[..., self.zero_idx : self.zero_idx + 1]
        V_valid = V_complex / (V_at_zero + 1e-8)
        
        return V_valid

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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    
    #! Init CF-Q-Network and Grid
    recent_returns = deque(maxlen=500)
    #omega_grid = create_uniform_grid(K=args.K, W=args.w, device=device)
    omega_grid = create_three_density_grid(K=args.K, W=args.w, device=device)
    actual_grid_size = len(omega_grid)

    q_network = CF_QNetwork(envs, actual_grid_size=actual_grid_size).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)
    target_network = CF_QNetwork(envs, actual_grid_size=actual_grid_size).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()
    episode_count = 0  # Track total number of completed episodes

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            #! CVI action selection
            with torch.no_grad():
                V_complex_all = q_network(torch.Tensor(obs).to(device))
                q_values = safe_collapse_q_values(omega_grid, V_complex_all, q_max_hint=args.q_max_bound)
                actions = torch.argmax(q_values, dim=1).cpu().numpy()
            
            #* C51 action selection for reference
            # actions, pmf = q_network.get_action(torch.Tensor(obs).to(device))
            # actions = actions.cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episode_count += 1
                    episode_return = info['episode']['r']
                    episode_length = info['episode']['l']
                    print(f"global_step={global_step}, episode={episode_count}, episodic_return={episode_return}, episodic_length={episode_length}")
                    writer.add_scalar("charts/episodic_return", episode_return, global_step)
                    writer.add_scalar("charts/episodic_length", episode_length, global_step)
                    recent_returns.append(episode_return)
                    writer.add_scalar("charts/moving_avg_return", np.mean(recent_returns), global_step)
                    
                    # Log return by episode count for fair comparison across algorithms
                    writer.add_scalar("charts/episodic_return_by_episode", episode_return, episode_count)
                    
                    # Log return at every 100th episode for milestone tracking
                    if episode_count % 100 == 0:
                        writer.add_scalar("charts/episodic_return_per_100_episodes", episode_return, episode_count)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        #! CVI logic
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                
                with torch.no_grad():
                    # 1. Get target CFs for all actions
                    target_V_complex_all = target_network(data.next_observations)

                    # 2. Double DQN: online network SELECTS the greedy next action,
                    #    target network EVALUATES it. Decouples selection from evaluation,
                    #    breaking the positive feedback loop that causes Q overestimation.
                    online_V_next_all = q_network(data.next_observations)
                    online_Q_next = safe_collapse_q_values(omega_grid, online_V_next_all, q_max_hint=args.q_max_bound)
                    next_actions = torch.argmax(online_Q_next, dim=1)  # selected by online network
                    
                    # 3. Select the CF of the greedy action (evaluated by target network)
                    batch_idx = torch.arange(args.batch_size, device=device)
                    target_V_next = target_V_complex_all[batch_idx, next_actions]
                    
                    # 4. Handle terminal states 
                    gammas = args.gamma * (1 - data.dones)
                    
                    # 5. Interpolate at scaled frequencies
                    interp_V = polar_interpolation(omega_grid, target_V_next, gammas)
                    # 6. Apply reward rotation: e^{i * w * R}
                    reward_rotation = torch.exp(1j * omega_grid.view(1, -1) * data.rewards)
                    # 7. Final Bellman Target
                    y_target = reward_rotation * interp_V 

                current_V_complex_all = q_network(data.observations)
                current_V = current_V_complex_all[batch_idx, data.actions.flatten()]
                
                # Weighted MSE Loss in Frequency Domain with Gaussian Weights
                # sigma controls how much of the spectrum contributes to the loss.
                # With W=1.0, sigma=0.3 covers the useful Q-encoding band while
                # providing enough gradient signal across the grid.
                sigma = 0.3
                weights = torch.exp(-(omega_grid ** 2) / (2 * sigma ** 2))
                weights = weights / weights.sum()
                unweighted_mse = torch.abs(current_V - y_target) ** 2
                
                weighted_mse = torch.sum(weights.view(1, -1) * unweighted_mse, dim=1)
                loss = torch.mean(weighted_mse)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/loss", loss.item(), global_step)
                    
                    current_Q_all = safe_collapse_q_values(omega_grid, current_V_complex_all, q_max_hint=args.q_max_bound)
                    current_Q_taken = current_Q_all[batch_idx, data.actions.flatten()]
                    
                    writer.add_scalar("losses/q_values", current_Q_taken.mean().item(), global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    
                    with torch.no_grad():
                        target_V_diag = target_network(data.observations)
                        target_Q_diag = safe_collapse_q_values(omega_grid, target_V_diag, q_max_hint=args.q_max_bound)
                        target_Q_taken_diag = target_Q_diag[batch_idx, data.actions.flatten()]
                        writer.add_scalar("diagnostics/target_q_values", target_Q_taken_diag.mean().item(), global_step)
                        
                        online_target_diff = (current_Q_all - target_Q_diag).abs().mean()
                        writer.add_scalar("diagnostics/q_online_target_diff", online_target_diff.item(), global_step)
                        
                        # Action gap: difference between best and second-best Q-value
                        q_sorted = current_Q_all.sort(dim=1, descending=True).values
                        action_gap = (q_sorted[:, 0] - q_sorted[:, 1]).mean()
                        writer.add_scalar("diagnostics/action_gap", action_gap.item(), global_step)
                        
                        # Phase safety check: max phase at collapse pairs
                        zero_idx_diag = len(omega_grid) // 2
                        pair1_omega = omega_grid[zero_idx_diag + 1].item()
                        max_q_est = current_Q_all.abs().max().item()
                        max_phase = 2 * pair1_omega * max_q_est
                        writer.add_scalar("diagnostics/max_phase_pair1", max_phase, global_step)
                        writer.add_scalar("diagnostics/phase_safe", float(max_phase < math.pi), global_step)
                        writer.add_scalar("diagnostics/max_q_magnitude", max_q_est, global_step)
                        
                        # Log how many collapse pairs are being used
                        max_omega_safe = math.pi / (2.0 * args.q_max_bound)
                        n_safe = sum(1 for k in range(1, zero_idx_diag) if omega_grid[zero_idx_diag + k].item() < max_omega_safe)
                        n_safe = max(n_safe, 1)
                        writer.add_scalar("diagnostics/n_safe_pairs", n_safe, global_step)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
                optimizer.step()
                
            # Target network update
            if args.target_network_frequency > 0 and global_step % args.target_network_frequency == 0:
                # Hard update: periodically copy online -> target
                target_network.load_state_dict(q_network.state_dict())
            elif args.target_network_frequency == 0:
                # Soft Polyak update every gradient step
                for target_param, param in zip(target_network.parameters(), q_network.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1.0 - args.tau) * target_param.data)
                
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        model_data = {
            "model_weights": q_network.state_dict(),
            "args": vars(args),
        }
        torch.save(model_data, model_path)
        print(f"model saved to {model_path}")
        
        
        #! We might need to evaluate in the near future, but for now we can skip!
        # from cleanrl_utils.evals.c51_eval import evaluate
        # episodic_returns = evaluate(
        #     model_path,
        #     make_env,
        #     args.env_id,
        #     eval_episodes=10,
        #     run_name=f"{run_name}-eval",
        #     Model=CF_QNetwork,
        #     device=device,
        #     epsilon=args.end_e,
        # )
        # for idx, episodic_return in enumerate(episodic_returns):
        #     writer.add_scalar("eval/episodic_return", episodic_return, idx)

        # if args.upload_model:
        #     from cleanrl_utils.huggingface import push_to_hub

        #     repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
        #     repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
        #     push_to_hub(args, episodic_returns, repo_id, "CVI", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
