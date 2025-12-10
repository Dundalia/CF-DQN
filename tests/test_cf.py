# tests/test_cf.py
# Unit tests for cleanrl_utils/cf.py
# Verifies shapes and compares against numpy reference from cvi_rl

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pytest

from cleanrl_utils.cf import (
    make_omega_grid,
    interpolate_cf_polar,
    collapse_cf_to_mean,
    reward_cf,
    unwrap_phase,
    complex_mse_loss,
)


class TestMakeOmegaGrid:
    """Tests for make_omega_grid function."""
    
    def test_output_shape(self):
        """Verify output has correct shape (exactly K points)."""
        W, K = 20.0, 128
        omegas = make_omega_grid(W, K)
        assert omegas.shape == (K,), f"Expected shape ({K},), got {omegas.shape}"
        assert omegas.dim() == 1, "Should be 1D tensor"
    
    def test_output_range(self):
        """Verify grid spans [-W, W]."""
        W, K = 20.0, 128
        omegas = make_omega_grid(W, K)
        assert omegas.min() >= -W, f"Min {omegas.min()} < -W={-W}"
        assert omegas.max() <= W, f"Max {omegas.max()} > W={W}"
        # Should include endpoints (approximately)
        assert abs(omegas.min() + W) < W * 0.1, "Grid should start near -W"
        assert abs(omegas.max() - W) < W * 0.1, "Grid should end near W"
    
    def test_sorted_ascending(self):
        """Verify grid is sorted in ascending order."""
        W, K = 20.0, 128
        omegas = make_omega_grid(W, K)
        assert torch.all(torch.diff(omegas) > 0), "Grid should be strictly increasing"
    
    def test_density_regions(self):
        """Verify three density regions (denser near zero)."""
        W, K = 20.0, 128
        omegas = make_omega_grid(W, K)
        
        # Count points in each region
        center_mask = torch.abs(omegas) <= 0.1 * W  # Inner 10%
        middle_mask = (torch.abs(omegas) > 0.1 * W) & (torch.abs(omegas) <= 0.4 * W)
        tail_mask = torch.abs(omegas) > 0.4 * W
        
        n_center = center_mask.sum().item()
        n_middle = middle_mask.sum().item()
        n_tail = tail_mask.sum().item()
        
        # Center should have roughly 50% of points (allow some tolerance)
        assert n_center >= K * 0.4, f"Center has {n_center} points, expected ~{K*0.5}"
        # Should have more points in center than in tails
        assert n_center > n_tail, "Center should be denser than tails"
    
    def test_device_placement(self):
        """Verify tensor is placed on correct device."""
        W, K = 20.0, 64
        
        # CPU
        omegas_cpu = make_omega_grid(W, K, device=torch.device('cpu'))
        assert omegas_cpu.device.type == 'cpu'
        
        # GPU (if available)
        if torch.cuda.is_available():
            omegas_gpu = make_omega_grid(W, K, device=torch.device('cuda'))
            assert omegas_gpu.device.type == 'cuda'
    
    def test_different_K_values(self):
        """Test with different K values - should return exactly K points."""
        W = 20.0
        for K in [32, 64, 128, 256, 512]:
            omegas = make_omega_grid(W, K)
            assert len(omegas) == K, f"Got {len(omegas)} points, expected exactly {K}"


class TestUnwrapPhase:
    """Tests for unwrap_phase function."""
    
    def test_linear_phase(self):
        """Phase that's already continuous should stay the same."""
        phase = torch.linspace(-3.0, 3.0, 100)
        unwrapped = unwrap_phase(phase)
        assert torch.allclose(phase, unwrapped, atol=1e-5)
    
    def test_wrap_correction(self):
        """Wrapped phase should be unwrapped correctly."""
        # Create phase with a wrap at pi
        t = torch.linspace(0, 4 * np.pi, 100)
        phase = t % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]
        
        unwrapped = unwrap_phase(phase)
        
        # Unwrapped should be monotonic
        diffs = torch.diff(unwrapped)
        # Allow small negative diffs due to numerical precision
        assert torch.all(diffs > -0.1), "Unwrapped phase should be roughly monotonic"
    
    def test_batch_dimension(self):
        """Test with batched input."""
        batch_size = 8
        K = 50
        phase = torch.randn(batch_size, K)
        
        unwrapped = unwrap_phase(phase, dim=-1)
        assert unwrapped.shape == (batch_size, K)
    
    def test_compare_with_numpy(self):
        """Compare with numpy's unwrap."""
        phase_np = np.linspace(0, 6 * np.pi, 100) % (2 * np.pi) - np.pi
        phase_torch = torch.from_numpy(phase_np).float()
        
        unwrapped_np = np.unwrap(phase_np)
        unwrapped_torch = unwrap_phase(phase_torch)
        
        # Should be close (may differ by multiples of 2*pi at boundaries)
        # Compare derivatives instead of absolute values
        diff_np = np.diff(unwrapped_np)
        diff_torch = torch.diff(unwrapped_torch).numpy()
        
        assert np.allclose(diff_np, diff_torch, atol=0.1), \
            "Unwrap derivatives should match numpy"


class TestInterpolateCFPolar:
    """Tests for interpolate_cf_polar function."""
    
    def test_output_shape_1d(self):
        """Test with 1D input."""
        K = 64
        K_target = 32
        omegas = make_omega_grid(20.0, K)
        target_omegas = make_omega_grid(15.0, K_target)
        cf = torch.exp(1j * omegas * 0.5)  # CF with mean 0.5
        
        result = interpolate_cf_polar(target_omegas, omegas, cf)
        assert result.shape == (K_target,), f"Expected ({K_target},), got {result.shape}"
    
    def test_output_shape_2d(self):
        """Test with 2D input [batch, K]."""
        batch = 16
        K = 64
        K_target = 32
        omegas = make_omega_grid(20.0, K)
        target_omegas = make_omega_grid(15.0, K_target)
        cf = torch.exp(1j * omegas.unsqueeze(0) * torch.randn(batch, 1))
        
        result = interpolate_cf_polar(target_omegas, omegas, cf)
        assert result.shape == (batch, K_target), f"Expected ({batch}, {K_target}), got {result.shape}"
    
    def test_output_shape_3d(self):
        """Test with 3D input [batch, n_actions, K]."""
        batch = 16
        n_actions = 4
        K = 64
        K_target = 32
        omegas = make_omega_grid(20.0, K)
        target_omegas = make_omega_grid(15.0, K_target)
        K_actual = omegas.shape[0]
        K_target_actual = target_omegas.shape[0]
        cf = torch.exp(1j * omegas.view(1, 1, K_actual) * torch.randn(batch, n_actions, 1))
        
        result = interpolate_cf_polar(target_omegas, omegas, cf)
        assert result.shape == (batch, n_actions, K_target_actual), \
            f"Expected ({batch}, {n_actions}, {K_target_actual}), got {result.shape}"
    
    def test_identity_interpolation(self):
        """Interpolating at same grid should return same values."""
        K = 64
        omegas = make_omega_grid(20.0, K)
        cf = torch.exp(1j * omegas * 0.5) * 0.9  # CF with some decay
        
        result = interpolate_cf_polar(omegas, omegas, cf)
        
        # Should be very close to original
        assert torch.allclose(result.real, cf.real, atol=1e-4), \
            "Real part should match for identity interpolation"
        assert torch.allclose(result.imag, cf.imag, atol=1e-4), \
            "Imag part should match for identity interpolation"
    
    def test_scaled_frequency_interpolation(self):
        """Test interpolation at γω (typical use case)."""
        gamma = 0.99
        K = 64
        omegas = make_omega_grid(20.0, K)
        target_omegas = gamma * omegas  # Scaled frequencies
        
        # Simple CF: exp(i * mu * omega)
        mu = 0.7
        cf = torch.exp(1j * omegas * mu)
        
        result = interpolate_cf_polar(target_omegas, omegas, cf)
        
        # At scaled frequencies, the CF should be exp(i * mu * gamma * omega)
        expected = torch.exp(1j * target_omegas * mu)
        
        # Allow some interpolation error
        error = complex_mse_loss(result, expected)
        assert error < 0.01, f"Interpolation error {error} too large"
    
    def test_magnitude_preservation(self):
        """Polar interpolation should preserve magnitude bounds."""
        K = 64
        omegas = make_omega_grid(20.0, K)
        target_omegas = 0.99 * omegas
        
        # CF with magnitude <= 1
        cf = torch.exp(1j * omegas * 0.5) * torch.exp(-omegas**2 * 0.01)
        
        result = interpolate_cf_polar(target_omegas, omegas, cf)
        
        # Magnitude should be <= max of original + small tolerance
        max_orig = torch.abs(cf).max()
        max_interp = torch.abs(result).max()
        assert max_interp <= max_orig + 0.1, \
            f"Interpolated magnitude {max_interp} exceeds original {max_orig}"
    
    def test_compare_with_numpy_reference(self):
        """Compare with numpy reference implementation."""
        K = 64
        omegas_torch = make_omega_grid(20.0, K)
        omegas_np = omegas_torch.numpy()
        
        # Create test CF
        mu = 0.6
        sigma = 0.3
        cf_np = np.exp(1j * omegas_np * mu - 0.5 * (sigma * omegas_np)**2)
        cf_torch = torch.from_numpy(cf_np)
        
        # Target frequencies
        gamma = 0.95
        target_np = gamma * omegas_np
        target_torch = torch.from_numpy(target_np).float()
        
        # NumPy reference (polar interpolation)
        mag_np = np.abs(cf_np)
        phase_np = np.unwrap(np.angle(cf_np))
        mag_interp_np = np.interp(target_np, omegas_np, mag_np)
        phase_interp_np = np.interp(target_np, omegas_np, phase_np)
        result_np = mag_interp_np * np.exp(1j * phase_interp_np)
        
        # PyTorch implementation
        result_torch = interpolate_cf_polar(target_torch, omegas_torch, cf_torch)
        
        # Compare
        error = np.abs(result_torch.numpy() - result_np).mean()
        assert error < 0.05, f"Mean error {error} between torch and numpy implementations"


class TestCollapseCFToMean:
    """Tests for collapse_cf_to_mean function."""
    
    def test_output_shape_1d(self):
        """Test with 1D input."""
        K = 64
        omegas = make_omega_grid(20.0, K)
        cf = torch.exp(1j * omegas * 0.5)
        
        result = collapse_cf_to_mean(omegas, cf)
        assert result.shape == (), f"Expected scalar, got {result.shape}"
    
    def test_output_shape_2d(self):
        """Test with 2D input [batch, K]."""
        batch = 16
        K = 64
        omegas = make_omega_grid(20.0, K)
        cf = torch.exp(1j * omegas.unsqueeze(0) * torch.randn(batch, 1))
        
        result = collapse_cf_to_mean(omegas, cf)
        assert result.shape == (batch,), f"Expected ({batch},), got {result.shape}"
    
    def test_output_shape_3d(self):
        """Test with 3D input [batch, n_actions, K]."""
        batch = 16
        n_actions = 4
        K = 64
        omegas = make_omega_grid(20.0, K)
        K_actual = omegas.shape[0]
        means = torch.randn(batch, n_actions, 1)
        cf = torch.exp(1j * omegas.view(1, 1, K_actual) * means)
        
        result = collapse_cf_to_mean(omegas, cf)
        assert result.shape == (batch, n_actions), \
            f"Expected ({batch}, {n_actions}), got {result.shape}"
    
    def test_known_mean_extraction(self):
        """Test mean extraction on CF with known mean."""
        K = 128
        omegas = make_omega_grid(20.0, K)
        
        # CF of distribution with mean μ: φ(ω) = exp(iμω) * (magnitude decay)
        # For Gaussian: φ(ω) = exp(iμω - σ²ω²/2)
        mu_true = 0.7
        sigma = 0.5
        cf = torch.exp(1j * omegas * mu_true - 0.5 * (sigma * omegas)**2)
        
        mu_est = collapse_cf_to_mean(omegas, cf, max_w=2.0)
        
        # Should recover mean accurately
        assert abs(mu_est.item() - mu_true) < 0.1, \
            f"Estimated mean {mu_est.item():.3f} != true mean {mu_true}"
    
    def test_batch_mean_extraction(self):
        """Test mean extraction with batched CFs."""
        batch = 8
        K = 128
        omegas = make_omega_grid(20.0, K)
        
        # Different means for each batch element
        mu_true = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9, -0.2, -0.5, 1.0])
        sigma = 0.3
        
        cf = torch.exp(1j * omegas.unsqueeze(0) * mu_true.unsqueeze(1) 
                      - 0.5 * (sigma * omegas.unsqueeze(0))**2)
        
        mu_est = collapse_cf_to_mean(omegas, cf, max_w=2.0)
        
        # All means should be close
        errors = (mu_est - mu_true).abs()
        assert errors.max() < 0.15, f"Max error {errors.max():.3f} too large"
    
    def test_compare_with_numpy_reference(self):
        """Compare with numpy reference implementation."""
        K = 128
        omegas_torch = make_omega_grid(20.0, K)
        omegas_np = omegas_torch.numpy()
        
        mu_true = 0.6
        sigma = 0.4
        cf_np = np.exp(1j * omegas_np * mu_true - 0.5 * (sigma * omegas_np)**2)
        cf_torch = torch.from_numpy(cf_np)
        
        # NumPy reference (gaussian method)
        max_w = 2.0
        mask = np.abs(omegas_np) <= max_w
        w_sub = omegas_np[mask]
        phi_sub = cf_np[mask]
        phase = np.unwrap(np.angle(phi_sub))
        mu_np = np.sum(w_sub * phase) / np.sum(w_sub ** 2)
        
        # PyTorch implementation
        mu_torch = collapse_cf_to_mean(omegas_torch, cf_torch, max_w=max_w)
        
        # Compare
        error = abs(mu_torch.item() - mu_np)
        assert error < 0.05, f"Error {error} between torch and numpy implementations"


class TestRewardCF:
    """Tests for reward_cf function."""
    
    def test_output_shape(self):
        """Test output shape."""
        batch = 16
        K = 64
        omegas = make_omega_grid(20.0, K)
        K_actual = omegas.shape[0]
        rewards = torch.randn(batch)
        
        result = reward_cf(omegas, rewards)
        assert result.shape == (batch, K_actual), f"Expected ({batch}, {K_actual}), got {result.shape}"
    
    def test_output_shape_with_extra_dim(self):
        """Test with rewards of shape [batch, 1]."""
        batch = 16
        K = 64
        omegas = make_omega_grid(20.0, K)
        K_actual = omegas.shape[0]
        rewards = torch.randn(batch, 1)
        
        result = reward_cf(omegas, rewards)
        assert result.shape == (batch, K_actual), f"Expected ({batch}, {K_actual}), got {result.shape}"
    
    def test_zero_reward(self):
        """CF of zero reward should be 1."""
        K = 64
        omegas = make_omega_grid(20.0, K)
        K_actual = omegas.shape[0]
        rewards = torch.zeros(4)
        
        result = reward_cf(omegas, rewards)
        
        # exp(i * omega * 0) = 1 for all omega
        expected = torch.ones(4, K_actual, dtype=torch.complex64)
        assert torch.allclose(result.real, expected.real, atol=1e-5)
        assert torch.allclose(result.imag, expected.imag, atol=1e-5)
    
    def test_known_reward(self):
        """Test with known reward value."""
        K = 64
        omegas = make_omega_grid(20.0, K)
        reward = 0.5
        rewards = torch.tensor([reward])
        
        result = reward_cf(omegas, rewards)
        expected = torch.exp(1j * omegas * reward).unsqueeze(0)
        
        assert torch.allclose(result.real, expected.real, atol=1e-5)
        assert torch.allclose(result.imag, expected.imag, atol=1e-5)
    
    def test_magnitude_is_one(self):
        """CF of deterministic reward has magnitude 1."""
        K = 64
        omegas = make_omega_grid(20.0, K)
        rewards = torch.randn(8)  # Any rewards
        
        result = reward_cf(omegas, rewards)
        magnitudes = torch.abs(result)
        
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-5), \
            "Magnitude of reward CF should be 1"


class TestComplexMSELoss:
    """Tests for complex_mse_loss function."""
    
    def test_zero_loss(self):
        """Identical tensors should have zero loss."""
        a = torch.randn(8, 64) + 1j * torch.randn(8, 64)
        loss = complex_mse_loss(a, a)
        assert loss.item() < 1e-10, f"Loss should be ~0, got {loss.item()}"
    
    def test_known_loss(self):
        """Test with known difference."""
        a = torch.ones(2, 2, dtype=torch.complex64)  # 1 + 0j
        b = torch.zeros(2, 2, dtype=torch.complex64)  # 0 + 0j
        
        # |1 - 0|² = 1² + 0² = 1
        loss = complex_mse_loss(a, b)
        assert abs(loss.item() - 1.0) < 1e-5, f"Expected loss 1.0, got {loss.item()}"
    
    def test_imaginary_difference(self):
        """Test with imaginary difference."""
        a = torch.zeros(2, 2, dtype=torch.complex64)
        b = 1j * torch.ones(2, 2, dtype=torch.complex64)  # 0 + 1j
        
        # |0 - j|² = 0² + 1² = 1
        loss = complex_mse_loss(a, b)
        assert abs(loss.item() - 1.0) < 1e-5, f"Expected loss 1.0, got {loss.item()}"


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_full_cf_pipeline(self):
        """Test the full CF pipeline: grid -> interpolate -> collapse."""
        batch = 4
        n_actions = 2
        K = 128
        gamma = 0.99
        
        # 1. Create frequency grid
        omegas = make_omega_grid(20.0, K)
        K_actual = omegas.shape[0]
        
        # 2. Create some CFs (simulating network output)
        mu_true = torch.tensor([[0.5, 0.8], [0.3, 0.6], [0.7, 0.4], [0.9, 0.2]])
        sigma = 0.3
        cf = torch.exp(1j * omegas.view(1, 1, K_actual) * mu_true.unsqueeze(-1)
                      - 0.5 * (sigma * omegas.view(1, 1, K_actual))**2)
        
        # 3. Interpolate at scaled frequencies
        scaled_omegas = gamma * omegas
        cf_scaled = interpolate_cf_polar(scaled_omegas, omegas, cf)
        
        assert cf_scaled.shape == (batch, n_actions, K_actual)
        
        # 4. Collapse to scalar Q-values
        q_values = collapse_cf_to_mean(omegas, cf)
        
        assert q_values.shape == (batch, n_actions)
        
        # Q-values should be close to true means
        errors = (q_values - mu_true).abs()
        assert errors.max() < 0.15, f"Max Q-value error {errors.max():.3f}"
    
    def test_bellman_target_computation(self):
        """Simulate CF Bellman target computation."""
        batch = 4
        K = 128
        gamma = 0.99
        
        # Create frequency grid
        omegas = make_omega_grid(20.0, K)
        K_actual = omegas.shape[0]
        
        # Rewards
        rewards = torch.tensor([1.0, 0.0, 0.5, -0.5])
        dones = torch.tensor([0.0, 0.0, 1.0, 0.0])
        
        # Next state CF (from target network)
        mu_next = 0.5
        cf_next = torch.exp(1j * omegas.unsqueeze(0) * mu_next
                          - 0.5 * (0.3 * omegas.unsqueeze(0))**2)
        cf_next = cf_next.expand(batch, K_actual)
        
        # Interpolate at scaled frequencies
        scaled_omegas = gamma * omegas
        cf_next_scaled = interpolate_cf_polar(scaled_omegas, omegas, cf_next)
        
        # Compute reward CF
        cf_reward = reward_cf(omegas, rewards)
        
        # Bellman target: reward_cf * cf_next_scaled * (1 - done) + reward_cf * done
        dones_expanded = dones.unsqueeze(-1)  # [batch, 1]
        cf_future = cf_next_scaled * (1 - dones_expanded) + (1 + 0j) * dones_expanded
        target_cf = cf_reward * cf_future
        
        assert target_cf.shape == (batch, K_actual)
        
        # Check that terminal state (index 2) has CF = exp(iωr) only
        terminal_cf = target_cf[2]
        expected_terminal = torch.exp(1j * omegas * rewards[2])
        assert torch.allclose(terminal_cf.real, expected_terminal.real, atol=1e-4)
        assert torch.allclose(terminal_cf.imag, expected_terminal.imag, atol=1e-4)


class TestQNetwork:
    """Tests for the CF-DQN QNetwork class."""
    
    def test_network_output_shapes(self):
        """Verify network produces correct output shapes."""
        import gymnasium as gym
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from cleanrl.cf_dqn import QNetwork, make_env
        
        # Create environment
        env_id = "CartPole-v1"
        envs = gym.vector.SyncVectorEnv([make_env(env_id, 1, 0, False, "test")])
        
        # Create network
        n_frequencies = 64
        freq_max = 20.0
        q_network = QNetwork(envs, n_frequencies=n_frequencies, freq_max=freq_max)
        
        # Test forward pass
        batch_size = 8
        obs = torch.randn(batch_size, 4)  # CartPole has 4D observations
        
        cf_all = q_network.forward(obs)
        assert cf_all.shape == (batch_size, 2, n_frequencies), \
            f"Expected ({batch_size}, 2, {n_frequencies}), got {cf_all.shape}"
        assert cf_all.dtype == torch.complex64, f"Expected complex64, got {cf_all.dtype}"
        
        envs.close()
    
    def test_get_action(self):
        """Verify get_action returns correct shapes."""
        import gymnasium as gym
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from cleanrl.cf_dqn import QNetwork, make_env
        
        env_id = "CartPole-v1"
        envs = gym.vector.SyncVectorEnv([make_env(env_id, 1, 0, False, "test")])
        
        n_frequencies = 64
        q_network = QNetwork(envs, n_frequencies=n_frequencies)
        
        batch_size = 8
        obs = torch.randn(batch_size, 4)
        
        # Without action
        action, cf = q_network.get_action(obs)
        assert action.shape == (batch_size,), f"Action shape: {action.shape}"
        assert cf.shape == (batch_size, n_frequencies), f"CF shape: {cf.shape}"
        
        # With action
        given_action = torch.zeros(batch_size, dtype=torch.long)
        action2, cf2 = q_network.get_action(obs, given_action)
        assert torch.all(action2 == given_action), "Should return given action"
        assert cf2.shape == (batch_size, n_frequencies)
        
        envs.close()
    
    def test_get_all_cf(self):
        """Verify get_all_cf returns correct shapes."""
        import gymnasium as gym
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from cleanrl.cf_dqn import QNetwork, make_env
        
        env_id = "CartPole-v1"
        envs = gym.vector.SyncVectorEnv([make_env(env_id, 1, 0, False, "test")])
        
        n_frequencies = 64
        q_network = QNetwork(envs, n_frequencies=n_frequencies)
        
        batch_size = 8
        n_actions = 2  # CartPole has 2 actions
        obs = torch.randn(batch_size, 4)
        
        cf_all, q_values = q_network.get_all_cf(obs)
        
        assert cf_all.shape == (batch_size, n_actions, n_frequencies), \
            f"CF all shape: {cf_all.shape}"
        assert q_values.shape == (batch_size, n_actions), \
            f"Q-values shape: {q_values.shape}"
        
        envs.close()
    
    def test_gradient_flow(self):
        """Verify gradients flow through the network."""
        import gymnasium as gym
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from cleanrl.cf_dqn import QNetwork, make_env
        
        env_id = "CartPole-v1"
        envs = gym.vector.SyncVectorEnv([make_env(env_id, 1, 0, False, "test")])
        
        q_network = QNetwork(envs, n_frequencies=64)
        
        obs = torch.randn(4, 4, requires_grad=False)
        action, cf = q_network.get_action(obs)
        
        # Compute a simple loss
        loss = (cf.real ** 2 + cf.imag ** 2).mean()
        loss.backward()
        
        # Check that gradients exist
        for name, param in q_network.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.all(param.grad == 0), f"Zero gradient for {name}"
        
        envs.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

