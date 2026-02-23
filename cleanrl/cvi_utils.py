import os
import torch
import math



def create_three_density_grid(K: int, W: float, device="cpu"):
    """
    Constructs the Three-Density Region frequency grid.
    Allocates 50% of points to inner 10%, 30% to middle, and 20% to tails.
    
    Args:
        K: Total number of grid points (must be even for symmetry).
        W: Maximum frequency range [-W, W].
        device: PyTorch device.
        
    Returns:
        1D Tensor of shape (K + 1,) representing the frequency grid centered at 0.0.
    """
    assert K % 2 == 0, "K must be even to maintain perfect symmetry around 0."
    half_k = K // 2
    
    n_inner = int(half_k * 0.50)
    n_mid = int(half_k * 0.30)
    n_tail = half_k - n_inner - n_mid
    
    inner_bound = 0.1 * W
    mid_bound = 0.5 * W 
    
    # Generate the positive half of the grid
    inner = torch.linspace(1e-5, inner_bound, n_inner, device=device)
    mid = torch.linspace(inner_bound + 1e-4, mid_bound, n_mid, device=device)
    tail = torch.linspace(mid_bound + 1e-4, W, n_tail, device=device)
    
    pos_grid = torch.cat([inner, mid, tail])
    
    # Mirror for the negative half, and explicitly insert 0.0 in the exact center
    neg_grid = -torch.flip(pos_grid, dims=[0])
    zero_point = torch.tensor([0.0], device=device)
    
    grid = torch.cat([neg_grid, zero_point, pos_grid])
    return grid

def unwrap_phase(phase, dim=-1):
    """
    ! PyTorch equivalent of numpy.unwrap.
    Removes 2*pi jumps in the phase angle to create a continuous line for interpolation.
    """
    diff = torch.diff(phase, dim=dim)
    diff_wrapped = (diff + math.pi) % (2 * math.pi) - math.pi
    
    first_element = phase.narrow(dim, 0, 1)
    unwrapped = torch.cat([first_element, first_element + torch.cumsum(diff_wrapped, dim=dim)], dim=dim)
    
    #TODO: to plot and see if the unwrapping is correct.
    #_plot_unwrap_phase_debug(phase, unwrapped, dim)
    return unwrapped

def polar_interpolation(omega_grid, target_V_complex, gammas):
    """
    Interpolates the target network's CF at the scaled frequencies (gamma * omega).
    
    Args:
        omega_grid: 1D Tensor of shape (K_actual,)
        target_V_complex: Complex Tensor of shape (Batch, K_actual) for the selected actions.
        gammas: Tensor of shape (Batch, 1) representing discount factors (handles terminal states).
    """
    
    # TODO: make sure that it knows how to handle a complex 64 tensor
    magnitudes = torch.abs(target_V_complex) # shape (Batch, K_actual)
    phases = torch.angle(target_V_complex) # shape (Batch, K_actual)
    unwrapped_phases = unwrap_phase(phases, dim=-1)
    
    gamma_target_omega = gammas * omega_grid.view(1, -1) # shape (Batch, K_actual)
    
    # Find interpolation indices
    idx_right = torch.searchsorted(omega_grid,gamma_target_omega) #https://docs.pytorch.org/docs/stable/generated/torch.searchsorted.html
    idx_right = torch.clamp(idx_right, 1, len(omega_grid) - 1) 
    idx_left = idx_right - 1
    
    omega_left = omega_grid[idx_left]
    omega_right = omega_grid[idx_right]
    
    # Interpolation weights
    t = (gamma_target_omega - omega_left) / (omega_right - omega_left + 1e-8)
    
    # Batch indices to gather correctly
    batch_idx = torch.arange(target_V_complex.shape[0], device=target_V_complex.device).unsqueeze(1)
    
    # Linearly interpolate components
    mag_left = magnitudes[batch_idx, idx_left]
    mag_right = magnitudes[batch_idx, idx_right]
    interp_mag = (1 - t) * mag_left + t * mag_right
    
    phase_left = unwrapped_phases[batch_idx, idx_left]
    phase_right = unwrapped_phases[batch_idx, idx_right]
    interp_phase = (1 - t) * phase_left + t * phase_right
    
    # Returns the interpolated CF values at the scaled frequencies
    return interp_mag * torch.complex(torch.cos(interp_phase), torch.sin(interp_phase))

def gaussian_collapse_q_values(omega_grid, V_complex, w_collapse):
    """
    Extracts the expected return (Q-value) from the Characteristic Function via OLS regression.
    
    Args:
        omega_grid: 1D Tensor of shape (K_actual,)
        V_complex: Complex Tensor of shape (..., K_actual)
        w_collapse: Maximum frequency range [-w_collapse, w_collapse] for the collapse.
        
    Returns:
        Q-values of shape (...)
    """
    phase = torch.angle(V_complex)
    unwrapped_phase = unwrap_phase(phase, dim=-1)
    
    # Filter for low-frequency window
    #TODO: make sure the collapse frequencies contains both positive and negative frequencies, and that the grid is symmetric around 0.
    low_freq_mask = torch.abs(omega_grid) <= w_collapse
    w_low = omega_grid[low_freq_mask]
    phase_low = unwrapped_phase[..., low_freq_mask]
    
    # Ordinary Least Squares through the origin: Q = sum(w * phase) / sum(w^2)
    numerator = torch.sum(w_low * phase_low, dim=-1)
    denominator = torch.sum(w_low ** 2, dim=-1)
    
    q_values = numerator / (denominator + 1e-8)
    return q_values


def _plot_unwrap_phase_debug(phase_tensor: torch.Tensor, unwrapped_tensor: torch.Tensor, dim: int) -> None:
    import matplotlib.pyplot as plt

    with torch.no_grad():
        t_cpu = phase_tensor.detach().to("cpu")
        u_cpu = unwrapped_tensor.detach().to("cpu")
        if t_cpu.ndim == 0:
            wrapped_curve = t_cpu.unsqueeze(0).numpy()
            unwrapped_curve = u_cpu.unsqueeze(0).numpy()
            axis = 0
        else:
            axis = dim % t_cpu.ndim
            wrapped_curve = t_cpu.moveaxis(axis, -1).reshape(-1, t_cpu.shape[axis])[0].numpy()
            unwrapped_curve = u_cpu.moveaxis(axis, -1).reshape(-1, u_cpu.shape[axis])[0].numpy()

    x = range(len(wrapped_curve))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(x, wrapped_curve, label="wrapped phase", linewidth=1.2, alpha=0.7)
    ax.plot(x, unwrapped_curve, label="unwrapped phase", linewidth=1.2)
    ax.set_title(f"unwrap_phase debug (dim={axis})")
    ax.set_xlabel("index")
    ax.set_ylabel("phase (rad)")
    ax.legend()
    fig.tight_layout()
    fig.canvas.draw_idle()
    plt.show(block=False)
    plt.pause(0.001)
    plt.close(fig)
