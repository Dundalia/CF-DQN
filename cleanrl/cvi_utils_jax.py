import jax
import jax.numpy as jnp

def get_cleaned_target_cf(omega_grid, V_complex_target, q_min=0.0, q_max=100.0):
    """
    The goal of this function is pretty simple, take the V_complex_target,
    convert it back to spatial domain to get the PDF, apply the the distributional mask to remove the incoherent probability mass,
    and then convert it back to the frequency domain to get a cleaned CF target.
    """
    K = V_complex_target.shape[-1]
    W = jnp.abs(omega_grid[0])
    dx = jnp.pi / W
    q_values_grid = jnp.linspace(-(K // 2) * dx, (K // 2 - 1) * dx, K)

    #* 1. Frequency -> Spatial: CF uses +i convention, so PDF recovery needs -i (= fft)
    V_shifted = jnp.fft.ifftshift(V_complex_target, axes=-1)
    pdf_complex = jnp.fft.fft(V_shifted, axis=-1)
    pdf = jnp.clip(jnp.fft.fftshift(pdf_complex.real, axes=-1), min=0.0)

    #* 2. Apply spatial mask to destroy the accumulated noise
    valid_mask = (q_values_grid >= q_min) & (q_values_grid <= q_max)
    pdf_clean = pdf * valid_mask.astype(jnp.float32)

    #* 3. Normalize back to a valid probability distribution
    pdf_clean = pdf_clean / (pdf_clean.sum(axis=-1, keepdims=True) + 1e-8)

    #* 4. Spatial -> Frequency: CF uses +i convention (= K * ifft)
    pdf_unshifted = jnp.fft.ifftshift(pdf_clean.astype(jnp.complex64), axes=-1)
    V_clean_shifted = K * jnp.fft.ifft(pdf_unshifted, axis=-1)
    cleaned_cf = jnp.fft.fftshift(V_clean_shifted, axes=-1)

    return cleaned_cf


def create_uniform_grid(K: int, W: float):
    """
    Constructs a uniform frequency grid required for Fast Fourier Transforms.
    Generates K points perfectly symmetric around 0, spanning [-W, W - dw].
    """
    assert K % 2 == 0, "K must be even for FFT symmetry."
    dw = 2.0 * W / K
    grid = jnp.linspace(-W, W - dw, K)
    return grid


def ifft_collapse_q_values(omega_grid, V_complex, q_min=0.0, q_max=100.0, return_diagnostics=False):
    """
    Extracts Q-values by transforming the CF back into a Probability Density Function (PDF)
    using the Inverse Fast Fourier Transform, then computing the expected value E[x].
    Requires a uniform frequency grid.
    """
    K = V_complex.shape[-1]
    W = jnp.abs(omega_grid[0])
    dx = jnp.pi / W

    #* Construct the corresponding spatial grid x: [-K/2 * dx, (K/2 - 1) * dx]
    q_values_grid = jnp.linspace(-(K // 2) * dx, (K // 2 - 1) * dx, K)

    #* Shift frequency 0 to index 0 for standard FFT Implementation
    #* [-W, ..., -dw, 0, dw, ..., W - dw] -> [0, ..., W - dw, -W, ..., -dw]
    V_shifted = jnp.fft.ifftshift(V_complex, axes=-1)

    #* CF→PDF: the CF uses exp(+iωx), so recovery needs exp(-iωx) = fft convention
    pdf_complex = jnp.fft.fft(V_shifted, axis=-1)

    #* Puts back the negative frequencies on the left and positive frequencies on the right
    pdf_shifted = jnp.fft.fftshift(pdf_complex.real, axes=-1)

    pdf_unmasked = jnp.clip(pdf_shifted, min=0.0)
    pdf_unmasked_normalized = pdf_unmasked / (pdf_unmasked.sum(axis=-1, keepdims=True) + 1e-8)
    expected_q_value_scalar_unmasked = jnp.sum(pdf_unmasked_normalized * q_values_grid, axis=-1)

    valid_mask = (q_values_grid >= q_min) & (q_values_grid <= q_max)
    pdf_masked = pdf_unmasked * valid_mask.astype(jnp.float32)
    pdf_masked_normalized = pdf_masked / (pdf_masked.sum(axis=-1, keepdims=True) + 1e-8)
    expected_q_value_scalar = jnp.sum(pdf_masked_normalized * q_values_grid, axis=-1)

    if return_diagnostics:
        return expected_q_value_scalar, expected_q_value_scalar_unmasked, pdf_unmasked

    return expected_q_value_scalar


def unwrap_phase(phase, axis=-1):
    """
    Removes 2*pi jumps in the phase angle to create a continuous line for interpolation.
    """
    diff = jnp.diff(phase, axis=axis)
    diff_wrapped = (diff + jnp.pi) % (2 * jnp.pi) - jnp.pi

    # Slice the first element along axis
    first_element = jnp.take(phase, jnp.array([0]), axis=axis)
    unwrapped = jnp.concatenate([first_element, first_element + jnp.cumsum(diff_wrapped, axis=axis)], axis=axis)

    return unwrapped


def polar_interpolation(omega_grid, target_V_complex, gammas):
    """
    Interpolates the target network's CF at the scaled frequencies (gamma * omega).
    Interpolates magnitude and unwrapped phase separately to prevent phase wrapping artifacts.
    """
    magnitudes = jnp.abs(target_V_complex)
    phases = jnp.angle(target_V_complex)
    unwrapped_phases = unwrap_phase(phases, axis=-1)

    gamma_target_omega = gammas * omega_grid.reshape(1, -1)

    # jnp.searchsorted expects a 1D sorted array; vmap over the batch rows of gamma_target_omega
    _searchsorted_row = lambda row: jnp.searchsorted(omega_grid, row)
    idx_right = jax.vmap(_searchsorted_row)(gamma_target_omega)
    idx_right = jnp.clip(idx_right, 1, omega_grid.shape[0] - 1)
    idx_left = idx_right - 1

    omega_left = omega_grid[idx_left]
    omega_right = omega_grid[idx_right]

    t = (gamma_target_omega - omega_left) / (omega_right - omega_left + 1e-8)

    batch_idx = jnp.arange(target_V_complex.shape[0]).reshape(-1, 1)

    #* Linearly interpolate magnitude
    mag_left = magnitudes[batch_idx, idx_left]
    mag_right = magnitudes[batch_idx, idx_right]
    interp_mag = (1 - t) * mag_left + t * mag_right

    #* Linearly interpolate the unwrapped phase
    phase_left = unwrapped_phases[batch_idx, idx_left]
    phase_right = unwrapped_phases[batch_idx, idx_right]
    interp_phase = phase_left + t * (phase_right - phase_left)

    return interp_mag * (jnp.cos(interp_phase) + 1j * jnp.sin(interp_phase))


