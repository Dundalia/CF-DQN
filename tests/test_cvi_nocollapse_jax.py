#!/usr/bin/env python3
"""
Unit tests for the analytic CF-DQN (no-collapse) implementation.
Run with: python tests/test_cvi_nocollapse_jax.py
"""
import sys
import numpy as np


def test_sample_frequencies():
    """Test that frequency sampling returns correct shapes and bounds."""
    import jax
    from cleanrl.cvi_utils_nocollapse_jax import sample_frequencies

    key = jax.random.PRNGKey(42)
    for N in [8, 32, 64]:
        for omega_max in [0.5, 1.0, 2.0]:
            omegas = sample_frequencies(key, N, omega_max)
            assert omegas.shape == (N,), f"Wrong shape: {omegas.shape}"
            assert float(omegas.min()) >= -omega_max, f"Below -omega_max"
            assert float(omegas.max()) <= omega_max, f"Above omega_max"
    print("✓ sample_frequencies: PASS")


def test_build_analytic_cf_at_zero():
    """At ω=0, the analytic CF must equal 1+0j regardless of m, σ."""
    import jax.numpy as jnp
    from cleanrl.cvi_utils_nocollapse_jax import build_analytic_cf

    m = jnp.array([[3.14, -1.0]])      # (1, 2) — one freq, 2 actions
    sigma = jnp.array([[0.5, 2.0]])     # (1, 2)
    omegas = jnp.zeros(1)              # ω = 0

    phi = build_analytic_cf(m, sigma, omegas)
    # At ω=0: κ(0)=0, exp(0)=1
    np.testing.assert_allclose(np.array(phi.real), 1.0, atol=1e-6,
        err_msg="Re(φ(0)) should be 1")
    np.testing.assert_allclose(np.array(phi.imag), 0.0, atol=1e-6,
        err_msg="Im(φ(0)) should be 0")
    print("✓ build_analytic_cf at ω=0: PASS")


def test_build_analytic_cf_magnitude_bound():
    """With σ ≥ 0 and κ(ω) = ω² ≥ 0, we must have |φ| ≤ 1."""
    import jax
    import jax.numpy as jnp
    from cleanrl.cvi_utils_nocollapse_jax import build_analytic_cf

    key = jax.random.PRNGKey(0)
    N, A = 100, 4
    m = jax.random.normal(key, (N, A)) * 10
    sigma = jax.nn.softplus(jax.random.normal(jax.random.PRNGKey(1), (N, A)))
    omegas = jnp.linspace(-3.0, 3.0, N)

    phi = build_analytic_cf(m, sigma, omegas)
    magnitudes = np.array(jnp.abs(phi))
    assert np.all(magnitudes <= 1.0 + 1e-5), (
        f"|φ| > 1 detected: max={magnitudes.max():.6f}")
    print("✓ build_analytic_cf |φ| ≤ 1: PASS")


def test_network_forward_shapes():
    """Test that AnalyticCF_QNetwork produces correct output shapes."""
    import jax
    import jax.numpy as jnp
    from cleanrl.cvi_dqn_nocollapse_jax import AnalyticCF_QNetwork

    obs_size, action_dim, embed_dim = 4, 2, 32
    key = jax.random.PRNGKey(0)
    net = AnalyticCF_QNetwork(obs_size, action_dim, embed_dim, key=key)

    obs = jnp.ones(obs_size)
    N = 16
    omegas = jnp.linspace(-1.0, 1.0, N)

    m, sigma = net(obs, omegas)
    assert m.shape == (N, action_dim), f"m shape wrong: {m.shape}"
    assert sigma.shape == (N, action_dim), f"sigma shape wrong: {sigma.shape}"
    assert np.all(np.array(sigma) >= 0), "sigma must be non-negative"
    print("✓ AnalyticCF_QNetwork forward shapes: PASS")


def test_network_q_values():
    """Test that q_values() returns correct shape and is deterministic."""
    import jax
    import jax.numpy as jnp
    from cleanrl.cvi_dqn_nocollapse_jax import AnalyticCF_QNetwork

    obs_size, action_dim, embed_dim = 4, 2, 32
    key = jax.random.PRNGKey(0)
    net = AnalyticCF_QNetwork(obs_size, action_dim, embed_dim, key=key)

    obs = jnp.ones(obs_size)
    q = net.q_values(obs)
    assert q.shape == (action_dim,), f"Q shape wrong: {q.shape}"
    assert q.dtype == jnp.float32, f"Q dtype wrong: {q.dtype}"

    # Should be deterministic
    q2 = net.q_values(obs)
    np.testing.assert_array_equal(np.array(q), np.array(q2),
        err_msg="q_values not deterministic")
    print("✓ AnalyticCF_QNetwork q_values: PASS")


def test_network_cf_valid():
    """Test that the CF built from network outputs satisfies validity conditions."""
    import jax
    import jax.numpy as jnp
    from cleanrl.cvi_dqn_nocollapse_jax import AnalyticCF_QNetwork
    from cleanrl.cvi_utils_nocollapse_jax import build_analytic_cf

    obs_size, action_dim, embed_dim = 4, 2, 32
    key = jax.random.PRNGKey(0)
    net = AnalyticCF_QNetwork(obs_size, action_dim, embed_dim, key=key)

    obs = jnp.ones(obs_size)
    N = 64
    omegas = jnp.linspace(-2.0, 2.0, N)

    m, sigma = net(obs, omegas)
    phi = build_analytic_cf(m, sigma, omegas)

    # Shape check
    assert phi.shape == (N, action_dim), f"φ shape wrong: {phi.shape}"

    # |φ(ω)| ≤ 1
    mags = np.array(jnp.abs(phi))
    assert np.all(mags <= 1.0 + 1e-5), f"|φ| > 1: max={mags.max()}"

    # φ(0) = 1+0j  — find the index closest to ω=0
    zero_idx = N // 2
    phi_zero = phi[zero_idx]
    np.testing.assert_allclose(np.array(phi_zero.real), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.array(phi_zero.imag), 0.0, atol=1e-5)

    print("✓ Network CF validity: PASS")


def test_network_batched_vmap():
    """Test that vmap over batch of states works."""
    import jax
    import jax.numpy as jnp
    from cleanrl.cvi_dqn_nocollapse_jax import AnalyticCF_QNetwork

    obs_size, action_dim, embed_dim = 4, 2, 32
    key = jax.random.PRNGKey(0)
    net = AnalyticCF_QNetwork(obs_size, action_dim, embed_dim, key=key)

    batch = 8
    N = 16
    obs_batch = jnp.ones((batch, obs_size))
    omegas_batch = jnp.broadcast_to(jnp.linspace(-1, 1, N)[None, :], (batch, N))

    m, sigma = jax.vmap(net)(obs_batch, omegas_batch)
    assert m.shape == (batch, N, action_dim), f"Batched m shape wrong: {m.shape}"
    assert sigma.shape == (batch, N, action_dim), f"Batched sigma shape wrong: {sigma.shape}"

    # Batched q_values
    q_batch = jax.vmap(net.q_values)(obs_batch)
    assert q_batch.shape == (batch, action_dim), f"Batched Q shape wrong: {q_batch.shape}"

    print("✓ Batched vmap: PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("Analytic CF-DQN (No-Collapse) Verification")
    print("=" * 60)

    tests = [
        test_sample_frequencies,
        test_build_analytic_cf_at_zero,
        test_build_analytic_cf_magnitude_bound,
        test_network_forward_shapes,
        test_network_q_values,
        test_network_cf_valid,
        test_network_batched_vmap,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: FAIL - {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    else:
        print("All tests passed!")
