"""Test theoretical prediction functions."""

import held
import jax
import jax.numpy as jnp
import numpy as np


def test_expected_ld_constant():
    """Test that expected_ld_constant runs without errors and returns positive values."""
    rng = np.random.default_rng(42)

    # Random population size between 1,000 and 100,000
    population_size = rng.uniform(1000, 100000)

    # Random bins
    left_bins = jnp.array([0.0, 0.1, 0.2, 0.3])
    right_bins = jnp.array([0.1, 0.2, 0.3, 0.4])

    # Random sample size between 10 and 100
    sample_size = rng.integers(10, 100)

    # Test without sample size correction
    result = held.expected_ld_constant(population_size, left_bins, right_bins)
    assert result.shape == (4,), f"Expected shape (4,), got {result.shape}"
    assert jnp.all(result > 0), "All LD values should be strictly greater than zero"

    # Test with sample size correction
    result_corrected = held.expected_ld_constant(
        population_size, left_bins, right_bins, sample_size=sample_size
    )
    assert result_corrected.shape == (4,), (
        f"Expected shape (4,), got {result_corrected.shape}"
    )
    assert jnp.all(result_corrected > 0), (
        "All corrected LD values should be strictly greater than zero"
    )


def test_expected_ld_piecewise_exponential():
    """Test that expected_ld_piecewise_exponential runs without errors and returns positive values."""
    rng = np.random.default_rng(123)

    # Random contemporary and ancestral population sizes between 1,000 and 100,000
    Ne_c = rng.uniform(1000, 100000)
    Ne_a = rng.uniform(1000, 100000)

    # Random transition time between 100 and 10,000 generations
    t0 = rng.uniform(100, 10000)

    # Random growth rate between -0.01 and 0.01
    alpha = rng.uniform(-0.01, 0.01)

    # Random bins
    left_bins = jnp.array([0.0, 0.1, 0.2, 0.3])
    right_bins = jnp.array([0.1, 0.2, 0.3, 0.4])

    # Random sample size between 10 and 100
    sample_size = rng.integers(10, 100)

    # Test without sample size correction
    result = held.expected_ld_piecewise_exponential(
        Ne_c, Ne_a, t0, alpha, left_bins, right_bins
    )
    assert result.shape == (4,), f"Expected shape (4,), got {result.shape}"
    assert jnp.all(result > 0), "All LD values should be strictly greater than zero"

    # Test with sample size correction
    result_corrected = held.expected_ld_piecewise_exponential(
        Ne_c, Ne_a, t0, alpha, left_bins, right_bins, sample_size=sample_size
    )
    assert result_corrected.shape == (4,), (
        f"Expected shape (4,), got {result_corrected.shape}"
    )
    assert jnp.all(result_corrected > 0), (
        "All corrected LD values should be strictly greater than zero"
    )


def test_expected_ld_piecewise_constant():
    """Test piecewise constant population size model with multiple epochs."""
    rng = np.random.default_rng(456)

    # Random bins
    left_bins = jnp.array([0.0, 0.1, 0.2, 0.3])
    right_bins = jnp.array([0.1, 0.2, 0.3, 0.4])

    # Test 2-epoch model with random values
    Ne_values_2 = jnp.array([rng.uniform(5000, 15000), rng.uniform(15000, 25000)])
    t_boundaries_2 = jnp.array([rng.uniform(500, 1500)])
    result_2 = held.expected_ld_piecewise_constant(
        Ne_values_2, t_boundaries_2, left_bins, right_bins
    )
    assert result_2.shape == (4,), f"Expected shape (4,), got {result_2.shape}"
    assert jnp.all(result_2 > 0), "All LD values should be strictly greater than zero"

    # Test 3-epoch model with random values
    Ne_values_3 = jnp.array(
        [rng.uniform(8000, 12000), rng.uniform(3000, 7000), rng.uniform(12000, 18000)]
    )
    t_boundaries_3 = jnp.array([rng.uniform(300, 700), rng.uniform(1500, 2500)])
    result_3 = held.expected_ld_piecewise_constant(
        Ne_values_3, t_boundaries_3, left_bins, right_bins
    )
    assert result_3.shape == (4,), f"Expected shape (4,), got {result_3.shape}"
    assert jnp.all(result_3 > 0), "All LD values should be strictly greater than zero"

    # Test 4-epoch model with random values
    Ne_values_4 = jnp.array(
        [
            rng.uniform(6000, 10000),
            rng.uniform(10000, 14000),
            rng.uniform(4000, 8000),
            rng.uniform(15000, 20000),
        ]
    )
    t_boundaries_4 = jnp.array(
        [rng.uniform(200, 400), rng.uniform(800, 1200), rng.uniform(2500, 3500)]
    )
    result_4 = held.expected_ld_piecewise_constant(
        Ne_values_4, t_boundaries_4, left_bins, right_bins
    )
    assert result_4.shape == (4,), f"Expected shape (4,), got {result_4.shape}"
    assert jnp.all(result_4 > 0), "All LD values should be strictly greater than zero"

    # Test with sample size correction using random value
    sample_size = rng.integers(30, 70)
    result_corrected = held.expected_ld_piecewise_constant(
        Ne_values_2, t_boundaries_2, left_bins, right_bins, sample_size=sample_size
    )
    assert result_corrected.shape == (4,), (
        f"Expected shape (4,), got {result_corrected.shape}"
    )
    assert jnp.all(result_corrected > 0), (
        "All corrected LD values should be strictly greater than zero"
    )

    # Test that constant model matches single epoch
    Ne_constant = 10000.0
    result_constant_old = held.expected_ld_constant(Ne_constant, left_bins, right_bins)
    result_constant_new = held.expected_ld_piecewise_constant(
        jnp.array([Ne_constant]), jnp.array([]), left_bins, right_bins
    )
    # Warning: we reduce tolerance because for a single epoch we have a closed form solution
    assert jnp.allclose(result_constant_old, result_constant_new, rtol=0.01), (
        "Single epoch piecewise should match constant model"
    )

    # Test that piecewise constant matches piecewise exponential with alpha=0
    Ne_c = 10000.0
    Ne_a = 20000.0
    t0 = 50.0
    alpha = 0.0

    result_exponential = held.expected_ld_piecewise_exponential(
        Ne_c, Ne_a, t0, alpha, left_bins, right_bins
    )
    result_constant_pw = held.expected_ld_piecewise_constant(
        jnp.array([Ne_c, Ne_a]), jnp.array([t0]), left_bins, right_bins
    )
    assert jnp.allclose(result_exponential, result_constant_pw, rtol=1e-4), (
        "Piecewise constant should match piecewise exponential with alpha=0"
    )


def test_derivatives_computable():
    """Test that derivatives can be computed for MLE optimization."""
    rng = np.random.default_rng(789)

    # Random bins
    left_bins = jnp.array([0.0, 0.1, 0.2, 0.3])
    right_bins = jnp.array([0.1, 0.2, 0.3, 0.4])

    # Test gradient of constant model
    population_size = rng.uniform(5000, 15000)
    grad_fn = jax.grad(
        lambda Ne: jnp.sum(held.expected_ld_constant(Ne, left_bins, right_bins))
    )
    gradient = grad_fn(population_size)
    assert jnp.isfinite(gradient), "Gradient should be finite for constant model"

    # Test gradient of piecewise exponential model
    Ne_c = rng.uniform(5000, 15000)
    Ne_a = rng.uniform(15000, 25000)
    t0 = rng.uniform(500, 1500)
    alpha = rng.uniform(-0.01, 0.01)

    grad_fn_Ne_c = jax.grad(
        lambda x: jnp.sum(
            held.expected_ld_piecewise_exponential(
                x, Ne_a, t0, alpha, left_bins, right_bins
            )
        )
    )
    grad_fn_Ne_a = jax.grad(
        lambda x: jnp.sum(
            held.expected_ld_piecewise_exponential(
                Ne_c, x, t0, alpha, left_bins, right_bins
            )
        )
    )
    grad_fn_t0 = jax.grad(
        lambda x: jnp.sum(
            held.expected_ld_piecewise_exponential(
                Ne_c, Ne_a, x, alpha, left_bins, right_bins
            )
        )
    )
    grad_fn_alpha = jax.grad(
        lambda x: jnp.sum(
            held.expected_ld_piecewise_exponential(
                Ne_c, Ne_a, t0, x, left_bins, right_bins
            )
        )
    )

    gradient_Ne_c = grad_fn_Ne_c(Ne_c)
    gradient_Ne_a = grad_fn_Ne_a(Ne_a)
    gradient_t0 = grad_fn_t0(t0)
    gradient_alpha = grad_fn_alpha(alpha)

    assert jnp.isfinite(gradient_Ne_c), "Gradient w.r.t. Ne_c should be finite"
    assert jnp.isfinite(gradient_Ne_a), "Gradient w.r.t. Ne_a should be finite"
    assert jnp.isfinite(gradient_t0), "Gradient w.r.t. t0 should be finite"
    assert jnp.isfinite(gradient_alpha), "Gradient w.r.t. alpha should be finite"

    # Test gradient of piecewise constant model
    Ne_values = jnp.array([rng.uniform(5000, 15000), rng.uniform(15000, 25000)])
    t_boundaries = jnp.array([rng.uniform(500, 1500)])

    grad_fn_Ne = jax.grad(
        lambda x: jnp.sum(
            held.expected_ld_piecewise_constant(x, t_boundaries, left_bins, right_bins)
        )
    )
    grad_fn_t = jax.grad(
        lambda x: jnp.sum(
            held.expected_ld_piecewise_constant(Ne_values, x, left_bins, right_bins)
        )
    )

    gradient_Ne = grad_fn_Ne(Ne_values)
    gradient_t = grad_fn_t(t_boundaries)

    assert gradient_Ne.shape == Ne_values.shape, (
        "Gradient shape should match Ne_values shape"
    )
    assert gradient_t.shape == t_boundaries.shape, (
        "Gradient shape should match t_boundaries shape"
    )
    assert jnp.all(jnp.isfinite(gradient_Ne)), (
        "All gradients w.r.t. Ne_values should be finite"
    )
    assert jnp.all(jnp.isfinite(gradient_t)), (
        "All gradients w.r.t. t_boundaries should be finite"
    )


if __name__ == "__main__":
    test_expected_ld_constant()
    test_expected_ld_piecewise_exponential()
    test_expected_ld_piecewise_constant()
    test_derivatives_computable()
    print("All tests passed!")
