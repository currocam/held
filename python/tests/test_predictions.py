"""Test theoretical prediction functions."""

import held
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


if __name__ == "__main__":
    test_expected_ld_constant()
    test_expected_ld_piecewise_exponential()
    print("All tests passed!")
