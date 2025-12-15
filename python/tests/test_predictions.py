"""Test theoretical prediction functions."""

import held
import jax
import jax.numpy as jnp
import numpy as np


def test_expected_ld_constant():
    """Test that expected_ld_constant runs without errors and returns positive values."""
    rng = np.random.default_rng()

    # Random population size between 1,000 and 100,000
    population_size = rng.uniform(1000, 100000)

    # Random bins
    left_bins = jnp.array([0.0001, 0.1, 0.2, 0.3])
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
    rng = np.random.default_rng()

    # Random contemporary and ancestral population sizes between 1,000 and 100,000
    Ne_c = rng.uniform(1000, 100000)
    Ne_a = rng.uniform(1000, 100000)

    # Random transition time between 1 and 100 generations
    t0 = rng.uniform(1, 100)

    # Random growth rate between -0.01 and 0.01
    alpha = rng.uniform(-0.01, 0.01)

    # Random bins
    left_bins = jnp.array([0.001, 0.1, 0.2, 0.3])
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


def test_expected_ld_exponential_carrying_capacity():
    """Test that expected_ld_exponential_carrying_capacity runs without errors and returns positive values."""
    rng = np.random.default_rng()

    # Random contemporary and ancestral population sizes between 1,000 and 100,000
    Ne_c = rng.uniform(1000, 100000)
    Ne_a = rng.uniform(1000, 100000)

    # Random growth rate between -0.01 and 0.01
    alpha = rng.uniform(-0.01, 0.01)

    # Random transition times
    t0 = rng.uniform(10, 80)
    t1 = rng.uniform(t0 + 1, t0 + 40)  # t1 > t0

    # Random bins
    left_bins = jnp.array([0.001, 0.1, 0.2, 0.3])
    right_bins = jnp.array([0.1, 0.2, 0.3, 0.4])

    # Random sample size between 10 and 100
    sample_size = rng.integers(10, 100)

    # Test without sample size correction
    result = held.expected_ld_exponential_carrying_capacity(
        Ne_c, Ne_a, alpha, t0, t1, left_bins, right_bins
    )
    assert result.shape == (4,), f"Expected shape (4,), got {result.shape}"
    assert jnp.all(result > 0), "All LD values should be strictly greater than zero"

    # Test with sample size correction
    result_corrected = held.expected_ld_exponential_carrying_capacity(
        Ne_c, Ne_a, alpha, t0, t1, left_bins, right_bins, sample_size=sample_size
    )
    assert result_corrected.shape == (4,), (
        f"Expected shape (4,), got {result_corrected.shape}"
    )
    assert jnp.all(result_corrected > 0), (
        "All corrected LD values should be strictly greater than zero"
    )


def test_expected_ld_piecewise_constant():
    """Test piecewise constant population size model with multiple epochs."""
    rng = np.random.default_rng()

    # Random bins
    left_bins = jnp.array([0.001, 0.1, 0.2, 0.3])
    right_bins = jnp.array([0.1, 0.2, 0.3, 0.4])

    # Test 2-epoch model with random values
    Ne_values_2 = jnp.array([rng.uniform(5000, 15000), rng.uniform(15000, 25000)])
    t_boundaries_2 = jnp.array([rng.uniform(1, 100)])
    result_2 = held.expected_ld_piecewise_constant(
        Ne_values_2, t_boundaries_2, left_bins, right_bins
    )
    assert result_2.shape == (4,), f"Expected shape (4,), got {result_2.shape}"
    assert jnp.all(result_2 > 0), "All LD values should be strictly greater than zero"

    # Test 3-epoch model with random values
    Ne_values_3 = jnp.array(
        [rng.uniform(8000, 12000), rng.uniform(3000, 7000), rng.uniform(12000, 18000)]
    )
    t_boundaries_3 = jnp.array([rng.uniform(1, 50), rng.uniform(50, 100)])
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
        [rng.uniform(1, 35), rng.uniform(35, 80), rng.uniform(80, 200)]
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


def test_expected_ld_secondary_introduction():
    """Test that expected_ld_secondary_introduction runs without errors and returns positive values."""
    rng = np.random.default_rng()

    # Random population sizes between 1,000 and 100,000
    Ne_c = rng.uniform(1000, 100000)
    Ne_f = rng.uniform(1000, 100000)
    Ne_a = rng.uniform(1000, 100000)

    # Random transition times
    t0 = rng.uniform(10, 80)
    t1 = rng.uniform(t0 + 1, t0 + 40)  # t1 > t0

    # Random migration rate between 0.0001 and 0.001
    migration_rate = rng.uniform(0.0001, 0.01)

    # Random bins
    left_bins = jnp.array([0.001, 0.1, 0.2, 0.3])
    right_bins = jnp.array([0.1, 0.2, 0.3, 0.4])

    # Random sample size between 10 and 100
    sample_size = rng.integers(10, 100)

    # Test without sample size correction
    result = held.expected_ld_secondary_introduction(
        Ne_c, Ne_f, Ne_a, t0, t1, migration_rate, left_bins, right_bins
    )
    assert result.shape == (4,), f"Expected shape (4,), got {result.shape}"
    assert jnp.all(result > 0), "All LD values should be strictly greater than zero"
    assert jnp.all(jnp.isfinite(result)), "All LD values should be finite"

    # Test with sample size correction
    result_corrected = held.expected_ld_secondary_introduction(
        Ne_c,
        Ne_f,
        Ne_a,
        t0,
        t1,
        migration_rate,
        left_bins,
        right_bins,
        sample_size=sample_size,
    )
    assert result_corrected.shape == (4,), (
        f"Expected shape (4,), got {result_corrected.shape}"
    )
    assert jnp.all(result_corrected > 0), (
        "All corrected LD values should be strictly greater than zero"
    )
    assert jnp.all(jnp.isfinite(result_corrected)), (
        "All corrected LD values should be finite"
    )
    # Compare with panmintic population size
    Ne_constant = 10000.0
    result_constant_old = held.expected_ld_constant(Ne_constant, left_bins, right_bins)
    # With a very small migration rate, both predictions should be close
    epsilon_migration = 1e-10
    result_constant_new = held.expected_ld_secondary_introduction(
        Ne_constant,
        Ne_constant,
        Ne_constant,
        10,
        20,
        epsilon_migration,
        left_bins,
        right_bins,
    )
    # Warning: we reduce tolerance because for a single epoch we have a closed form solution
    assert jnp.allclose(result_constant_old, result_constant_new, rtol=0.001), (
        "Single epoch piecewise should match constant model"
    )
    # With a large migration rate, the predictions should be different
    epsilon_migration_large = 1e-2
    result_constant_new_large = held.expected_ld_secondary_introduction(
        Ne_constant,
        Ne_constant,
        Ne_constant,
        10,
        20,
        epsilon_migration_large,
        left_bins,
        right_bins,
    )
    assert not jnp.allclose(
        result_constant_old, result_constant_new_large, rtol=0.001
    ), "Large migration rate should lead to different predictions"


def test_derivatives_computable():
    """Test that derivatives can be computed for MLE optimization."""
    rng = np.random.default_rng()

    # Random bins
    left_bins = jnp.array([0.0001, 0.1, 0.2, 0.3])
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
    t0 = rng.uniform(1, 100)
    alpha = rng.uniform(-0.1, 0.1)

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
    t_boundaries = jnp.array([rng.uniform(1, 100)])

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

    # Test gradient of piecewise island model
    Ne_c = rng.uniform(5000, 15000)
    Ne_f = rng.uniform(5000, 15000)
    Ne_a = rng.uniform(15000, 25000)
    t0 = rng.uniform(1, 100)
    t1 = rng.uniform(t0 + 1, t0 + 100)
    migration_rate = rng.uniform(0.0001, 0.05)

    grad_fn_Ne_c = jax.grad(
        lambda x: jnp.sum(
            held.expected_ld_secondary_introduction(
                x, Ne_f, Ne_a, t0, t1, migration_rate, left_bins, right_bins
            )
        )
    )
    grad_fn_Ne_f = jax.grad(
        lambda x: jnp.sum(
            held.expected_ld_secondary_introduction(
                Ne_c, x, Ne_a, t0, t1, migration_rate, left_bins, right_bins
            )
        )
    )
    grad_fn_Ne_a = jax.grad(
        lambda x: jnp.sum(
            held.expected_ld_secondary_introduction(
                Ne_c, Ne_f, x, t0, t1, migration_rate, left_bins, right_bins
            )
        )
    )
    grad_fn_t0 = jax.grad(
        lambda x: jnp.sum(
            held.expected_ld_secondary_introduction(
                Ne_c, Ne_f, Ne_a, x, t1, migration_rate, left_bins, right_bins
            )
        )
    )
    grad_fn_t1 = jax.grad(
        lambda x: jnp.sum(
            held.expected_ld_secondary_introduction(
                Ne_c, Ne_f, Ne_a, t0, x, migration_rate, left_bins, right_bins
            )
        )
    )
    grad_fn_m = jax.grad(
        lambda x: jnp.sum(
            held.expected_ld_secondary_introduction(
                Ne_c, Ne_f, Ne_a, t0, t1, x, left_bins, right_bins
            )
        )
    )

    gradient_Ne_c = grad_fn_Ne_c(Ne_c)
    gradient_Ne_f = grad_fn_Ne_f(Ne_f)
    gradient_Ne_a = grad_fn_Ne_a(Ne_a)
    gradient_t0 = grad_fn_t0(t0)
    gradient_t1 = grad_fn_t1(t1)
    gradient_m = grad_fn_m(migration_rate)

    assert jnp.isfinite(gradient_Ne_c), "Gradient w.r.t. Ne_c should be finite"
    assert jnp.isfinite(gradient_Ne_f), "Gradient w.r.t. Ne_f should be finite"
    assert jnp.isfinite(gradient_Ne_a), "Gradient w.r.t. Ne_a should be finite"
    assert jnp.isfinite(gradient_t0), "Gradient w.r.t. t0 should be finite"
    assert jnp.isfinite(gradient_t1), "Gradient w.r.t. t1 should be finite"
    assert jnp.isfinite(gradient_m), "Gradient w.r.t. migration_rate should be finite"


def test_expected_sample_heterozygosity_constant_montecarlo():
    """Test expected_sample_heterozygosity_constant using Monte Carlo simulation with msprime."""
    import msprime

    # Test parameters
    Ne = 10000.0
    mu = 1
    num_replicates = 25_000
    random_seed = 42

    # Theoretical prediction
    expected_het = held.expected_sample_heterozygosity_constant(Ne, mu)

    # Monte Carlo simulation using diversity statistic
    demography = msprime.Demography.isolated_model([Ne])

    diversities = []
    for ts in msprime.sim_ancestry(
        samples=1,
        demography=demography,
        num_replicates=num_replicates,
        random_seed=random_seed,
    ):
        diversities.append(ts.diversity(mode="branch") / 2)
    diversities = np.asarray(diversities)
    # Monte Carlo estimate
    mc_estimate = float(np.mean(diversities * 2 * mu))
    # Check that the theoretical prediction matches the Monte Carlo estimate
    # Using 3 standard errors for 99.7% confidence
    se = np.std(diversities) / np.sqrt(num_replicates)
    assert abs(expected_het - mc_estimate) < 3 * se, (
        f"Expected {expected_het}, got {mc_estimate} ± {3 * se}"
    )


def test_expected_sample_heterozygosity_piecewise_exponential_montecarlo():
    """Test expected_sample_heterozygosity_piecewise_exponential using Monte Carlo simulation."""
    import msprime

    # Test parameters - growth scenario
    Ne_c = 5000.0
    Ne_a = 20000.0
    t0 = 100.0
    alpha = 0.005  # Positive growth rate
    mu = 1
    num_replicates = 50_000
    random_seed = 123

    # Theoretical prediction
    expected_het = held.expected_sample_heterozygosity_piecewise_exponential(
        Ne_c, Ne_a, t0, alpha, mu
    )

    # Create demographic model
    demography = msprime.Demography()
    demography.add_population(name="pop", initial_size=Ne_c)
    demography.add_population_parameters_change(
        time=0, initial_size=Ne_c, growth_rate=alpha
    )
    demography.add_population_parameters_change(
        time=t0, initial_size=Ne_a, growth_rate=0
    )

    # Monte Carlo simulation using diversity
    diversities = []
    for ts in msprime.sim_ancestry(
        samples=1,
        demography=demography,
        num_replicates=num_replicates,
        random_seed=random_seed,
    ):
        diversities.append(ts.diversity(mode="branch") / 2)
    diversities = np.asarray(diversities)
    # Monte Carlo estimate
    mc_estimate = float(np.mean(diversities * 2 * mu))
    se = np.std(diversities) / np.sqrt(num_replicates)

    assert abs(expected_het - mc_estimate) < 3 * se, (
        f"Expected {expected_het}, got {mc_estimate} ± {3 * se}"
    )


def test_expected_sample_heterozygosity_exponential_carrying_capacity_montecarlo():
    """Test expected_sample_heterozygosity_exponential_carrying_capacity using Monte Carlo simulation."""
    import msprime

    # Test parameters
    Ne_c = 8000.0
    Ne_a = 25000.0
    alpha = 0.015
    t0 = 50.0
    t1 = 150.0
    mu = 1
    num_replicates = 25000
    random_seed = 456

    # Theoretical prediction
    expected_het = held.expected_sample_heterozygosity_exponential_carrying_capacity(
        Ne_c, Ne_a, alpha, t0, t1, mu
    )

    # Create demographic model
    demography = msprime.Demography()
    demography.add_population(name="pop", initial_size=Ne_c)
    # Phase 1: Constant at Ne_c from 0 to t0
    # Phase 2: Exponential growth from t0 to t1
    demography.add_population_parameters_change(
        time=t0, initial_size=Ne_c, growth_rate=alpha, population="pop"
    )
    # Phase 3: Constant at Ne_a from t1 onwards
    demography.add_population_parameters_change(
        time=t1, initial_size=Ne_a, growth_rate=0, population="pop"
    )

    # Monte Carlo simulation using diversity
    diversities = []
    for ts in msprime.sim_ancestry(
        samples=1,
        demography=demography,
        num_replicates=num_replicates,
        random_seed=random_seed,
    ):
        diversities.append(ts.diversity(mode="branch") / 2)
    diversities = np.asarray(diversities)
    # Monte Carlo estimate
    mc_estimate = float(np.mean(diversities * 2 * mu))
    se = np.std(diversities) / np.sqrt(num_replicates)

    assert abs(expected_het - mc_estimate) < 3 * se, (
        f"Expected {expected_het}, got {mc_estimate} ± {3 * se}"
    )


def test_expected_sample_heterozygosity_secondary_introduction_montecarlo():
    """Test expected_sample_heterozygosity_secondary_introduction using Monte Carlo simulation."""
    import msprime

    # Test parameters
    Ne_c = 10000.0
    Ne_f = 8000.0
    Ne_a = 20000.0
    t0 = 50.0
    t1 = 200.0
    migration_rate = 0.001
    mu = 1
    num_replicates = 25000
    random_seed = 789

    # Theoretical prediction
    expected_het = held.expected_sample_heterozygosity_secondary_introduction(
        Ne_c, Ne_f, Ne_a, t0, t1, migration_rate, mu
    )

    # Create demographic model with island model (migration between two populations)
    demography = msprime.Demography()
    demography.add_population(name="pop1", initial_size=Ne_c)
    demography.add_population(name="pop2", initial_size=Ne_c)

    # Phase 1: Contemporary phase with no migration (0 to t0)
    # Phase 2: Migration phase (t0 to t1)
    demography.add_population_parameters_change(
        time=t0, initial_size=Ne_f, population="pop1"
    )
    demography.add_population_parameters_change(
        time=t0, initial_size=Ne_f, population="pop2"
    )
    demography.add_migration_rate_change(
        time=t0, rate=migration_rate, source="pop1", dest="pop2"
    )
    demography.add_migration_rate_change(
        time=t0, rate=migration_rate, source="pop2", dest="pop1"
    )

    # Phase 3: Ancestral phase - populations merge (t1 onwards)
    demography.add_population_parameters_change(
        time=t1, initial_size=Ne_a, population="pop1"
    )
    demography.add_population_split(time=t1, derived=["pop2"], ancestral="pop1")

    # Monte Carlo simulation - sample from same population using diversity
    diversities = []
    for ts in msprime.sim_ancestry(
        samples={"pop1": 1},
        demography=demography,
        num_replicates=num_replicates,
        random_seed=random_seed,
    ):
        diversities.append(ts.diversity(mode="branch") / 2)
    diversities = np.asarray(diversities)
    # Monte Carlo estimate
    mc_estimate = float(np.mean(diversities * 2 * mu))
    se = np.std(diversities) / np.sqrt(num_replicates)

    assert abs(expected_het - mc_estimate) < 3 * se, (
        f"Expected {expected_het}, got {mc_estimate} ± {3 * se}"
    )


if __name__ == "__main__":
    test_expected_ld_constant()
    test_expected_ld_piecewise_exponential()
    test_expected_ld_exponential_carrying_capacity()
    test_expected_ld_piecewise_constant()
    test_expected_ld_secondary_introduction()
    test_derivatives_computable()
    test_expected_sample_heterozygosity_constant_montecarlo()
    test_expected_sample_heterozygosity_piecewise_exponential_montecarlo()
    test_expected_sample_heterozygosity_exponential_carrying_capacity_montecarlo()
    test_expected_sample_heterozygosity_secondary_introduction_montecarlo()
    print("All tests passed!")
