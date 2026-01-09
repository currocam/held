"Check correctness from a simple python implementation."

import held
import numpy as np


def compute_ld_python(genotypes1, genotypes2):
    s = len(genotypes1)
    ld = 0.0
    ld_square = 0.0
    for i in range(s):
        prod = genotypes1[i] * genotypes2[i]
        ld += prod
        ld_square += prod * prod
    return (ld * ld - ld_square) / (s * (s - 1.0))


def standardize_genotypes(genotypes):
    total = genotypes.sum()
    allele_frequency = total / (2.0 * len(genotypes))
    maf = min(allele_frequency, 1.0 - allele_frequency)
    if maf < 0.25:
        return None
    mean = 2.0 * allele_frequency
    std = np.sqrt(2.0 * allele_frequency * (1.0 - allele_frequency))
    return (genotypes - mean) / std


def compute_ld_stats_python(genotypes, positions, bins):
    n_variants, n_samples = genotypes.shape
    n_bins = len(bins.left_bins)
    # Storage for statistics
    counts = np.zeros(n_bins, dtype=int)
    ld_sum = np.zeros(n_bins)
    ld_square_sum = np.zeros(n_bins)
    # Standardize all genotypes
    standardized = []
    valid_positions = []
    for i in range(n_variants):
        # Check for invalid genotypes
        if np.any(genotypes[i] < 0) or np.any(genotypes[i] > 2):
            continue
        std_geno = standardize_genotypes(genotypes[i])
        if std_geno is not None:
            standardized.append(std_geno)
            valid_positions.append(positions[i])
    # Compute LD for all pairs
    for i in range(len(standardized)):
        for j in range(i + 1, len(standardized)):
            distance = valid_positions[j] - valid_positions[i]
            # Find which bin this distance falls into
            for bin_idx in range(n_bins):
                if bins.left_bins[bin_idx] <= distance <= bins.right_bins[bin_idx]:
                    ld_value = compute_ld_python(standardized[i], standardized[j])
                    counts[bin_idx] += 1
                    ld_sum[bin_idx] += ld_value
                    ld_square_sum[bin_idx] += ld_value * ld_value
                    break
    # Compute mean and variance
    result = np.zeros((n_bins, 3))
    for i in range(n_bins):
        if counts[i] > 1:
            mean = ld_sum[i] / counts[i]
            # Welford's online algorithm variance
            variance = (
                ld_square_sum[i] - (ld_sum[i] * ld_sum[i]) / counts[i]
            ) / counts[i]
            result[i] = [mean, variance, counts[i]]
        else:
            result[i] = [np.nan, np.nan, counts[i]]
    return result


def test_linkage_disequilibrium_two_batches():
    n_samples = 50
    # First batch: 10 variants
    batch1_size = 10
    genotypes_batch1 = np.random.randint(
        0, 3, size=(batch1_size, n_samples), dtype=np.int32
    )
    positions_batch1 = np.arange(1000, 1000 + batch1_size * 100, 100, dtype=np.int32)
    batch2_size = 15
    genotypes_batch2 = np.random.randint(
        0, 3, size=(batch2_size, n_samples), dtype=np.int32
    )
    positions_batch2 = np.arange(2000, 2000 + batch2_size * 100, 100, dtype=np.int32)
    all_genotypes = np.vstack([genotypes_batch1, genotypes_batch2])
    all_positions = np.concatenate([positions_batch1, positions_batch2])
    bins = held.Bins([0.0, 100.0, 500.0], [99.0, 499.0, 1500.0])
    stats = held.StreamingStats(3)
    stats.update_batch(genotypes_batch1, positions_batch1, bins)
    stats.update_batch(genotypes_batch2, positions_batch2, bins)
    result_rust = stats.finalize(bins)
    result_python = compute_ld_stats_python(all_genotypes, all_positions, bins)
    for i in range(3):
        assert result_rust[i, 2] == result_python[i, 2], (
            f"Bin {i}: counts don't match (Rust: {result_rust[i, 2]}, Python: {result_python[i, 2]})"
        )
        if result_python[i, 2] > 1:
            np.testing.assert_allclose(
                result_rust[i, 0],
                result_python[i, 0],
                rtol=1e-5,
                atol=1e-8,
                err_msg=f"Bin {i}: means don't match",
            )
            np.testing.assert_allclose(
                result_rust[i, 1],
                result_python[i, 1],
                rtol=1e-5,
                atol=1e-8,
                err_msg=f"Bin {i}: variances don't match",
            )
        else:
            # Both should be NaN
            assert np.isnan(result_rust[i, 0]) and np.isnan(result_python[i, 0]), (
                f"Bin {i}: both means should be NaN when count <= 1"
            )
            assert np.isnan(result_rust[i, 1]) and np.isnan(result_python[i, 1]), (
                f"Bin {i}: both variances should be NaN when count <= 1"
            )


def test_simple_case():
    genotypes = np.array(
        [
            [2, 2, 2],  # pos 100
            [1, 1, 1],  # pos 200
            [2, 1, 0],  # pos 300
            [0, 2, 1],  # pos 500
        ],
        dtype=np.int32,
    )
    positions = np.array([100, 200, 300, 500], dtype=np.int32)
    bins = held.Bins([0.0, 100.0, 300.0], [99.0, 299.0, 600.0])
    stats = held.StreamingStats(3)
    stats.update_batch(genotypes, positions, bins)
    result_rust = stats.finalize(bins)
    result_python = compute_ld_stats_python(genotypes, positions, bins)
    np.testing.assert_allclose(
        result_rust[:, 2], result_python[:, 2], err_msg="Counts don't match"
    )
    for i in range(3):
        if result_python[i, 2] > 1:
            np.testing.assert_allclose(
                result_rust[i, :2],
                result_python[i, :2],
                rtol=1e-5,
                atol=1e-8,
                err_msg=f"Bin {i}: statistics don't match",
            )


def test_filtering():
    genotypes = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],  # pos 100, MAF=0.0
            [2, 2, 2, 2, 2, 2, 2, 2],  # pos 200, MAF=0.0
            [1, 1, 1, 1, 1, 1, 1, 1],  # pos 300, MAF=0.5
            [0, 0, 0, 0, 0, 0, 0, 2],  # pos 400, MAF=0.125
            [0, 0, 0, 0, 1, 1, 2, 2],  # pos 500, MAF=0.375
            [0, 0, 0, 0, 1, 1, 2, 3],  # Invalid because of 3
            [3, 0, 0, 0, 1, 1, 2, 0],  # Invalid because of 3
            [0, 0, 0, 0, 1, 1, -1, 0],  # Invalid because of -1
            [3, 0, 0, 0, 1, 1, -2, 0],  # Invalid because of both
        ],
        dtype=np.int32,
    )
    positions = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900], dtype=np.int32)
    bins = held.Bins([0.0, 100.0], [99.0, 300.0])
    stats = held.StreamingStats(2)
    stats.update_batch(genotypes, positions, bins)
    result_rust = stats.finalize(bins)
    result_python = compute_ld_stats_python(genotypes, positions, bins)
    assert result_rust[1, 2] == 1, "Should have exactly 1 pair after MAF filtering"
    np.testing.assert_allclose(
        result_rust, result_python, rtol=1e-5, atol=1e-8, equal_nan=True
    )


def test_ld_from_tree_sequence():
    import held
    import msprime

    ts = msprime.sim_ancestry(
        10, population_size=1000, recombination_rate=1e-8, sequence_length=2e7
    )
    res = held.ld_from_tree_sequence(ts, 1e-8)
    assert res.shape == (19, 3), "Should have 3 statistics and 19 bins"
    # All values should be nan
    assert np.isnan(res[:, :2]).all()
    assert (res[:, 2] == 0).all()
    # Add mutations
    mts = msprime.sim_mutations(ts, rate=1e-8)
    res = held.ld_from_tree_sequence(mts, 1e-8)
    assert res.shape == (19, 3), "Should have 3 statistics and 19 bins"

    # Simple function
    def no_chunk_ld(ts, recombination_rate, bins=None):
        if bins is None:
            left_bins = np.arange(0.5, 0.5 * 20, 0.5)
            right_bins = left_bins + 0.5
            bins = (left_bins, right_bins)
        elif len(bins) != 2:
            raise ValueError("bins must be a tuple of two arrays")
        elif len(bins[0]) != len(bins[1]):
            raise ValueError("bins must have the same length")
        elif np.any(bins[1] <= bins[0]):
            raise ValueError("bins must be increasing")
        bins = held.Bins(
            bins[0] / 100 / recombination_rate, bins[1] / 100 / recombination_rate
        )
        n_bins = len(bins)
        stats = held.StreamingStats(n_bins)
        genotype_matrix = ts.genotype_matrix()
        genotype_matrix = genotype_matrix[:, ::2] + genotype_matrix[:, 1::2]
        positions = ts.sites_position.astype("int32")
        stats.update_batch(genotype_matrix, positions, bins)
        return stats.finalize(bins)

    res2 = no_chunk_ld(mts, 1e-8)
    assert res.shape == res2.shape, "Chunk size should be equivalent"
    np.testing.assert_allclose(res, res2, rtol=1e-5, atol=1e-8, equal_nan=True)


def test_simulation_api():
    import held
    import msprime

    # Test 1: Simple isolated model
    demo = msprime.Demography.isolated_model([5000])
    data = held.simulate_from_msprime(
        demography=demo,
        sample_size=20,
        sequence_length=1e7,
        recombination_rate=1e-8,
        mutation_rate=1e-8,
        random_seed=46832746,
        num_chromosomes=3,
        num_workers=1,
    )

    # Check data is a dictionary
    assert isinstance(data, dict), "Result should be a dictionary"

    # Check required keys
    required_keys = [
        "mean",
        "std",
        "left_bins",
        "right_bins",
        "sample_size",
        "num_chromosomes",
        "data",
    ]
    for key in required_keys:
        assert key in data, f"Missing key: {key}"

    # Get length of bins
    n_bins = len(data["left_bins"])
    assert len(data["right_bins"]) == n_bins, (
        "left_bins and right_bins should have same length"
    )
    assert data["mean"].shape == (n_bins,), "mean should be a n_bins vector"
    assert data["std"].shape == (n_bins,), "std should be a n_bins vector"
    assert data["data"].shape == (3, n_bins), (
        "data should be a (num_chromosomes, n_bins) matrix"
    )
    assert data["sample_size"] == 20, "sample_size should match input"
    assert data["num_chromosomes"] == 3, "num_chromosomes should match input"

    # Test 2: Parallel execution
    data_parallel = held.simulate_from_msprime(
        demography=demo,
        sample_size=20,
        sequence_length=1e7,
        recombination_rate=1e-8,
        mutation_rate=1e-8,
        random_seed=999,  # Different seed for parallel test
        num_chromosomes=3,
        num_workers=2,
        progress_bar=False,
    )

    # Parallel execution should also produce valid results
    assert data_parallel["mean"].shape == (n_bins,), (
        "Parallel mean should be a n_bins vector"
    )
    assert data_parallel["num_chromosomes"] == 3, (
        "Parallel num_chromosomes should match"
    )
    assert data_parallel["data"].shape == (3, n_bins), (
        "Parallel data should be (num_chromosomes, n_bins)"
    )

    # Verify mean is computed correctly from raw data
    np.testing.assert_allclose(
        data["mean"], data["data"].mean(axis=0), rtol=1e-10, equal_nan=True
    )

    # Test 3: Custom bins
    custom_left = np.array([0.5, 1.5, 3.0])
    custom_right = np.array([1.5, 3.0, 5.0])
    data_custom = held.simulate_from_msprime(
        demography=demo,
        sample_size=20,
        sequence_length=1e7,
        recombination_rate=1e-8,
        mutation_rate=1e-8,
        random_seed=123,
        num_chromosomes=2,
        left_bins=custom_left,
        right_bins=custom_right,
        num_workers=1,
        progress_bar=False,
    )

    assert len(data_custom["mean"]) == 3, "Should have 3 bins"
    # Bins are in cM (input cM -> bp -> back to cM via * recombination_rate * 100)
    np.testing.assert_allclose(data_custom["left_bins"], custom_left / 100, rtol=1e-10)
    np.testing.assert_allclose(
        data_custom["right_bins"], custom_right / 100, rtol=1e-10
    )
