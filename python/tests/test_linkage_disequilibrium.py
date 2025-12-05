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


def test_maf_filtering():
    genotypes = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],  # pos 100, MAF=0.0
            [2, 2, 2, 2, 2, 2, 2, 2],  # pos 200, MAF=0.0
            [1, 1, 1, 1, 1, 1, 1, 1],  # pos 300, MAF=0.5
            [0, 0, 0, 0, 0, 0, 0, 2],  # pos 400, MAF=0.125
            [0, 0, 0, 0, 1, 1, 2, 2],  # pos 500, MAF=0.375
        ],
        dtype=np.int32,
    )
    positions = np.array([100, 200, 300, 400, 500], dtype=np.int32)
    bins = held.Bins([0.0, 100.0], [99.0, 300.0])
    stats = held.StreamingStats(2)
    stats.update_batch(genotypes, positions, bins)
    result_rust = stats.finalize(bins)
    result_python = compute_ld_stats_python(genotypes, positions, bins)
    assert result_rust[1, 2] == 1, "Should have exactly 1 pair after MAF filtering"
    np.testing.assert_allclose(
        result_rust, result_python, rtol=1e-5, atol=1e-8, equal_nan=True
    )
