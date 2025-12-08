from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from .held import *

__doc__ = held.__doc__
if hasattr(held, "__all__"):
    __all__ = held.__all__


# helper functions
def _construct_bins(
    recombination_rate: float,
    left_bins: Optional[NDArray],
    right_bins: Optional[NDArray],
) -> Tuple[NDArray, NDArray]:
    """Construct and validate bin arrays, converting from centiMorgans to base pairs."""
    if left_bins is None or right_bins is None:
        # Default binning scheme in cM
        left_bins_cm = np.arange(0.5, 0.5 * 20, 0.5)
        right_bins_cm = left_bins_cm + 0.5
    else:
        left_bins_cm = np.asarray(left_bins)
        right_bins_cm = np.asarray(right_bins)

    if len(left_bins_cm) != len(right_bins_cm):
        raise ValueError("left_bins and right_bins must have the same length")
    if np.any(right_bins_cm <= left_bins_cm):
        raise ValueError("right_bins must be greater than left_bins")

    # Convert from centiMorgans to base pairs
    left_bins_bp = left_bins_cm / 100 / recombination_rate
    right_bins_bp = right_bins_cm / 100 / recombination_rate

    return left_bins_bp, right_bins_bp


# Helper function to re-scale Legendre Gaussian quadrature rules
def gauss(a, b, n=10):
    """
    Compute nodes and weights for Gaussian quadrature over [a, b].

    Args:
        a (float): Lower bound of the integration interval.
        b (float): Upper bound of the integration interval.
        n (int, optional): Number of quadrature points. Defaults to 10.

    Returns:
        tuple: Tuple of arrays (nodes, weights) for Gaussian quadrature.
    """
    x, w = np.polynomial.legendre.leggauss(n)
    w = (b - a) / 2 * w
    x = (b - a) / 2 * x + (a + b) / 2
    return x, w


# Processing tree sequence data


def ld_from_tree_sequence(
    ts: Any,
    recombination_rate: float,
    left_bins: Optional[NDArray] = None,
    right_bins: Optional[NDArray] = None,
    chunk_size: int = 10_000,
) -> NDArray:
    """
    Compute linkage disequilibrium statistics from a tskit tree sequence.

    This function processes the tree sequence in chunks to compute LD statistics
    across distance bins. If memory is limited, adjust the chunk_size parameter.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence object to analyze
    recombination_rate : float
        Recombination rate per base pair per generation
    left_bins : NDArray, optional
        Left endpoints of distance bins in centiMorgans (cM).
        If not provided, uses default binning scheme (0.5 to 9.5 cM in 0.5 cM steps)
    right_bins : NDArray, optional
        Right endpoints of distance bins in centiMorgans (cM).
        If not provided, uses default binning scheme
    chunk_size : int, optional
        Number of loci to process at a time. Default is 10,000

    Returns
    -------
    NDArray
        Array with shape (n_bins, 3) containing [mean, variance, count]
        for each distance bin
    """
    left_bins_bp, right_bins_bp = _construct_bins(
        recombination_rate, left_bins, right_bins
    )
    bins_obj = held.Bins(left_bins_bp, right_bins_bp)
    n_bins = len(bins_obj)
    stats = held.StreamingStats(n_bins)

    # Use tskit's Variant object for efficient decoding
    import tskit

    variant = tskit.Variant(ts)
    num_samples = ts.num_samples
    n_diploids = num_samples // 2
    num_sites = ts.num_sites
    positions_all = ts.sites_position.astype("int32")

    # Preallocate chunk arrays
    genotype_buffer = np.empty((chunk_size, n_diploids), dtype=np.int32)

    # Process in chunks
    for start_idx in range(0, num_sites, chunk_size):
        end_idx = min(start_idx + chunk_size, num_sites)
        chunk_len = end_idx - start_idx
        genotype_chunk = genotype_buffer[:chunk_len]
        # Decode variants efficiently
        for i, site_id in enumerate(range(start_idx, end_idx)):
            variant.decode(site_id)
            genotype_chunk[i] = variant.genotypes[0::2] + variant.genotypes[1::2]
        positions_chunk = positions_all[start_idx:end_idx]
        stats.update_batch(genotype_chunk, positions_chunk, bins_obj)

    return stats.finalize(bins_obj)


# Predictions from theory


# Correction for finite sample size
def correct_ld_finite_sample(mu, sample_size):
    """
    Apply finite sample size correction proposed by Fournier et al. (2023).

    Args:
        mu (float): Computed mean of E[X_iX_jY_iY_j].
        sample_size (int): Number of diploid individuals.

    Returns:
        float: Corrected E[X_iX_jY_iY_j] value.
    """
    S = 2 * sample_size
    beta = 1 / (S - 1) ** 2
    alpha = ((S**2 - S + 2) ** 2) / ((S**2 - 3 * S + 2) ** 2)
    return (alpha - beta) * mu + 4 * beta


@jax.jit
def expected_ld_constant(population_size, left_bins, right_bins, sample_size=None):
    """
    Compute the expected linkage disequilibrium under a constant population size model.

    Args:
        population_size (float): Population size.
        left_bins (array-like): Left bin edges.
        right_bins (array-like): Right bin edges.
        sample_size (int, optional): Number of diploid individuals. Defaults to None.

    Returns:
        array: Expected LD values.
    """
    Ne = population_size
    u_i = jnp.asarray(left_bins)
    u_j = jnp.asarray(right_bins)
    # Expected LD constant for haploid data
    mu = (-jnp.log(4 * Ne * u_i + 1) + jnp.log(4 * Ne * u_j + 1)) / (
        4 * Ne * (u_j - u_i)
    )
    if sample_size is not None:
        return correct_ld_finite_sample(mu, sample_size)
    return mu


def _process_chromosome_worker(args: Tuple) -> NDArray:
    """Worker function for parallel chromosome simulation."""
    import msprime

    (
        seed,
        sample_size,
        demes_graph,
        sequence_length,
        recombination_rate,
        mutation_rate,
        left_bins_bp,
        right_bins_bp,
    ) = args

    # Reconstruct demography from demes graph
    demography = msprime.Demography.from_demes(demes_graph)

    ts = msprime.sim_ancestry(
        samples=sample_size,
        demography=demography,
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        random_seed=seed,
    )
    mts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=seed)

    # Create Bins object only within worker (not passed between processes)
    bins_obj = held.Bins(left_bins_bp, right_bins_bp)
    stats = held.StreamingStats(len(bins_obj))

    # Process the tree sequence
    import tskit

    variant = tskit.Variant(mts)
    num_samples = mts.num_samples
    n_diploids = num_samples // 2
    positions_all = mts.sites_position.astype("int32")

    genotypes = np.empty((mts.num_sites, n_diploids), dtype=np.int32)
    for i, site_id in enumerate(range(mts.num_sites)):
        variant.decode(site_id)
        genotypes[i] = variant.genotypes[0::2] + variant.genotypes[1::2]

    stats.update_batch(genotypes, positions_all, bins_obj)
    return stats.finalize(bins_obj)[:, 0]


def simulate_from_msprime(
    demography: Any,
    sample_size: int,
    sequence_length: float,
    recombination_rate: float,
    mutation_rate: float,
    random_seed: int,
    num_chromosomes: int = 1,
    left_bins: Optional[NDArray] = None,
    right_bins: Optional[NDArray] = None,
    progress_bar: bool = True,
    num_workers: int = 1,
) -> Dict[str, Any]:
    """
    Simulate linkage disequilibrium data from msprime demographic models.

    This function simulates multiple chromosomes under a given demographic model
    and computes LD statistics across distance bins. Supports parallel execution
    for faster computation.

    Parameters
    ----------
    demography : msprime.Demography
        An msprime Demography object specifying the demographic model
    sample_size : int
        Number of diploid individuals to sample
    sequence_length : float
        Length of each simulated chromosome in base pairs
    recombination_rate : float
        Recombination rate per base pair per generation
    mutation_rate : float
        Mutation rate per base pair per generation
    random_seed : int
        Random seed for reproducibility
    num_chromosomes : int, optional
        Number of independent chromosomes to simulate. Default is 1
    left_bins : NDArray, optional
        Left endpoints of distance bins in centiMorgans (cM).
        If not provided, uses default binning scheme
    right_bins : NDArray, optional
        Right endpoints of distance bins in centiMorgans (cM).
        If not provided, uses default binning scheme
    progress_bar : bool, optional
        Whether to display a progress bar. Default is True
    num_workers : int, optional
        Number of parallel workers. If > 1, uses multiprocessing. Default is 1

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'data': LD values array with shape (num_chromosomes, n_bins)
        - 'mean': Mean LD values across chromosomes for each bin
        - 'std': Standard deviation across chromosomes for each bin
        - 'left_bins': Left bin edges in recombination units
        - 'right_bins': Right bin edges in recombination units
        - 'sample_size': Number of diploid samples
        - 'num_chromosomes': Number of simulated chromosomes

    Examples
    --------
    >>> import msprime
    >>> import held
    >>> demography = msprime.Demography.isolated_model([10000])
    >>> data = held.simulate_from_msprime(
    ...     demography=demography,
    ...     sample_size=20,
    ...     sequence_length=1e7,
    ...     recombination_rate=1e-8,
    ...     mutation_rate=1e-8,
    ...     random_seed=42,
    ...     num_chromosomes=10,
    ...     num_workers=4
    ... )
    """
    import msprime
    import numpy as np

    left_bins_bp, right_bins_bp = _construct_bins(
        recombination_rate, left_bins, right_bins
    )

    if num_workers > 1:
        import multiprocess as mp

        # Generate seeds for each chromosome
        rng = np.random.RandomState(random_seed)
        seeds = [rng.randint(1, 2**31 - 1) for _ in range(num_chromosomes)]

        # Convert demography to demes for pickling (demes graphs are picklable)
        demes_graph = demography.to_demes()

        # Create argument tuples for each worker
        worker_args = [
            (
                seed,
                sample_size,
                demes_graph,
                sequence_length,
                recombination_rate,
                mutation_rate,
                left_bins_bp,
                right_bins_bp,
            )
            for seed in seeds
        ]

        # Process in parallel
        with mp.Pool(num_workers) as pool:
            if progress_bar:
                from tqdm.auto import tqdm

                data = list(
                    tqdm(
                        pool.imap(_process_chromosome_worker, worker_args),
                        total=num_chromosomes,
                        desc="Simulating chromosomes",
                        unit="chr",
                        colour="green",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                    )
                )
            else:
                data = list(pool.map(_process_chromosome_worker, worker_args))

    else:
        # Serial processing (original implementation)
        replicates = msprime.sim_ancestry(
            samples=sample_size,
            demography=demography,
            sequence_length=sequence_length,
            recombination_rate=recombination_rate,
            num_replicates=num_chromosomes,
            random_seed=random_seed,
        )
        data = []
        if progress_bar:
            from tqdm.auto import tqdm

            replicates = tqdm(
                replicates,
                total=num_chromosomes,
                desc="Simulating chromosomes",
                unit="chr",
                colour="green",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )
        for ts in replicates:
            mts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=random_seed)
            data.append(
                ld_from_tree_sequence(mts, recombination_rate, left_bins, right_bins)[
                    :, 0
                ]
            )

    data_array = np.array(data)
    return {
        "mean": data_array.mean(axis=0),
        "std": data_array.std(axis=0, ddof=1),
        "left_bins": left_bins_bp * recombination_rate,
        "right_bins": right_bins_bp * recombination_rate,
        "sample_size": sample_size,
        "num_chromosomes": num_chromosomes,
        "data": data_array,
    }
