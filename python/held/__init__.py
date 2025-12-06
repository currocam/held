import numpy as np

from .held import *

__doc__ = held.__doc__
if hasattr(held, "__all__"):
    __all__ = held.__all__


def ld_from_tree_sequence(ts, recombination_rate, bins=None, chunk_size=10_000):
    """
    Compute linkage disequilibrium statistics from a tskit tree sequence.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence object to analyze
    recombination_rate : float
        Recombination rate.
    bins : tuple of 2-arrays, optional
        Left and right endpoints of distance bins for LD computation in Morgan.
        If not provided, uses the default binning scheme.
    chunk_size : int, optional
        Number of mutations to process at a time. Default is 10_000.
    Returns
    -------
    numpy.ndarray
        Array with shape (n_bins, 3) containing [mean, variance, count]
        for each distance bin
    """
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

    # Process variants in chunks
    genotype_chunk = []
    positions_chunk = []

    for variant in ts.variants():
        # Convert to diploid genotypes
        diploid_genotypes = variant.genotypes[::2] + variant.genotypes[1::2]
        genotype_chunk.append(diploid_genotypes)
        positions_chunk.append(int(variant.site.position))

        # Process chunk when it reaches chunk_size
        if len(genotype_chunk) >= chunk_size:
            genotype_matrix = np.array(genotype_chunk, dtype=np.int32)
            positions_array = np.array(positions_chunk, dtype=np.int32)
            stats.update_batch(genotype_matrix, positions_array, bins)
            genotype_chunk = []
            positions_chunk = []

    # Process remaining variants
    if genotype_chunk:
        genotype_matrix = np.array(genotype_chunk, dtype=np.int32)
        positions_array = np.array(positions_chunk, dtype=np.int32)
        stats.update_batch(genotype_matrix, positions_array, bins)

    return stats.finalize(bins)
