import numpy as np

from .held import *

__doc__ = held.__doc__
if hasattr(held, "__all__"):
    __all__ = held.__all__


def ld_from_tree_sequence(ts, recombination_rate, bins=None, chunk_size=10_000):
    """
    Compute linkage disequilibrium statistics from a tskit tree sequence. If memory is limited, use this function to compute LD statistics in batches.

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
        Number of loci to process at a time. Default is 10,000.

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
    # Use tskit's Variant object for efficient decoding
    import tskit

    variant = tskit.Variant(ts)
    num_samples = ts.num_samples
    n_diploids = num_samples // 2
    num_sites = ts.num_sites
    positions_all = ts.sites_position.astype("int32")
    # Process in chunks
    # Preallocate chunk arrays
    genotype_buffer = np.empty((chunk_size, n_diploids), dtype=np.int32)
    for start_idx in range(0, num_sites, chunk_size):
        end_idx = min(start_idx + chunk_size, num_sites)
        chunk_len = end_idx - start_idx
        genotype_chunk = genotype_buffer[:chunk_len]
        # Decode variants efficiently
        for i, site_id in enumerate(range(start_idx, end_idx)):
            variant.decode(site_id)
            # Convert to diploid genotypes
            np.add(
                variant.genotypes[::2], variant.genotypes[1::2], out=genotype_chunk[i]
            )
        positions_chunk = positions_all[start_idx:end_idx]
        stats.update_batch(genotype_chunk, positions_chunk, bins)
    return stats.finalize(bins)
