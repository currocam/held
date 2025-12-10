"""Linkage disequilibrium computation from tree sequences."""

from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from . import held


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
