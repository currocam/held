"""held - High-throughput Estimation of Linkage Disequilibrium."""

from .held import *
from .ld import ld_from_tree_sequence
from .predictions import (
    correct_ld_finite_sample,
    expected_ld_constant,
    expected_ld_piecewise_constant,
    expected_ld_piecewise_exponential,
    simulate_from_msprime,
)

__doc__ = held.__doc__
if hasattr(held, "__all__"):
    __all__ = held.__all__

__all__ += [
    "ld_from_tree_sequence",
    "correct_ld_finite_sample",
    "expected_ld_constant",
    "expected_ld_piecewise_constant",
    "expected_ld_piecewise_exponential",
    "simulate_from_msprime",
]
