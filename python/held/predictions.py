"""Theoretical predictions and simulations for linkage disequilibrium."""

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from jax.scipy.special import exp1
from . import held
from .ld import _construct_bins


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


@jax.jit
def expected_sample_heterozygosity_constant(population_size, mu):
    """
    Compute the expected sample heterozygosity under a constant population size model.

    Args:
        population_size (float): Population size.
        mu (float):  Mutation rate per bp

    Returns:
        array: Expected sample heterozygosity values.
    """
    return 4 * population_size * mu


# Pre-compute quadrature rules for expected_ld_piecewise_exponential
_LEGENDRE_X_200, _LEGENDRE_W_200 = np.polynomial.legendre.leggauss(50)
_LEGENDRE_X_200 = jnp.asarray(_LEGENDRE_X_200)
_LEGENDRE_W_200 = jnp.asarray(_LEGENDRE_W_200)

# There's a singularity at alpha = 0. We do branch here if epsilon is small
ALPHA_EPSILON = 1e-7


@jax.jit
def expected_ld_piecewise_exponential(
    Ne_c,
    Ne_a,
    t0,
    alpha,
    left_bins,
    right_bins,
    sample_size=None,
):
    """
    Compute expected LD (E[X_iX_jY_iY_j]) under a two-phase exponential demography.

    Args:
        Ne_c (float): Contemporary diploid effective population size.
        Ne_a (float): Ancestral diploid effective population size.
        t0 (float): Time of transition from exponential to constant phase.
        alpha (float): Rate of change of Ne during the exponential phase.
        left_bins (array-like): Left distances for SNP pairs.
        right_bins (array-like): Right distances for SNP pairs.
        sample_size (int, optional): Number of diploid individuals. If provided, applies finite sample correction.

    Returns:
        array: Expected LD values across SNP distance bins.
    """
    u_i = jnp.asarray(left_bins)
    u_j = jnp.asarray(right_bins)

    def Su_piece1(alpha, Ne_c, t0, u):
        u = u[None, :]
        # If alpha is not close to zero
        t = (t0 - 0) / 2 * _LEGENDRE_X_200 + (t0 + 0) / 2
        t = t[:, None]
        inner1 = jnp.exp(
            (
                2 * t * alpha**2 * Ne_c
                - 4 * t * u * Ne_c * alpha
                - jnp.exp(t * alpha)
                + 1
            )
            / Ne_c
            / alpha
            / 2
        )
        integral_inner1 = jnp.sum(
            inner1 * _LEGENDRE_W_200[:, None] * (t0 - 0) / 2, axis=0
        )
        res1 = 1 / Ne_c * integral_inner1 / 2
        # If alpha is close to zero we use Taylor series
        res2 = (-jnp.exp(-t0 * (4 * Ne_c * u + 1) / Ne_c / 2) + 1) / (4 * Ne_c * u + 1)
        return jnp.where(jnp.abs(alpha) < ALPHA_EPSILON, res2, res1)

    # There is a closed-form solution for this piece
    def Su_piece2(alpha, Nec, Nea, t0, u):
        # Auto-generated code
        # fmt: off
        res1 = 1 / (4 * u * Nea + 1) * jnp.exp(-(4 * u * Nec * alpha * t0 + jnp.exp(t0 * alpha) - 1) / Nec / alpha / 2)
        # If alpha is close to zero we use Taylor expansion for alpha=0
        res2 = 1 / (4 * u * Nea + 1) * jnp.exp(-t0 * (4 * Nec * u + 1) / Nec / 2) - 1 / (4 * u * Nea + 1) * jnp.exp(-t0 * (4 * Nec * u + 1) / Nec / 2) * t0 ** 2 / Nec * alpha / 4
        # fmt: on
        return jnp.where(jnp.abs(alpha) < ALPHA_EPSILON, res2, res1)

    # Numerical integration using pre-computed Legendre quadrature
    u_points = jnp.array([gauss(a, b, 10)[0] for (a, b) in zip(u_i, u_j)])
    u_weights = jnp.array([gauss(a, b, 10)[1] / (b - a) for (a, b) in zip(u_i, u_j)])
    u_col = u_points.flatten()

    # First integral: [0, t0]
    integral_piece1 = Su_piece1(alpha, Ne_c, t0, u_col)
    # Second integral: [t0, ∞)
    integral_piece2 = Su_piece2(alpha, Ne_c, Ne_a, t0, u_col)
    res_flat = integral_piece1 + integral_piece2
    res_matrix = res_flat.reshape(u_points.shape)
    res_per_bin = jnp.sum(res_matrix * u_weights, axis=1)
    if sample_size is not None:
        return correct_ld_finite_sample(res_per_bin, sample_size)
    return res_per_bin


@jax.jit
def expected_sample_heterozygosity_piecewise_exponential(Ne_c, Ne_a, t0, alpha, mu):
    """
    Compute the expected sample heterozygosity under a two-phase exponential demography.

    Args:
        Ne_c (float): Contemporary diploid effective population size.
        Ne_a (float): Ancestral diploid effective population size.
        t0 (float): Time of transition from exponential to constant phase.
        alpha (float): Rate of change of Ne during the exponential phase.
        mu (float):  Mutation rate per bp

    Returns:
        array: Expected sample heterozygosity values.
    """
    w = (t0 - 0) / 2 * _LEGENDRE_W_200
    t = (t0 - 0) / 2 * _LEGENDRE_X_200 + (0 + t0) / 2
    piece1 = sum(
        w
        * t
        / Ne_c
        * jnp.exp((2 * t * alpha**2 * Ne_c - jnp.exp(t * alpha) + 1) / alpha / Ne_c / 2)
        / 2
    )
    piece2 = (2 * Ne_a + t0) * jnp.exp(-(-1 + jnp.exp(t0 * alpha)) / alpha / Ne_c / 2)
    piece1_taylor = (
        -2 * jnp.exp(-0.1e1 / Ne_c * t0 / 2) * Ne_c
        - jnp.exp(-0.1e1 / Ne_c * t0 / 2) * t0
        + 2 * Ne_c
    )
    piece2_taylor = (2 * Ne_a + t0) * jnp.exp(-1 / Ne_c * t0 / 2)
    e_tmrca_nonzero = piece1 + piece2
    e_tmrca_taylor = piece1_taylor + piece2_taylor
    # If alpha is too close to zero
    e_tmrca = jnp.where(jnp.abs(alpha) < 1e-7, e_tmrca_taylor, e_tmrca_nonzero)
    return e_tmrca * 2 * mu


@jax.jit
def expected_ld_exponential_carrying_capacity(
    Ne_c,
    Ne_a,
    alpha,
    t0,
    t1,
    left_bins,
    right_bins,
    sample_size=None,
):
    """
    Compute expected LD (E[X_iX_jY_iY_j]) under exponential growth followed by carrying capacity.

    Args:
        Ne_c (float): Contemporary diploid effective population size.
        Ne_a (float): Ancestral diploid effective population size.
        alpha (float): Rate of change of Ne during the exponential phase.
        t0 (float): Time when population reaches carrying capacity.
        t1 (float): Time when exponential phase begins.
        left_bins (array-like): Left distances for SNP pairs.
        right_bins (array-like): Right distances for SNP pairs.
        sample_size (int, optional): Number of diploid individuals. If provided, applies finite sample correction.

    Returns:
        array: Expected LD values across SNP distance bins.
    """
    u_i = jnp.asarray(left_bins)
    u_j = jnp.asarray(right_bins)
    # Numerical integration using pre-computed Legendre quadrature
    u_points = jnp.array([gauss(a, b, 10)[0] for (a, b) in zip(u_i, u_j)])
    u_weights = jnp.array([gauss(a, b, 10)[1] / (b - a) for (a, b) in zip(u_i, u_j)])
    u = u_points.flatten()
    # Auto-generated code
    # fmt: off
    Nec, Nea = Ne_c, Ne_a
    # Close-form pieces:
    piece1 = (1 - jnp.exp(-t0 * (4 * u * Nec + 1) / Nec / 2)) / (4 * u * Nec + 1)
    # There's a singularity at alpha=0
    piece3_nonzero = jnp.exp((1 - jnp.exp(-(t0 - t1) * alpha) - (4 * Nec * t1 * u + t0) * alpha) / alpha / Nec / 2) / (4 * u * Nea + 1)
    piece3_taylor = jnp.exp(-t1 * (4 * u * Nec + 1) / Nec / 2) / (4 * u * Nea + 1)
    piece3 = jnp.where(
        jnp.abs(alpha) < ALPHA_EPSILON, piece3_taylor, piece3_nonzero
    )
    # Numerical integration [t0, t1]
    times2 = (t1 - t0) / 2 * _LEGENDRE_X_200 + (t1 + t0) / 2
    def S_ut_piece2(alpha, Nec, t0, t, u):
        t = t[:, None]
        u = u[None, :]
        res_nonzero = 1 / Nec * jnp.exp((-jnp.exp((t - t0) * alpha) + 1 + (2 * t - 2 * t0) * Nec * alpha ** 2 + (-4 * Nec * t * u - t0) * alpha) / alpha / Nec / 2) / 2
        res_taylor = 1 / Nec * jnp.exp(-t * (4 * u * Nec + 1) / Nec / 2) / 2
        return jnp.where(
            jnp.abs(alpha) < ALPHA_EPSILON, res_taylor, res_nonzero
        )
    piece2 = jnp.sum(
        S_ut_piece2(alpha, Ne_c, t0, times2, u) * _LEGENDRE_W_200[:, None] * (t1 - t0) / 2, axis=0
    )
    # fmt: on
    res_flat = piece1 + piece2 + piece3
    res_matrix = res_flat.reshape(u_points.shape)
    res_per_bin = jnp.sum(res_matrix * u_weights, axis=1)
    if sample_size is not None:
        return correct_ld_finite_sample(res_per_bin, sample_size)
    return res_per_bin


@jax.jit
def expected_sample_heterozygosity_exponential_carrying_capacity(
    Ne_c, Ne_a, alpha, t0, t1, mu
):
    """
    Compute the expected sample heterozygosity under exponential growth followed by carrying capacity.

    Args:
        Ne_c (float): Contemporary diploid effective population size.
        Ne_a (float): Ancestral diploid effective population size.
        alpha (float): Rate of change of Ne during the exponential phase.
        t0 (float): Time when population reaches carrying capacity.
        t1 (float): Time when exponential phase begins.
        mu (float):  Mutation rate per bp

    Returns:
        array: Expected sample heterozygosity values.
    """
    t = (t1 - t0) / 2 * _LEGENDRE_X_200 + (t0 + t1) / 2
    w = (t1 - t0) / 2 * _LEGENDRE_W_200
    int_piece = sum(
        w
        * (
            t
            * jnp.exp(
                (
                    -jnp.exp((t - t0) * alpha)
                    + 1
                    + (2 * t - 2 * t0) * Ne_c * alpha**2
                    - t0 * alpha
                )
                / alpha
                / Ne_c
                / 2
            )
        )
    )
    expected_tmrca_nonzero = (
        (
            int_piece
            + (2 * t1 + 4 * Ne_a)
            * Ne_c
            * jnp.exp(
                -(t0 * alpha + jnp.exp(-(t0 - t1) * alpha) - 1) / alpha / Ne_c / 2
            )
            + (-4 * Ne_c**2 - 2 * Ne_c * t0) * jnp.exp(-1 / Ne_c * t0 / 2)
            + 4 * Ne_c**2
        )
        / Ne_c
        / 2
    )
    # Approaching to zero
    expected_tmrca_taylor = (2 * Ne_a - 2 * Ne_c) * jnp.exp(
        -1 / Ne_c * t1 / 2
    ) + 2 * Ne_c
    expected_tmrca = jnp.where(
        jnp.abs(alpha) < ALPHA_EPSILON, expected_tmrca_taylor, expected_tmrca_nonzero
    )
    return expected_tmrca * 2 * mu


@jax.jit
def expected_ld_piecewise_constant(
    Ne_values,
    t_boundaries,
    left_bins,
    right_bins,
    sample_size=None,
):
    """
    Compute expected LD (E[X_iX_jY_iY_j]) under a multi-epoch constant population size model.

    Args:
        Ne_values (array-like): Population sizes for each epoch. Shape: (n_epochs,)
        t_boundaries (array-like): Time boundaries between epochs. Shape: (n_epochs-1,)
            The first epoch runs from 0 to t_boundaries[0], second from t_boundaries[0] to t_boundaries[1], etc.
            The last epoch runs from t_boundaries[-1] to infinity.
        left_bins (array-like): Left distances for SNP pairs.
        right_bins (array-like): Right distances for SNP pairs.
        sample_size (int, optional): Number of diploid individuals. If provided, applies finite sample correction.

    Returns:
        array: Expected LD values across SNP distance bins.

    Examples:
        >>> # Two-epoch model: Ne=10000 from 0-1000 generations, Ne=5000 thereafter
        >>> Ne_values = jnp.array([10000.0, 5000.0])
        >>> t_boundaries = jnp.array([1000.0])
        >>> left_bins = jnp.array([0.0, 0.1, 0.2])
        >>> right_bins = jnp.array([0.1, 0.2, 0.3])
        >>> result = expected_ld_piecewise_constant(Ne_values, t_boundaries, left_bins, right_bins)
    """
    Ne_values = jnp.asarray(Ne_values)
    t_boundaries = jnp.asarray(t_boundaries)
    u_i = jnp.asarray(left_bins)
    u_j = jnp.asarray(right_bins)

    n_epochs = len(Ne_values)
    if len(t_boundaries) != n_epochs - 1:
        raise ValueError(
            f"Expected {n_epochs - 1} time boundaries for {n_epochs} epochs"
        )

    def S_ut_constant(Ne, Gamma_prev, t_prev, t, u):
        """Survival function for constant Ne epoch."""
        t = t[:, None]
        u = u[None, :]
        Gamma = Gamma_prev + (t - t_prev) / (2 * Ne)
        return jnp.exp(-2 * t * u - Gamma) / (2 * Ne)

    # Prepare quadrature for u
    u_points = jnp.array([gauss(a, b, 10)[0] for (a, b) in zip(u_i, u_j)])
    u_weights = jnp.array([gauss(a, b, 10)[1] / (b - a) for (a, b) in zip(u_i, u_j)])
    u_col = u_points.flatten()

    # Compute integrals for each epoch
    total_integral = jnp.zeros_like(u_col)
    Gamma_prev = 0.0
    t_prev = 0.0

    for epoch in range(n_epochs):
        Ne = Ne_values[epoch]

        if epoch < n_epochs - 1:
            # Finite interval [t_prev, t_curr]
            t_curr = t_boundaries[epoch]
            times = (t_curr - t_prev) / 2 * _LEGENDRE_X_200 + (t_curr + t_prev) / 2
            f_t = S_ut_constant(Ne, Gamma_prev, t_prev, times, u_col)
            integral = jnp.sum(
                f_t * _LEGENDRE_W_200[:, None] * (t_curr - t_prev) / 2, axis=0
            )

            # Update Gamma_prev for next epoch
            Gamma_prev = Gamma_prev + (t_curr - t_prev) / (2 * Ne)
            t_prev = t_curr
        else:
            # Last epoch: [t_prev, ∞)
            trans_legendre_x = 0.5 * _LEGENDRE_X_200 + 0.5
            trans_legendre_w = 0.5 * _LEGENDRE_W_200
            times = t_prev + trans_legendre_x / (1 - trans_legendre_x)
            f_t = S_ut_constant(Ne, Gamma_prev, t_prev, times, u_col)
            integral = jnp.sum(
                f_t
                * (trans_legendre_w[:, None] / (1 - trans_legendre_x)[:, None] ** 2),
                axis=0,
            )

        total_integral += integral

    res_matrix = total_integral.reshape(u_points.shape)
    res_per_bin = jnp.sum(res_matrix * u_weights, axis=1)

    if sample_size is not None:
        return correct_ld_finite_sample(res_per_bin, sample_size)
    return res_per_bin


@jax.jit
def expected_ld_secondary_introduction(
    Ne_c,
    Ne_f,
    Ne_a,
    t0,
    t1,
    migration_rate,
    left_bins,
    right_bins,
    sample_size=None,
):
    """
    Compute expected LD (E[X_iX_jY_iY_j]) under a three-phase island model with migration.

    Args:
        Ne_c (float): Contemporary diploid effective population size (migration activated).
        Ne_f (float): Intermediate diploid effective population size (migration activated).
        Ne_a (float): Ancestral diploid effective population size (no migration).
        t0 (float): Time of transition from contemporary to migration phase.
        t1 (float): Time of transition from migration to ancestral phase.
        migration_rate (float): Migration rate during the intermediate phase.
        left_bins (array-like): Left distances for SNP pairs.
        right_bins (array-like): Right distances for SNP pairs.
        sample_size (int, optional): Number of diploid individuals. If provided, applies finite sample correction.

    Returns:
        array: Expected LD values across SNP distance bins.
    """
    # Prepare quadrature for u
    u_i = jnp.asarray(left_bins)
    u_j = jnp.asarray(right_bins)
    # Some boiler-plate to avoid modifying the monster below as much as possible
    Nec, Nef, Nea, T1, T2, m = Ne_c, Ne_f, Ne_a, t0, t1, migration_rate
    # Auto-generated code
    # fmt: off
    res_per_bin =  8 * (-(Nea * Nec * m - Nec / 4 + Nea / 4) * (exp1(T2 * (4 * Nea * u_i + 1) / Nea / 2) - exp1(T2 * (4 * Nea * u_j + 1) / Nea / 2)) * Nef * (Nea * Nef * m + Nef ** 2 * m + Nea / 2 - Nef / 2) * (Nec * m + 0.1e1 / 0.2e1) * Nec * (Nea * m - 0.1e1 / 0.2e1) * jnp.exp((((-4 * Nef * T2 * m + T1 - T2) * Nec - Nef * T1) * Nea + T2 * Nec * Nef) / Nea / Nec / Nef / 2) - 2 * (Nec - Nef) * (Nea * Nec * m - Nec / 4 + Nea / 4) * (exp1(T2 * (4 * Nea * u_i + 1) / Nea / 2) - exp1(T2 * (4 * Nea * u_j + 1) / Nea / 2)) * Nef * (Nea * Nef * m + Nea / 4 - Nef / 4) * m * Nec * jnp.exp(((-2 * m * (T1 + T2) * Nec - T1) * Nea + T2 * Nec) / Nec / Nea / 2) + (Nec - Nef) * (exp1(T1 * (4 * Nea * u_i + 1) / Nea / 2) - exp1(T1 * (4 * Nea * u_j + 1) / Nea / 2)) * Nea ** 3 * (m * Nef + 0.1e1 / 0.2e1) * Nef * m ** 2 * (Nec * m + 0.1e1 / 0.2e1) * Nec * jnp.exp(-T1 * (4 * Nea * Nec * m + Nea - Nec) / Nec / Nea / 2) - 4 * Nea * (Nea * Nec * m - Nec / 4 + Nea / 4) * (exp1(2 * T1 * (0.1e1 / 0.4e1 + (m + u_i) * Nef) / Nef) - exp1(2 * T1 * (0.1e1 / 0.4e1 + (m + u_j) * Nef) / Nef) - exp1(2 * T2 * (0.1e1 / 0.4e1 + (m + u_i) * Nef) / Nef) + exp1(2 * T2 * (0.1e1 / 0.4e1 + (m + u_j) * Nef) / Nef)) * (m * Nef + 0.1e1 / 0.4e1) * (Nea * Nef * m + Nef ** 2 * m + Nea / 2 - Nef / 2) * (Nec * m + 0.1e1 / 0.2e1) * Nec * (Nea * m - 0.1e1 / 0.2e1) * jnp.exp((Nec - Nef) / Nef / Nec * T1 / 2) + 8 * Nef * (Nea * Nef * m + Nea / 4 - Nef / 4) * (Nea * (m * Nef + 0.1e1 / 0.2e1) * (Nec * m + 0.1e1 / 0.4e1) * ((Nec * m + 0.1e1 / 0.2e1) * Nea + Nec ** 2 * m - Nec / 2) * (Nea * m - 0.1e1 / 0.2e1) * exp1(2 * T1 * (0.1e1 / 0.4e1 + (m + u_i) * Nec) / Nec) / 2 - Nea * (m * Nef + 0.1e1 / 0.2e1) * (Nec * m + 0.1e1 / 0.4e1) * ((Nec * m + 0.1e1 / 0.2e1) * Nea + Nec ** 2 * m - Nec / 2) * (Nea * m - 0.1e1 / 0.2e1) * exp1(2 * T1 * (0.1e1 / 0.4e1 + (m + u_j) * Nec) / Nec) / 2 + jnp.exp(-T2 * (2 * Nea * m - 1) / Nea / 2) * (Nea * Nec * m - Nec / 4 + Nea / 4) * (m * Nef + 0.1e1 / 0.2e1) * m * Nec ** 2 * exp1(T2 * (4 * Nea * u_i + 1) / Nea / 2) / 2 - jnp.exp(-T2 * (2 * Nea * m - 1) / Nea / 2) * (Nea * Nec * m - Nec / 4 + Nea / 4) * (m * Nef + 0.1e1 / 0.2e1) * m * Nec ** 2 * exp1(T2 * (4 * Nea * u_j + 1) / Nea / 2) / 2 + (-(Nec - Nef) * (Nea * Nec * m - Nec / 4 + Nea / 4) * (exp1(T1 * (m + 2 * u_i)) - exp1(T1 * (m + 2 * u_j)) - exp1(T2 * (m + 2 * u_i)) + exp1(T2 * (m + 2 * u_j))) * m ** 2 * Nec * jnp.exp(-T1 * (2 * Nec * m + 1) / Nec / 2) / 2 + (m * Nef + 0.1e1 / 0.2e1) * ((Nec * m + 0.1e1 / 0.4e1) * ((Nec * m + 0.1e1 / 0.2e1) * Nea + Nec ** 2 * m - Nec / 2) * (Nea * m - 0.1e1 / 0.2e1) * jnp.log(1 + (4 * m + 4 * u_i) * Nec) / 2 - (Nec * m + 0.1e1 / 0.4e1) * ((Nec * m + 0.1e1 / 0.2e1) * Nea + Nec ** 2 * m - Nec / 2) * (Nea * m - 0.1e1 / 0.2e1) * jnp.log(1 + (4 * m + 4 * u_j) * Nec) / 2 + ((Nec / 4 - Nea * Nec * m - Nea / 4) * exp1(T2 * (m + 2 * u_i)) + (Nea * Nec * m - Nec / 4 + Nea / 4) * exp1(T2 * (m + 2 * u_j)) + Nea * (Nec * m + 0.1e1 / 0.2e1) * jnp.log(4 * Nea * u_i + 1) / 2 + Nea * (-Nec * m - 0.1e1 / 0.2e1) * jnp.log(4 * Nea * u_j + 1) / 2 + (jnp.log(m + 2 * u_j) - jnp.log(m + 2 * u_i)) * (Nea * Nec * m - Nec / 4 + Nea / 4)) * m ** 2 * Nec ** 2)) * Nea)) / Nec / Nea / Nef / (2 * Nea * m - 1) / (4 * Nea * Nec * m + Nea - Nec) / (2 * m * Nef + 1) / (4 * Nea * Nef * m + Nea - Nef) / (2 * Nec * m + 1) / (-u_j + u_i) # noqa
    # fmt: on
    if sample_size is not None:
        return correct_ld_finite_sample(res_per_bin, sample_size)
    return res_per_bin


@jax.jit
def expected_sample_heterozygosity_secondary_introduction(
    Ne_c, Ne_f, Ne_a, t0, t1, migration_rate, mu
):
    """
    Compute the expected sample heterozygosity under a three-phase island model with migration.

    Args:
        Ne_c (float): Contemporary diploid effective population size (migration activated).
        Ne_f (float): Intermediate diploid effective population size (migration activated).
        Ne_a (float): Ancestral diploid effective population size (no migration).
        t0 (float): Time of transition from contemporary to migration phase.
        t1 (float): Time of transition from migration to ancestral phase.
        migration_rate (float): Migration rate during the intermediate phase.
        mu (float):  Mutation rate per bp

    Returns:
        array: Expected sample heterozygosity values.
    """
    Nec, Nef, Nea, T1, T2, m = Ne_c, Ne_f, Ne_a, t0, t1, migration_rate
    expected_tmrca = (
        (
            32
            * (m * Nec + 0.1e1 / 0.2e1)
            * (m * Nec + 0.1e1 / 0.4e1)
            * (Nef * (Nea + Nef) * m + Nea / 2 - Nef / 2)
            * jnp.exp(((-4 * Nef * T2 * m + T1 - T2) * Nec - Nef * T1) / Nec / Nef / 2)
            + 64
            * (m * Nec + 0.1e1 / 0.4e1)
            * (m * Nef + 0.1e1 / 0.4e1)
            * (-Nef + Nec)
            * jnp.exp((-2 * m * (T1 + T2) * Nec - T1) / Nec / 2)
            + 128
            * (m * Nef + 0.1e1 / 0.2e1)
            * (
                -(m * Nec + 0.1e1 / 0.2e1)
                * (-Nef + Nec)
                * (Nea * m + 0.3e1 / 0.4e1)
                * jnp.exp(-T1 * (4 * m * Nec + 1) / Nec / 2)
                / 4
                + (
                    (-m * Nec - 0.1e1 / 0.4e1) * jnp.exp(-T2 * m)
                    + (m * Nec + 0.1e1 / 0.2e1) * (Nea * m + 0.3e1 / 0.4e1)
                )
                * (m * Nef + 0.1e1 / 0.4e1)
                * Nec
            )
        )
        / (4 * m * Nec + 1)
        / (2 * m * Nef + 1)
        / (4 * m * Nef + 1)
        / (2 * m * Nec + 1)
    )
    return expected_tmrca * 2 * mu


def _process_chromosome_worker(args: Tuple) -> dict:
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
    ld = stats.finalize(bins_obj)[:, 0]
    return dict(ld=ld, diversity=mts.diversity())


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

    from .ld import ld_from_tree_sequence

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
            ld = ld_from_tree_sequence(mts, recombination_rate, left_bins, right_bins)[
                :, 0
            ]
            data.append(dict(ld=ld, diversity=mts.diversity()))
    data_array = np.array([x["ld"] for x in data])
    diversity = np.array([x["diversity"] for x in data])
    return {
        "mean": data_array.mean(axis=0),
        "std": data_array.std(axis=0, ddof=1),
        "left_bins": left_bins_bp * recombination_rate,
        "right_bins": right_bins_bp * recombination_rate,
        "sample_size": sample_size,
        "num_chromosomes": num_chromosomes,
        "data": data_array,
        "diversity": diversity,
    }
