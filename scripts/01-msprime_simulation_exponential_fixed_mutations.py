#!/usr/bin/env -S uv -n run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "held",
#     "msprime==1.3.4",
#     "pandas==2.3.3",
# ]
#
# [tool.uv.sources]
# held = { git = "https://github.com/currocam/held" }
# ///
import argparse

import held
import msprime
import numpy as np
import pandas as pd

NUM_WORKERS = 8
RANDOM_SEED = 237273
RECOMBINATION_RATE = 1e-8
SEQUENCE_LENGTH = int(2e8)
NUM_CHROMOSOMES = 50  # simulate 50 independent chromosomes
SAMPLE_SIZE = 100
NUM_MUTATIONS = 250_000
MAF = 0.25


def allele_frequencies(ts, sample_sets=None):
    if sample_sets is None:
        sample_sets = [ts.samples()]
    n = np.array([len(x) for x in sample_sets])

    def f(x):
        return x / n

    return ts.sample_count_stat(
        sample_sets,
        f,
        len(sample_sets),
        windows="sites",
        polarised=True,
        mode="site",
        strict=False,
        span_normalise=False,
    )


def worker(args):
    """Worker function for parallel chromosome simulation."""
    import msprime

    (
        seed,
        sample_size,
        demes_graph,
        sequence_length,
        recombination_rate,
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
    mu = 1e-15
    rng = np.random.default_rng(seed)
    while ts.num_sites < NUM_MUTATIONS:
        seed = rng.integers(1, 2**32 - 1, 1)
        ts = msprime.sim_mutations(ts, rate=mu, keep=True, random_seed=seed)
        freqs = allele_frequencies(ts).flatten()
        mask = np.bitwise_and(freqs > MAF, freqs < (1 - MAF))
        ts = ts.delete_sites(np.where(mask)[0])
        mu = mu * 2
    # Discard excess
    num_exceeded = ts.num_sites - NUM_MUTATIONS
    site_ids = [site.id for site in ts.sites()]
    discard_ids = rng.choice(site_ids, num_exceeded, replace=False)
    mts = ts.delete_sites(discard_ids)
    # Process the tree sequence
    return held.ld_from_tree_sequence(
        ts=mts,
        recombination_rate=recombination_rate,
    )[:, 0]


def analysis(Ne_c, Ne_a, t0, alpha):
    demo = msprime.Demography.isolated_model([Ne_c], growth_rate=[alpha])
    demo.add_population_parameters_change(time=t0, initial_size=Ne_a, growth_rate=0)
    demes_graph = demo.to_demes()
    import multiprocess as mp

    left_bins_bp, right_bins_bp = held._construct_bins(RECOMBINATION_RATE, None, None)

    rng = np.random.RandomState(RANDOM_SEED)
    seeds = [rng.randint(1, 2**31 - 1) for _ in range(NUM_CHROMOSOMES)]
    worker_args = [
        (
            seed,
            SAMPLE_SIZE,
            demes_graph,
            SEQUENCE_LENGTH,
            RECOMBINATION_RATE,
        )
        for seed in seeds
    ]
    with mp.Pool(NUM_WORKERS) as pool:
        from tqdm.auto import tqdm

        data = list(
            tqdm(
                pool.imap(worker, worker_args),
                total=NUM_CHROMOSOMES,
                desc="Simulating chromosomes",
                unit="chr",
                colour="green",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )
        )
    data_array = np.array(data)
    data = {
        "mean": data_array.mean(axis=0),
        "std": data_array.std(axis=0, ddof=1),
        "left_bins": left_bins_bp * RECOMBINATION_RATE,
        "right_bins": right_bins_bp * RECOMBINATION_RATE,
        "sample_size": SAMPLE_SIZE,
        "num_chromosomes": NUM_CHROMOSOMES,
        "num_mutations": NUM_MUTATIONS,
        "data": data_array,
        "demes": demo.to_demes(),
    }
    # Prediction from theory
    data["predictions"] = held.expected_ld_piecewise_exponential(
        Ne_c=Ne_c,
        Ne_a=Ne_a,
        t0=t0,
        alpha=alpha,
        left_bins=data["left_bins"],
        right_bins=data["right_bins"],
        sample_size=SAMPLE_SIZE,
    )
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("Ne_c", type=float, help="Current effective population size")
    parser.add_argument("Ne_a", type=float, help="Ancestral effective population size")
    parser.add_argument("t0", type=float, help="Time of population size change")
    parser.add_argument("outfile", type=str, help="Output pickle file")
    args = parser.parse_args()

    alpha = (np.log(args.Ne_c) - np.log(args.Ne_a)) / args.t0
    result = analysis(args.Ne_c, args.Ne_a, args.t0, alpha)
    pd.to_pickle(result, args.outfile)
