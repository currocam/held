#!/usr/bin/env -S uv run --script
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

import jax
jax.config.update("jax_enable_x64", True)

NUM_WORKERS = 12
RANDOM_SEED = 1761846837
RECOMBINATION_RATE = 1e-8
SEQUENCE_LENGTH = int(1e8)
NUM_CHROMOSOMES = 50  # simulate 50 independent chromosomes
SAMPLE_SIZE = 100
NUM_MUTATIONS = 50_000
MAF = 0.25


def allele_frequencies(ts, sample_sets=None):
    if sample_sets is None:
        sample_sets = [ts.samples()]
    n = np.array([len(x) for x in sample_sets])

    def f(x):
        return x / n

    if ts.num_sites == 0:
        return np.asarray([])

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
        mask = np.bitwise_or(freqs < MAF, freqs > (1 - MAF))
        site_ids = np.asarray([site.id for site in ts.sites()])
        ts = ts.delete_sites(site_ids[mask])
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


def analysis(Ne_c, Ne_f, Ne_a, t_0, t_1, migration_rate):
    demography = msprime.Demography()
    demography.add_population(name="focal", initial_size=Ne_c)
    demography.add_population(name="source", initial_size=Ne_a)
    demography.add_population(name="ancestral", initial_size=Ne_a)
    demography.add_migration_rate_change(
        time=0, rate=migration_rate, source="focal", dest="source"
    )
    demography.add_population_parameters_change(
        population="focal", initial_size=Ne_f, time=t_0
    )
    demography.add_population_split(
        time=t_1, derived=["focal", "source"], ancestral="ancestral"
    )

    demes_graph = demography.to_demes()
    import multiprocess as mp

    left_bins_bp, right_bins_bp = held._construct_bins(RECOMBINATION_RATE, None, None)

    rng = np.random.RandomState(RANDOM_SEED)
    seeds = [rng.randint(1, 2**31 - 1) for _ in range(NUM_CHROMOSOMES)]
    worker_args = [
        (
            seed,
            {"focal": SAMPLE_SIZE},  # Sample from target population
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
        "demes": demography.to_demes(),
    }
    # Prediction from theory
    data["predictions"] = held.expected_ld_secondary_introduction(
        Ne_c=Ne_c,
        Ne_f=Ne_f,
        Ne_a=Ne_a,
        t0=t_0,
        t1=t_1,
        migration_rate=migration_rate,
        left_bins=data["left_bins"],
        right_bins=data["right_bins"],
        sample_size=SAMPLE_SIZE,
    )
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate secondary introduction scenario with continuous migration"
    )
    parser.add_argument(
        "Ne_c", type=float, help="Current effective population size of focal population"
    )
    parser.add_argument(
        "Ne_f",
        type=float,
        help="Effective population size of focal population at migration start",
    )
    parser.add_argument(
        "Ne_a",
        type=float,
        help="Effective population size of source/ancestral population",
    )
    parser.add_argument(
        "t_0", type=float, help="Time when migration starts (generations)"
    )
    parser.add_argument(
        "t_1", type=float, help="Time of population split (generations)"
    )
    parser.add_argument(
        "migration_rate",
        type=float,
        help="Continuous migration rate from source to focal",
    )
    parser.add_argument("outfile", type=str, help="Output pickle file")
    args = parser.parse_args()

    result = analysis(
        args.Ne_c, args.Ne_f, args.Ne_a, args.t_0, args.t_1, args.migration_rate
    )
    pd.to_pickle(result, args.outfile)
