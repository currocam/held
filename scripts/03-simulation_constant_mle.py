#!/usr/bin/env -S uv -n run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "held",
#     "msprime==1.3.4",
#     "msprime==1.3.4",
#     "optimistix==0.0.11",
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

RECOMBINATION_RATE = 1e-8
MUTATION_RATE = 1e-8
SEQUENCE_LENGTH = int(1e8)
MAF = 0.25

Ne = 10_000

    demo = msprime.Demography.isolated_model([Ne])
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
    data["predictions"] = held.expected_ld_constant(
        population_size=Ne,
        left_bins=data["left_bins"],
        right_bins=data["right_bins"],
        sample_size=SAMPLE_SIZE,
    )
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("Ne", type=int, help="Effective population size")
    parser.add_argument("outfile", type=str, help="Output pickle file")
    args = parser.parse_args()

    result = analysis(args.Ne)
    pd.to_pickle(result, args.outfile)
