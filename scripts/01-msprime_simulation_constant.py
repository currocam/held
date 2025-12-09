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
import pandas as pd

NUM_WORKERS = 4
RANDOM_SEED = 1761846837
RECOMBINATION_RATE = 1e-8
MUTATION_RATE = 1e-8
SEQUENCE_LENGTH = int(2e8)  # must be an integer for msprime
NUM_CHROMOSOMES = 50  # simulate 50 independent chromosomes
SAMPLE_SIZE = 100


def analysis(Ne):
    demo = msprime.Demography.isolated_model([Ne])
    data = held.simulate_from_msprime(
        demography=demo,
        sample_size=SAMPLE_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        recombination_rate=RECOMBINATION_RATE,
        mutation_rate=MUTATION_RATE,
        random_seed=RANDOM_SEED,
        num_chromosomes=NUM_CHROMOSOMES,
        num_workers=NUM_WORKERS,
    )
    data["demes"] = demo.to_demes()
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
