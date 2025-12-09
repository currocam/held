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
import sys
import time

import held
import msprime
import pandas as pd


def run_benchmark(n_samples, mutation_rate, replicate):
    """Run a single benchmark with the given parameters."""
    # Fixed parameters
    sequence_length = 1e8
    recombination_rate = 1e-8
    population_size = 1000

    # Simulate tree sequence
    ts = msprime.sim_ancestry(
        samples=n_samples,
        population_size=population_size,
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        random_seed=replicate + 100,
    )
    mts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=replicate + 100)

    # Time the preprocessing
    start_time = time.time()
    held.ld_from_tree_sequence(mts, recombination_rate)
    end_time = time.time()

    elapsed = end_time - start_time
    return {
        "n_samples": n_samples,
        "mutation_rate": mutation_rate,
        "n_mutations": mts.num_sites,
        "replicate": replicate,
        "runtime_seconds": elapsed,
    }


if __name__ == "__main__":
    n_samples = [10, 50, 100, 200]
    mutation_rates = [1e-8, 5e-8, 1e-7]
    num_replicates = 5

    data = []

    for n_sample in n_samples:
        for mutation_rate in mutation_rates:
            for replicate in range(num_replicates):
                res = run_benchmark(
                    n_sample,
                    mutation_rate,
                    replicate=replicate,
                )
                data.append(res)

    pd.DataFrame(data).to_csv(sys.stdout, index=False)
