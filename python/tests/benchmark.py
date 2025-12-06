import time

import held
import msprime
import numpy as np

# Set up benchmark parameters
n_runs = 5
timings = []

print("Running benchmark...")
for run in range(n_runs):
    # Simulate tree sequence
    ts = msprime.sim_ancestry(
        samples=50,
        population_size=1000,
        sequence_length=1e8,
        recombination_rate=1e-8,
        random_seed=run + 100,
    )
    mts = msprime.sim_mutations(ts, rate=5e-8, random_seed=run + 100)

    print(f"\nRun {run + 1}/{n_runs}:")
    print(f"  Number of sites: {mts.num_sites}")
    print(f"  Number of samples: {mts.num_samples}")

    # Time the LD computation
    start_time = time.time()
    result = held.ld_from_tree_sequence(mts, 1e-8)
    end_time = time.time()

    elapsed = end_time - start_time
    timings.append(elapsed)
    print(f"  Processing time: {elapsed:.3f} seconds")

print("\n" + "=" * 50)
print("BENCHMARK RESULTS")
print("=" * 50)
print(f"Number of runs: {n_runs}")
print(f"Mean processing time: {np.mean(timings):.3f} ± {np.std(timings):.3f} seconds")
print(f"Min time: {np.min(timings):.3f} seconds")
print(f"Max time: {np.max(timings):.3f} seconds")
