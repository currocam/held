import csv
import time

import held
import msprime
import numpy as np

# Benchmark parameters
n_samples_range = [10, 25, 50, 100, 200]
mutation_rates = [1e-8, 5e-8, 1e-7, 5e-7, 1e-6]
n_replicates = 3
sequence_length = 1e8
recombination_rate = 1e-8
population_size = 1000

results = []

print("Running preprocessing benchmark...")
print(f"Testing {len(n_samples_range)} sample sizes × {len(mutation_rates)} mutation rates × {n_replicates} replicates")
print("=" * 70)

total_runs = len(n_samples_range) * len(mutation_rates) * n_replicates
current_run = 0

for n_samples in n_samples_range:
    for mutation_rate in mutation_rates:
        for replicate in range(n_replicates):
            current_run += 1
            
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
            result = held.ld_from_tree_sequence(mts, recombination_rate)
            end_time = time.time()
            
            elapsed = end_time - start_time
            
            results.append({
                'n_samples': n_samples,
                'mutation_rate': mutation_rate,
                'n_mutations': mts.num_sites,
                'replicate': replicate,
                'runtime_seconds': elapsed
            })
            
            print(f"[{current_run}/{total_runs}] samples={n_samples}, mut_rate={mutation_rate:.1e}, "
                  f"n_sites={mts.num_sites}, time={elapsed:.3f}s")

# Write results to CSV
output_file = 'benchmark_results.csv'
with open(output_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['n_samples', 'mutation_rate', 'n_mutations', 'replicate', 'runtime_seconds'])
    writer.writeheader()
    writer.writerows(results)

print("\n" + "=" * 70)
print(f"Results written to {output_file}")
print(f"Total runs: {len(results)}")
print(f"Sample sizes: {n_samples_range}")
print(f"Mutation rates: {[f'{r:.1e}' for r in mutation_rates]}")
