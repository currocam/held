import held
import msprime
import numpy as np

ts = msprime.sim_ancestry(
    samples=10, population_size=5000, sequence_length=1e8, recombination_rate=1e-8
)
mts = msprime.sim_mutations(ts, rate=5e-8)
held.ld_from_tree_sequence(mts, 1e-8)
