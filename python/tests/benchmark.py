import held
import msprime
import numpy as np

ts = msprime.sim_ancestry(
    samples=10, population_size=5000, sequence_length=1e8, recombination_rate=1e-8
)
mts = msprime.sim_mutations(ts, rate=5e-8)

# Bins in centimorgans
left_bins = np.arange(0.5, 0.5 * 20, 0.5)
right_bins = left_bins + 0.5
# In bp
bins = held.Bins(left_bins * 1e6, right_bins * 1e6)

genotype_matrix = mts.genotype_matrix()
genotype_matrix = genotype_matrix[:, ::2] + genotype_matrix[:, 1::2]
positions = mts.sites_position.astype("int32")
stats = held.StreamingStats(19)
stats.update_batch(genotype_matrix, positions, bins)
stats.finalize(bins)
