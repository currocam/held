import held
import msprime
import multiprocess as mp

demo = msprime.Demography.isolated_model([5000])
data = held.simulate_from_msprime(
    demography=demo,
    sample_size=20,
    sequence_length=1e8,
    recombination_rate=1e-8,
    mutation_rate=1e-8,
    random_seed=46832746,
    num_chromosomes=8,
    num_workers=8,
)

prediction = held.expected_ld_constant(
    5000, data["left_bins"], data["right_bins"], data["sample_size"]
)
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_palette("colorblind")

sns.scatterplot(x=data["left_bins"], y=data["mean"], label="Simulation")
sns.lineplot(x=data["left_bins"], y=prediction, label="Prediction")

plt.xlabel("Position (cM)")
plt.ylabel("LD")
plt.title("Expected LD under constant demography")
plt.legend()
plt.show()
