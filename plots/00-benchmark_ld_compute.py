#!/usr/bin/env -S uv -n run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "arviz-plots==0.7.0",
#     "matplotlib==3.10.7",
#     "matplotlib-label-lines",
#     "numpy==2.3.5",
#     "pandas==2.3.3",
#     "plotnine==0.15.1",
#     "scienceplots==2.2.0",
#     "seaborn==0.13.2",
# ]
# ///


__generated_with = "0.18.3"

# %%
# Initialization code that runs before all other cells
import matplotlib
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)

# Aesthetics
textwidth = 3.31314
aspect_ratio = 6/8
scale = 1.0
width = textwidth * scale
height = width * aspect_ratio

# %%
import seaborn as sns
import numpy as np
import pandas as pd

# %%
df = pd.read_csv("results/tables/00-benchmark_ld_compute.csv")
df = df[df["n_samples"] > 10]
df

# %%
fig, ax = plt.subplots()
for n, gr in df.groupby("n_samples"):
    sns.regplot(
        x=gr["n_mutations"] ** 2,
        y=gr["runtime_seconds"],
        scatter=True,
        label=f"N. diploids = {n}",
        ax=ax,
    )
# Labels, title, legend
ax.set_xlabel(r"Squared \#mutations ($n^2$)")
ax.set_ylabel("Runtime (seconds)")
ax.set_title("Quadratic runtime")
plt.legend()
plt.tight_layout()
plt.savefig("plots/00-benchmark_ld_compute.pdf")
plt.show()

# %%
matplotlib.use("pgf")
plt.savefig("plots/00-benchmark_ld_compute.pgf")
