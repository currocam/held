#!/usr/bin/env -S uv -n run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "demes==0.2.3",
#     "demesdraw==0.4.1",
#     "jax==0.8.1",
#     "matplotlib==3.10.7",
#     "msprime==1.3.4",
#     "numpy==2.3.5",
#     "pandas==2.3.3",
#     "scienceplots==2.2.0",
#     "seaborn==0.13.2",
# ]
# ///


__generated_with = "0.18.3"

# %%
# Initialization code that runs before all other cells
import scienceplots
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("science")
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
aspect_ratio = 6 / 8
scale = 1.0
width = textwidth * scale
height = width * aspect_ratio

# %%
import pandas as pd
import seaborn as sns
import demesdraw


# %%
def plot(files, labels, title, outfile):
    datas = [pd.read_pickle(file) for file in files]
    midpoints = (datas[0]["left_bins"] + datas[0]["right_bins"]) / 2
    # Create figure with two subplots side by side
    fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(width * 2, height))
    # Add overall title
    fig.suptitle(title)
    # Custom colors
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    # Left plot: Correlation data
    for i in range(4):
        ax1.scatter(midpoints, datas[i]["mean"], s=40, alpha=0.7, color=colors[i])
        ax1.plot(
            midpoints,
            datas[i]["mean"],
            label=labels[i],
            linewidth=2,
            color=colors[i],
        )
    ax1.set_xlabel("Bin midpoint (Morgan)")
    ax1.set_ylabel(r"$\mathbb E[X_iX_jY_iY_j]$")
    # Right plot: Demography
    handles = []
    for i in range(4):
        # Pass the figure explicitly to prevent new figure creation
        lc = demesdraw.size_history(datas[i]["demes"], ax=ax2, colours=colors[i])
        handles.append(plt.Line2D([], [], color=colors[i], label=labels[i]))

    # Force the axes to use reasonable limits
    ax2.autoscale(enable=True, tight=False)

    # Rotate y-axis label text
    ax2.set_ylabel("Population size", rotation=90, labelpad=5, va="center")
    ax2.set_xlabel("Time (generations ago)")
    # Shared legend at the bottom with space adjustment
    fig.legend(
        handles=handles, loc="lower center", ncols=4, bbox_to_anchor=(0.5, -0.02)
    )
    # Adjust layout to prevent legend overlap
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    # Close the figure after saving to prevent accumulation
    fig.savefig(outfile)
    return fig


# %%


# %%
files_decline = [
    "results/pickles/msprime/exponential_fixed/1000_10000_25.pkl",
    "results/pickles/msprime/exponential_fixed/5000_10000_25.pkl",
    "results/pickles/msprime/exponential_fixed/1000_10000_50.pkl",
    "results/pickles/msprime/exponential_fixed/5000_10000_50.pkl",
]
labels_decline = [
    r"$\{N_c=1\mathrm{e}{3},t_0=25\}$",
    r"$\{N_c=5\mathrm{e}{3},t_0=25\}$",
    r"$\{N_c=1\mathrm{e}{3},t_0=50\}$",
    r"$\{N_c=5\mathrm{e}{3},t_0=50\}$",
]

files_constant = [
    "results/pickles/msprime/constant_fixed/500.pkl",
    "results/pickles/msprime/constant_fixed/1000.pkl",
    "results/pickles/msprime/constant_fixed/2000.pkl",
    "results/pickles/msprime/constant_fixed/10000.pkl",
]
labels_constant = [
    r"$N_c=5\mathrm{e}{2}$",
    r"$N_c=1\mathrm{e}{3}$",
    r"$N_c=2\mathrm{e}{3}$",
    r"$N_c=1\mathrm{e}{4}$",
]

files_growth = [
    "results/pickles/msprime/exponential_fixed/10000_1000_25.pkl",
    "results/pickles/msprime/exponential_fixed/5000_1000_25.pkl",
    "results/pickles/msprime/exponential_fixed/10000_1000_50.pkl",
    "results/pickles/msprime/exponential_fixed/5000_1000_50.pkl",
]

labels_growth = [
    r"$\{N_c=1\mathrm{e}{4},t_0=25\}$",
    r"$\{N_c=5\mathrm{e}{3},t_0=25\}$",
    r"$\{N_c=1\mathrm{e}{4},t_0=50\}$",
    r"$\{N_c=5\mathrm{e}{3},t_0=50\}$",
]

# %%
# Concatenate in the order you want the rows to appear
files = files_decline + files_constant + files_growth
labels = labels_decline + labels_constant + labels_growth

# Preload midpoints from the first file (same for all)
data0 = pd.read_pickle(files[0])
midpoints = (data0["left_bins"] + data0["right_bins"]) / 2

# Grid layout: 3 rows × 4 columns
nrows = 3
ncols = 4

fig, axes = plt.subplots(
    nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True
)

axes = axes.flatten()

for ax, file, label in zip(axes, files, labels):
    data = pd.read_pickle(file)
    z_scores = (data["data"] - data["predictions"]) / data["std"]
    z_scores = z_scores.ravel()

    ax.axhline(-2, color="black", linestyle="--")
    ax.axhline(2, color="black", linestyle="--")

    sns.scatterplot(
        x=np.tile(midpoints, z_scores.size // midpoints.size), y=z_scores, ax=ax, s=10
    )

    ax.set_title(label, fontsize=10)

# Remove unused panels (in case fewer than 12 files)
for ax in axes[len(files) :]:
    ax.remove()

plt.tight_layout()
plt.savefig("plots/01-msprime_simulation_prediction/z_scores.pdf")
plt.show()
