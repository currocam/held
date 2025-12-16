#!/usr/bin/env -S uv run --script
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

# Initialization code that runs before all other cells
import scienceplots
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import demesdraw

plt.style.use("science")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    }
)

# Aesthetics
textwidth = 3.31314
aspect_ratio = 6 / 8
scale = 1.0
width = textwidth * scale
height = width * aspect_ratio


def plot(file, label, color):
    data = pd.read_pickle(file)
    midpoints = (data["left_bins"] + data["right_bins"]) / 2

    # Create figure with two subplots side by side
    fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(width * 2, height))

    # Left plot: Correlation data
    ax1.scatter(midpoints, data["mean"], s=40, alpha=0.7, color=color)
    ax1.plot(midpoints, data["predictions"], label=label, linewidth=2, color=color)
    ax1.set_xlabel("Bin midpoint (Morgan)")
    ax1.set_xscale("log")
    ax1.set_ylabel(r"$\mathbb{E}[X_iX_jY_iY_j]$")
    ax1.legend()

    # Right plot: Demography
    demesdraw.tubes(data["demes"], ax=ax2, colours=color)
    ax2.autoscale(enable=True, tight=False)
    ax2.set_ylabel("Population size", rotation=90, labelpad=5, va="center")
    ax2.set_xlabel("Time (generations ago)")

    plt.tight_layout()
    return fig


# Define files and their properties
files = [
    "results/pickles/msprime/secondary_introduction/10000_500_10000_25_50_high.pkl",
    "results/pickles/msprime/secondary_introduction/10000_500_10000_25_50_low.pkl",
    "results/pickles/msprime/secondary_introduction/5000_500_10000_25_50_high.pkl",
    "results/pickles/msprime/secondary_introduction/5000_500_10000_25_50_low.pkl",
]

labels = [
    r"$\{N_c=1\mathrm{e}{5},m=1\mathrm{e}{-2}\}$",
    r"$\{N_c=1\mathrm{e}{5},m=5\mathrm{e}{-4}\}$",
    r"$\{N_c=5\mathrm{e}{4},m=1\mathrm{e}{-2}\}$",
    r"$\{N_c=5\mathrm{e}{4},m=5\mathrm{e}{-4}\}$",
]

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

# Create one PDF file per plot
for i, (file, label, color) in enumerate(zip(files, labels, colors)):
    fig = plot(file, label, color)

    # Save as PDF
    outfile = f"plots/02-msprime_simulation_secondary_introduction/scenario_{i + 1}.pdf"
    fig.savefig(outfile)

    # Save as PGF
    outfile_pgf = (
        f"plots/02-msprime_simulation_secondary_introduction/scenario_{i + 1}.pgf"
    )
    fig.savefig(outfile_pgf)

    plt.close(fig)

print("All plots saved successfully!")
