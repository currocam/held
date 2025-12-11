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
from matplotlib.backends.backend_pdf import PdfPages

plt.style.use("science")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",  # Add math packages
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
def plot(files, labels, title):
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
        ax1.scatter(midpoints, datas[i]["mean"], alpha=0.7, color=colors[i])
        ax1.plot(
            midpoints, datas[i]["predictions"], label=labels[i], color=colors[i]
        )
    ax1.set_xlabel("Bin midpoint (Morgan)")
    ax1.set_ylabel(
        r"$\mathbb{E}[X_iX_jY_iY_j]$"
    )  # Fixed: use \mathbb{E} instead of \mathbb E
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
    plt.tight_layout(rect=(0.0, 0.05, 1.0, 0.96))

    return fig


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
files_invasion = [
    "results/pickles/msprime/invasion_fixed/5000_10000_25_10.pkl",
    "results/pickles/msprime/invasion_fixed/5000_10000_25_100.pkl",
    "results/pickles/msprime/invasion_fixed/5000_10000_50_10.pkl",
    "results/pickles/msprime/invasion_fixed/5000_10000_50_100.pkl",
]
labels_invasion = [
    r"$\{N_f=10,t_0=25\}$",
    r"$\{N_f=100,t_0=25\}$",
    r"$\{N_f=10,t_0=50\}$",
    r"$\{N_f=100,t_0=50\}$",
]
files_three= [
    "results/pickles/msprime/three_epochs_fixed/10000_1000_10000_25_50.pkl",
    "results/pickles/msprime/three_epochs_fixed/10000_1000_10000_25_100.pkl",
    "results/pickles/msprime/three_epochs_fixed/10000_5000_10000_25_50.pkl",
    "results/pickles/msprime/three_epochs_fixed/10000_5000_10000_25_100.pkl",
]
labels_three = [
    r"$\{N_c=1\mathrm{e}{3},t_1=50\}$",
    r"$\{N_c=1\mathrm{e}{3},t_1=100\}$",
    r"$\{N_c=5\mathrm{e}{3},t_1=50\}$",
    r"$\{N_c=5\mathrm{e}{3},t_1=100\}$",
]
files_carrying_capacity = [
    "results/pickles/msprime/carrying_capacity/5000_10000_25_75_100.pkl",
    "results/pickles/msprime/carrying_capacity/5000_10000_25_75_10.pkl",
    "results/pickles/msprime/carrying_capacity/5000_10000_50_75_10.pkl",
    "results/pickles/msprime/carrying_capacity/5000_10000_50_75_100.pkl",
]
labels_carrying_capacity = [
    r"$\{N_f=100,t_0=25\}$",
    r"$\{N_f=10,t_0=25\}$",
    r"$\{N_f=10,t_0=50\}$",
    r"$\{N_f=100,t_0=50\}$",
]
# %%
# Create all three plots and save to a single PDF
with PdfPages("plots/01-msprime_simulation_prediction/all.pdf") as pdf:
    fig1 = plot(files_decline, labels_decline, "Decline scenario")
    pdf.savefig(fig1)
    plt.close(fig1)

    fig2 = plot(files_constant, labels_constant, "Constant scenario")
    pdf.savefig(fig2)
    plt.close(fig2)

    fig3 = plot(files_growth, labels_growth, "Growth scenario")
    pdf.savefig(fig3)
    plt.close(fig3)

    fig4 = plot(files_invasion, labels_invasion, "Invasion scenario")
    pdf.savefig(fig4)
    plt.close(fig4)

    fig5 = plot(files_three, labels_three, "Bottleneck scenario")
    pdf.savefig(fig5)
    plt.close(fig5)

    fig6 = plot(files_carrying_capacity, labels_carrying_capacity, "Carrying capacity scenario")
    pdf.savefig(fig6)
    plt.close(fig6)
# %%
# Also save as PGF files (one per scenario)
fig1 = plot(files_decline, labels_decline, "Decline scenario")
fig1.savefig("plots/01-msprime_simulation_prediction/decline.pgf")
plt.close(fig1)

fig2 = plot(files_constant, labels_constant, "Constant scenario")
fig2.savefig("plots/01-msprime_simulation_prediction/constant.pgf")
plt.close(fig2)

fig3 = plot(files_growth, labels_growth, "Growth scenario")
fig3.savefig("plots/01-msprime_simulation_prediction/growth.pgf")
plt.close(fig3)

fig4 = plot(files_invasion, labels_invasion, "Invasion scenario")
fig4.savefig("plots/01-msprime_simulation_prediction/invasion.pgf")
plt.close(fig4)

fig5 = plot(files_three, labels_three, "Bottleneck scenario")
fig5.savefig("plots/01-msprime_simulation_prediction/bottleneck.pgf")
plt.close(fig5)

fig6 = plot(files_carrying_capacity, labels_carrying_capacity, "Carrying capacity scenario")
fig6.savefig("plots/01-msprime_simulation_prediction/carrying_capacity.pgf")
plt.close(fig6)

