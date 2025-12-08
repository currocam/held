# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib==3.10.7",
#     "numpy==2.3.5",
#     "pandas==2.3.3",
#     "plotnine==0.15.1",
#     "seaborn==0.13.2",
# ]
# ///

import marimo

__generated_with = "0.18.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    return pd, plt, sns


@app.cell
def _(pd):
    df = pd.read_csv("results/tables/00-benchmark_ld_compute.csv")
    df
    return (df,)


@app.cell
def _(df, sns):
    g = sns.scatterplot(data=df, x="n_mutations", y="runtime_seconds", hue="n_samples")
    g
    return


@app.cell
def _(df, plt, sns):
    sns.set(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(8, 6))
    for n, gr in df.groupby("n_samples"):
        sns.regplot(
            x=gr["n_mutations"] ** 2,
            y=gr["runtime_seconds"],
            scatter=True,
            label=f"n_samples = {n}",
            truncate=False,
            ax=ax,
        )

    # Labels, title, legend
    ax.set_xlabel("Mutations ^ 2")
    ax.set_ylabel("Runtime (seconds)")

    plt.tight_layout()
    plt.show()

    return


if __name__ == "__main__":
    app.run()
