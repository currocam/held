UV_BINARY = "/data/antwerpen/210/vsc21003/bin/uv"
UV_TMPDIR = "/tmp/vsc21003/uv"
ENVMODULES = "calcua/2025a Rust/1.85.1-GCCcore-14.2.0"
import math


rule all:
    input:
        "plots/00-benchmark_ld_compute.pdf",
        "plots/01-msprime_simulation_prediction/all.pdf",

rule prediction_plot:
    input:
        expand(
            "results/pickles/msprime/constant_fixed/{Ne}.pkl",
            Ne=[500, 1000, 2000, 5000, 10_000],
        ),
        # Decline
        expand(
            "results/pickles/msprime/exponential_fixed/{Ne_c}_{Ne_a}_{t0}.pkl",
            Ne_c=[1000, 5000],
            Ne_a=[10_000],
            t0=[25, 50],
        ),
        # Expansion
        expand(
            "results/pickles/msprime/exponential_fixed/{Ne_c}_{Ne_a}_{t0}.pkl",
            Ne_c=[10_000, 5000],
            Ne_a=[1000],
            t0=[25, 50],
        ),
        # Invasion
        expand(
            "results/pickles/msprime/invasion_fixed/{Ne_c}_{Ne_a}_{t0}_{Ne_f}.pkl",
            Ne_c=[5000],
            Ne_a=[10_000],
            t0=[25, 50],
            Ne_f=[10, 100],
        ),
        # Carrying capacity
        expand(
            "results/pickles/msprime/carrying_capacity/{Ne_c}_{Ne_a}_{t0}_{t1}_{Ne_f}.pkl",
            Ne_c=[5000],
            Ne_a=[10_000],
            t0=[25, 50],
            t1=[75],
            Ne_f=[10, 100],
        ),
        # Three-epochs
        expand(
            "results/pickles/msprime/three_epochs_fixed/{Ne_1}_{Ne_2}_{Ne_3}_{t1}_{t2}.pkl",
            Ne_1=[10_000],
            Ne_2=[1_000, 5000],
            Ne_3=[10_000],
            t1=[25],
            t2=[50, 100],
        ),
        script=[
            "plots/01-msprime_simulation_prediction.py",
            "plots/01-msprime_simulation_zscores.py",
        ],
    output:
        multiext(
            "plots/01-msprime_simulation_prediction/",
            "all.pdf",
            "decline.pgf",
            "constant.pgf",
            "growth.pgf",
            "carrying_capacity.pgf",
        ),
        #"plots/01-msprime_simulation_prediction/z_scores.pdf",
    localrule: True
    shell:
        """
        export UV_CACHE_DIR="{UV_TMPDIR}"
        ./{input.script[0]}
        #./{input.script[1]}
        """

rule prediction_three_epochs_fixed:
    input:
        script="scripts/01-msprime_simulation_three_epochs_fixed.py",
    output:
        "results/pickles/msprime/three_epochs_fixed/{Ne_1}_{Ne_2}_{Ne_3}_{t1}_{t2}.pkl",
    envmodules:
        ENVMODULES,
    threads: 8
    localrule: False
    shell:
        """
        export UV_CACHE_DIR="{UV_TMPDIR}"
        {UV_BINARY} -n run --refresh --script {input.script} {wildcards.Ne_1} {wildcards.Ne_2} {wildcards.Ne_3} {wildcards.t1} {wildcards.t2} {output}
        """


rule prediction_constant_fixed:
    input:
        script="scripts/01-msprime_simulation_constant_fixed_mutations.py",
    output:
        "results/pickles/msprime/constant_fixed/{Ne}.pkl",
    envmodules:
        ENVMODULES,
    threads: 8
    localrule: False
    shell:
        """
        export UV_CACHE_DIR="{UV_TMPDIR}"
        {UV_BINARY} -n run --refresh --script {input.script} {wildcards.Ne} {output}
        """


rule prediction_exponential_fixed:
    input:
        script="scripts/01-msprime_simulation_exponential_fixed_mutations.py",
    output:
        "results/pickles/msprime/exponential_fixed/{Ne_c}_{Ne_a}_{t0}.pkl",
    envmodules:
        ENVMODULES,
    threads: 8
    localrule: False
    shell:
        """
        export UV_CACHE_DIR="{UV_TMPDIR}"
        {UV_BINARY} -n run --refresh --script {input.script} {wildcards.Ne_c} {wildcards.Ne_a} {wildcards.t0} {output}
        """


rule prediction_invasion_fixed:
    input:
        script="scripts/01-msprime_simulation_invasion_fixed_mutations.py",
    output:
        "results/pickles/msprime/invasion_fixed/{Ne_c}_{Ne_a}_{t0}_{Ne_f}.pkl",
    envmodules:
        ENVMODULES,
    threads: 8
    localrule: False
    shell:
        """
        export UV_CACHE_DIR="{UV_TMPDIR}"
        {UV_BINARY} -n run --refresh --script {input.script} {wildcards.Ne_c} {wildcards.Ne_a} {wildcards.t0} {wildcards.Ne_f} {output}
        """


rule benchmark_ld_compute:
    input:
        script="scripts/00-benchmark_ld_compute.py",
    output:
        "results/tables/00-benchmark_ld_compute.csv",
    envmodules:
        ENVMODULES,
    shell:
        """
        export UV_CACHE_DIR="{UV_TMPDIR}"
        ./{input.script} > {output}
        """


rule benchmark_ld_plot:
    input:
        script="plots/00-benchmark_ld_compute.py",
        data="results/tables/00-benchmark_ld_compute.csv",
    output:
        multiext("plots/00-benchmark_ld_compute", ".pdf", ".pgf"),
    shell:
        """
        export UV_CACHE_DIR="{UV_TMPDIR}"
        ./{input.script} < {input.data}
        """


rule plot_secondary_fixed:
    input:
        expand(
            "results/pickles/msprime/secondary_introduction/{Ne_c}_{Ne_f}_{Ne_a}_{t_0}_{t_1}_{migration}.pkl",
            Ne_c=[10_000, 5_000],
            Ne_f=[500],
            Ne_a=[10_000],
            t_0=[25],
            t_1=[50],
            migration=["low", "high"],
        ),


rule prediction_secondary_fixed:
    input:
        script="scripts/02-msprime_simulation_secondary_introduction.py",
    output:
        "results/pickles/msprime/secondary_introduction/{Ne_c}_{Ne_f}_{Ne_a}_{t_0}_{t_1}_{migration}.pkl",
    envmodules:
        ENVMODULES,
    threads: 8
    localrule: False
    params:
        migration_rate=lambda wildcards: dict(low=0.0005, high=0.01)[
            wildcards.migration
        ],
    shell:
        """
        export UV_CACHE_DIR="{UV_TMPDIR}"
        {UV_BINARY} -n run --refresh --script {input.script} {wildcards.Ne_c} {wildcards.Ne_f} {wildcards.Ne_a} {wildcards.t_0} {wildcards.t_1} {params.migration_rate} {output}
        """


rule prediction_carrying_capacity_fixed:
    input:
        script="scripts/01-msprime_simulation_three_epochs_carrying_capacity.py",
    output:
        "results/pickles/msprime/carrying_capacity/{Ne_c}_{Ne_a}_{t0}_{t1}_{Ne_f}.pkl",
    envmodules:
        ENVMODULES,
    threads: 8
    localrule: False
    params:
        alpha=lambda wildcards: (
            math.log(float(wildcards["Ne_c"])) - math.log(float(wildcards["Ne_f"]))
        )
        / float(wildcards["t0"]),
    shell:
        """
        export UV_CACHE_DIR="{UV_TMPDIR}"
        {UV_BINARY} -n run --refresh --script {input.script} {wildcards.Ne_c} {wildcards.Ne_a} {params.alpha} {wildcards.t0} {wildcards.t1} {output}
        """
