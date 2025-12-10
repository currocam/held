UV_BINARY = "/data/antwerpen/210/vsc21003/bin/uv"
UV_TMPDIR = "/tmp/vsc21003/uv"
ENVMODULES = "calcua/2025a Rust/1.85.1-GCCcore-14.2.0"


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
        ),
        "plots/01-msprime_simulation_prediction/z_scores.pdf",
    shell:
        """
        export UV_CACHE_DIR="{UV_TMPDIR}"
        ./{input.script[0]}
        ./{input.script[1]}
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

