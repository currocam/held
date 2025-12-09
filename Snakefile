UV_BINARY = "/data/antwerpen/210/vsc21003/bin/uv"
UV_TMPDIR = "/tmp/vsc21003/uv"
ENVMODULES = "calcua/2025a Rust/1.85.1-GCCcore-14.2.0"


rule all:
    input:
        "plots/00-benchmark_ld_compute.pdf",
        expand(
            "results/pickles/msprime/constant_fixed/{Ne}.pkl",
            Ne=[500, 1000, 2000, 5000, 10_000],
        ),


rule prediction_constant_fixed:
    input:
        script="scripts/01-msprime_simulation_constant_fixed_mutations.py",
    output:
        "results/pickles/msprime/constant_fixed/{Ne}.pkl",
    envmodules:
        ENVMODULES,
    threads: 4
    localrule: False
    shell:
        """
        export UV_CACHE_DIR="{{UV_TMPDIR}}"
        /data/antwerpen/210/vsc21003/bin/uv -n run --refresh --script {input.script} {wildcards.Ne} {output}
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
        export UV_CACHE_DIR="{{UV_TMPDIR}}"
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
        export UV_CACHE_DIR="{{UV_TMPDIR}}"
        ./{input.script} < {input.data}
        """
