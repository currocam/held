UV_TMPDIR = "/tmp/vsc21003/uv"


rule all:
    input:
        "results/tables/00-benchmark_ld_compute.csv",


rule benchmark_ld_compute:
    input:
        script="scripts/00-benchmark_ld_compute.py",
    output:
        "results/tables/00-benchmark_ld_compute.csv",
    shell:
        """
        export UV_CACHE_DIR="{{UV_TMPDIR}}"
        ./{input.script} > {output}
        """
