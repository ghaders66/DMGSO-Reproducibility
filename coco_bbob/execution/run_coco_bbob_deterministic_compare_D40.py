# -*- coding: utf-8 -*-
"""
Independent COCO/BBOB D40 deterministic comparison runner.

This file does NOT modify the original runner.

Default use:
    python runners/run_coco_bbob_deterministic_compare_D40.py \
      --dims 40 \
      --budget-mult 1000 \
      --max-functions 24 \
      --runs 5 \
      --output-suffix D40
"""

from run_coco_bbob_deterministic_compare import *  # noqa: F403


def summarize_results(run_summary_path: Path, output_suffix: str = "") -> None:
    df = pd.read_csv(run_summary_path)

    suffix = f"_{output_suffix}" if output_suffix else ""

    function_summary = (
        df.groupby(
            ["suite", "algorithm", "function_id", "dimension"],
            dropna=False,
        )
        .agg(
            runs=("run_id", "count"),
            instances=("instance_id", "nunique"),
            error_mean=("error", "mean"),
            error_median=("error", "median"),
            error_std=("error", "std"),
            error_min=("error", "min"),
            error_max=("error", "max"),
            best_f_mean=("best_f", "mean"),
            best_f_median=("best_f", "median"),
            fe_used_mean=("fe_used", "mean"),
            fe_used_median=("fe_used", "median"),
            wall_time_mean=("wall_time_sec", "mean"),
            wall_time_median=("wall_time_sec", "median"),
            n_accept_mean=("n_accept", "mean"),
            n_reloc_mean=("n_reloc", "mean"),
        )
        .reset_index()
    )

    dimension_summary = (
        df.groupby(
            ["suite", "algorithm", "dimension"],
            dropna=False,
        )
        .agg(
            runs=("run_id", "count"),
            functions=("function_id", "nunique"),
            instances=("instance_id", "nunique"),
            error_mean=("error", "mean"),
            error_median=("error", "median"),
            error_std=("error", "std"),
            error_min=("error", "min"),
            error_max=("error", "max"),
            best_f_mean=("best_f", "mean"),
            best_f_median=("best_f", "median"),
            fe_used_mean=("fe_used", "mean"),
            fe_used_median=("fe_used", "median"),
            wall_time_mean=("wall_time_sec", "mean"),
            wall_time_median=("wall_time_sec", "median"),
            n_accept_mean=("n_accept", "mean"),
            n_reloc_mean=("n_reloc", "mean"),
        )
        .reset_index()
    )

    function_out = OUT_TABLES / f"coco_bbob_deterministic_function_summary{suffix}.csv"
    dimension_out = OUT_TABLES / f"coco_bbob_deterministic_dimension_summary{suffix}.csv"

    function_summary.to_csv(function_out, index=False)
    dimension_summary.to_csv(dimension_out, index=False)

    print("[OK] saved:", function_out)
    print("[OK] saved:", dimension_out)


def run_coco_bbob_d40(
    dims,
    budget_mult,
    max_functions,
    runs,
    algorithms,
    trace,
    trace_every,
    overwrite,
    output_suffix,
):
    suffix = f"_{output_suffix}" if output_suffix else ""

    run_summary_path = (
        OUT_TABLES /
        f"coco_bbob_deterministic_run_summary{suffix}.csv"
    )

    if overwrite and run_summary_path.exists():
        run_summary_path.unlink()

    suite = cocoex.Suite("bbob", "", "")

    for problem in suite:
        meta = get_problem_metadata(problem)

        fid = int(meta["function_id"])
        dim = int(meta["dimension"])

        if dim not in dims:
            continue

        if max_functions is not None and fid > max_functions:
            continue

        budget = int(budget_mult * dim)

        for run_id in range(runs):
            for algorithm in algorithms:
                alg = algorithm.upper()

                objective_problem = suite.get_problem(problem.index)
                meta = get_problem_metadata(objective_problem)

                try:
                    result = run_algorithm(
                        problem=objective_problem,
                        algorithm=alg,
                        budget=budget,
                        run_id=run_id,
                        trace=trace,
                        trace_every=trace_every,
                    )

                    best_f = float(result["best_f"])
                    fe_used = int(result["fe_used"])
                    n_accept = result["n_accept"]
                    n_reloc = result["n_reloc"]
                    wall = float(result["wall_time_sec"])
                    status = "ok"

                    if trace and alg == "DMGSO" and result["trace_rows"]:
                        trace_path = (
                            OUT_TRACES
                            / f"trace_coco_bbob_DMGSO_f{fid:02d}_i{meta['instance_id']}_D{dim}_run{run_id}{suffix}.csv"
                        )
                        write_trace_csv(result["trace_rows"], trace_path)

                except Exception as exc:
                    best_f = float("inf")
                    fe_used = budget
                    n_accept = np.nan
                    n_reloc = np.nan
                    wall = np.nan
                    status = f"failed: {type(exc).__name__}: {exc}"

                f_opt = float(meta["f_opt"])
                error = float(best_f - f_opt)

                if not np.isfinite(error):
                    error = float("inf")

                row = {
                    "suite": "COCO-BBOB",
                    "algorithm": alg,
                    "function_id": fid,
                    "instance_id": meta["instance_id"],
                    "function_name": meta["function_name"],
                    "dimension": dim,
                    "run_id": run_id,
                    "FE_budget": budget,
                    "budget_mult": budget_mult,
                    "fe_used": fe_used,
                    "best_f": best_f,
                    "f_opt": f_opt,
                    "error": error,
                    "n_accept": n_accept,
                    "n_reloc": n_reloc,
                    "wall_time_sec": wall,
                    "status": status,
                }

                append_csv(row, run_summary_path)

                print(
                    f"[{status}] {alg:11s} | F{fid:02d} | "
                    f"I{meta['instance_id']} | D={dim:2d} | "
                    f"run={run_id + 1}/{runs} | "
                    f"budget={budget} | error={error:.3e} | FE={fe_used}"
                )

    summarize_results(run_summary_path, output_suffix)

    print("\nCOCO/BBOB D40 deterministic comparison finished.")
    print("Run summary:", run_summary_path)


def parse_args_d40():
    parser = argparse.ArgumentParser(
        description="Run independent D40 deterministic DFO comparison on COCO/BBOB."
    )

    parser.add_argument("--dims", nargs="+", type=int, default=[40])
    parser.add_argument("--budget-mult", type=int, default=1000)
    parser.add_argument("--max-functions", type=int, default=24)
    parser.add_argument("--runs", type=int, default=5)

    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["DMGSO", "NELDER-MEAD", "POWELL", "COBYLA", "COMPASS"],
    )

    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--trace-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-suffix", type=str, default="D40")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args_d40()

    run_coco_bbob_d40(
        dims=args.dims,
        budget_mult=args.budget_mult,
        max_functions=args.max_functions,
        runs=args.runs,
        algorithms=args.algorithms,
        trace=args.trace,
        trace_every=args.trace_every,
        overwrite=args.overwrite,
        output_suffix=args.output_suffix,
    )