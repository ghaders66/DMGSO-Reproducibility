import csv
import re
import shutil
from pathlib import Path

import cocoex


ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = (
    ROOT / "coco_bbob" / "processed" /
    "bbob_fopt_cocoex_2.8.2.csv"
)

DIMENSIONS = [2, 5, 10, 20, 40]
FUNCTIONS = list(range(1, 25))
INSTANCES = [
    1, 2, 3, 4, 5,
    71, 72, 73, 74, 75,
    76, 77, 78, 79, 80,
]

FOPT_RE = re.compile(
    r"Fopt\s*\(\s*([+-]?[0-9.eE+-]+)\s*\)"
)


def extract_one(fid, iid, dim):
    folder_name = (
        f"fopt_tmp_f{fid:03d}_i{iid:02d}_d{dim:02d}"
    )

    local_folder = Path("exdata") / folder_name

    if local_folder.exists():
        shutil.rmtree(local_folder)

    # COCO instance_indices refers to the position within the
    # suite instance list, not to the BBOB instance ID itself.
    instance_position = INSTANCES.index(iid) + 1

    suite = cocoex.Suite(
        "bbob",
        "",
        (
            f"dimensions:{dim} "
            f"function_indices:{fid} "
            f"instance_indices:{instance_position}"
        ),
    )

    if len(suite) != 1:
        raise RuntimeError(
            f"Expected exactly one problem for "
            f"F{fid} I{iid} D{dim}; got {len(suite)}"
        )

    problem = suite.get_problem(0)

    if (
        int(problem.id_function) != fid
        or int(problem.id_instance) != iid
        or int(problem.dimension) != dim
    ):
        raise RuntimeError(
            f"COCO metadata mismatch for requested "
            f"F{fid} I{iid} D{dim}: got "
            f"F{problem.id_function} "
            f"I{problem.id_instance} "
            f"D{problem.dimension}"
        )

    expected_id = (
        f"bbob_f{fid:03d}_i{iid:02d}_d{dim:02d}"
    )

    if problem.id != expected_id:
        raise RuntimeError(
            f"Problem ID mismatch: expected {expected_id}, "
            f"got {problem.id}"
        )

    observer = cocoex.Observer(
        "bbob",
        (
            f"result_folder: {folder_name} "
            f"algorithm_name: fopt_extract"
        ),
    )

    problem.observe_with(observer)

    # One evaluation makes the official BBOB logger
    # write the header containing Fopt.
    x = problem.initial_solution
    _ = problem(x)

    dat_files = list(local_folder.rglob("*.dat"))

    if len(dat_files) != 1:
        raise RuntimeError(
            f"Expected one .dat file for {expected_id}; "
            f"found {len(dat_files)}"
        )

    lines = dat_files[0].read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    if not lines:
        raise RuntimeError(
            f"Empty .dat file for {expected_id}"
        )

    match = FOPT_RE.search(lines[0])

    if not match:
        raise RuntimeError(
            f"Could not parse Fopt for {expected_id}: "
            f"{lines[0]}"
        )

    f_opt = float(match.group(1))

    shutil.rmtree(local_folder)

    return {
        "suite": "bbob",
        "cocoex_version": getattr(
            cocoex, "__version__", "unknown"
        ),
        "function_id": fid,
        "instance_id": iid,
        "dimension": dim,
        "problem_id": expected_id,
        "f_opt": f_opt,
    }


def main():
    OUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    expected = (
        len(FUNCTIONS)
        * len(INSTANCES)
        * len(DIMENSIONS)
    )

    rows = []

    print("=" * 72)
    print("COCO/BBOB OFFICIAL FOPT EXTRACTION")
    print("=" * 72)
    print(
        "cocoex version :",
        getattr(cocoex, "__version__", "unknown"),
    )
    print("Expected rows  :", expected)
    print()

    count = 0

    for dim in DIMENSIONS:
        for fid in FUNCTIONS:
            for iid in INSTANCES:

                row = extract_one(
                    fid,
                    iid,
                    dim,
                )

                rows.append(row)
                count += 1

                if (
                    count % 100 == 0
                    or count == expected
                ):
                    print(
                        f"[{count:4d}/{expected}] "
                        f"last={row['problem_id']} "
                        f"f_opt={row['f_opt']:.12g}"
                    )

    if len(rows) != expected:
        raise RuntimeError(
            f"Extraction incomplete: "
            f"{len(rows)} != {expected}"
        )

    keys = [
        (
            r["function_id"],
            r["instance_id"],
            r["dimension"],
        )
        for r in rows
    ]

    if len(keys) != len(set(keys)):
        raise RuntimeError(
            "Duplicate problem keys detected."
        )

    with OUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "suite",
                "cocoex_version",
                "function_id",
                "instance_id",
                "dimension",
                "problem_id",
                "f_opt",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Extraction completed successfully.")
    print("Rows written    :", len(rows))
    print("Output          :", OUT_CSV)
    print("=" * 72)


if __name__ == "__main__":
    main()
