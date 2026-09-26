# Reproducibility Guide

This guide describes how to verify and reproduce the computational evidence
associated with the revised DMGSO manuscript.

The repository separates historical execution provenance, frozen analysis
evidence, and repository-portable reproduction support. This distinction is
intentional: reproduction should not overwrite or silently alter the frozen
evidence used for the manuscript.

## 1. Obtain the repository

Clone the repository and enter its root directory:

```bash
git clone https://github.com/ghaders66/DMGSO-Reproducibility.git
cd DMGSO-Reproducibility
```

For archival reproduction, a versioned release should be used once the final
archival release and persistent identifier are available.

## 2. Create the Python environment

The audited revision environment uses Python 3.11.15.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r environment/requirements.txt
```

Exact direct dependency versions are pinned in `environment/requirements.txt`.
The recorded operating-system and hardware information is provided in
`environment/environment_info.txt`.

## 3. Verify frozen evidence

Before analysis, verify the SHA-256 manifests. From the repository root:

```bash
(cd cec2014 && sha256sum -c provenance/SHA256SUMS.txt)
(cd coco_bbob && sha256sum -c provenance/SHA256SUMS.txt)
(cd ablation && sha256sum -c provenance/SHA256SUMS.txt)
(cd sensing_topology && sha256sum -c provenance/SHA256SUMS.txt)
(cd environment && sha256sum -c provenance/SHA256SUMS.txt)
```

All entries should report `OK`. A checksum failure means that the affected
file no longer matches the audited packaged artifact and should not be treated
as the frozen manuscript evidence.

The benchmark directories preserve the execution code, frozen datasets,
statistical outputs, figures where applicable, and their provenance records.

## 4. Frozen evidence versus new reproduction output

Files under the packaged frozen-data and reported-output directories represent
the audited revision evidence. They should be treated as read-only reference
artifacts.

New reproduction results should be written separately rather than replacing
the frozen files. The sensing-topology portable workflow enforces this by
writing generated results under `sensing_topology/reproduction_output/`.
That directory is excluded from version control.

## 5. Repository-portable sensing-topology verification

The sensing-topology workflow provides repository-local portable entry points.
The statistical analysis can be regenerated from the frozen data with:

```bash
python sensing_topology/statistics/analyze_sensing_topology_reproducible.py
```

The combined frozen-data audit can then be run with:

```bash
python sensing_topology/audit/audit_sensing_topology_reproducible.py
```

These portable scripts write to `sensing_topology/reproduction_output/` and
do not overwrite the frozen evidence.

In the package audit, regeneration of the six topology statistical outputs
produced byte-for-byte identical files to the frozen statistical outputs.
The combined four-topology audit also passed all design, pairing, FE-budget,
status, finite-error, and official-optimum consistency checks.

The full controlled topology experiment can be rerun, at substantially greater
computational cost, with:

```bash
python sensing_topology/execution/run_sensing_topology_reproducible.py
```

## 6. Historical benchmark execution code

The CEC2014 and COCO/BBOB benchmark directories retain the audited historical
execution scripts associated with the reported experiments. These files are
preserved for provenance rather than silently rewritten for presentation.

Some historical entry points may retain workstation-specific execution paths
or assumptions from the original experiment. Their inclusion establishes the
exact executed code and protocol; it should not be interpreted as a claim that
every historical runner is a repository-portable one-command entry point.

The frozen run-level datasets permit independent inspection and downstream
statistical verification without requiring every expensive optimization run
to be repeated.

## 7. Analysis and figure-generation materials

CEC2014 and COCO/BBOB external-comparison analysis scripts are located under
the corresponding `statistics/` directories. Publication plotting scripts
and frozen figure outputs are retained under the corresponding `figures/`
directories.

Ablation execution, summary, and plotting materials are retained under
`ablation/cec2014/` and `ablation/coco_bbob/`.

No sensing-topology figure is supplied because that controlled experiment
supports a textual result in the revised manuscript rather than a dedicated
publication figure.

## 8. Protocol and algorithm consistency

Exact benchmark settings, FE budgets, run/configuration semantics, DMGSO
parameters, statistical units, and software versions are documented in
`docs/EXPERIMENTAL_PROTOCOL.md`.

The mapping between manuscript Algorithm 1 and the audited implementation is
documented in `docs/ALGORITHM_TO_CODE.md`.

Together with the SHA-256 manifests and frozen datasets, these documents
provide the implementation, protocol, provenance, and analysis information
needed for independent verification of the reported computational results.

## 9. Recommended verification order

1. Create the pinned Python environment.
2. Verify all SHA-256 manifests.
3. Inspect `docs/EXPERIMENTAL_PROTOCOL.md` for the exact experiment design.
4. Inspect `docs/ALGORITHM_TO_CODE.md` for pseudocode-to-code consistency.
5. Reproduce downstream analyses from frozen data.
6. Rerun expensive optimization experiments only when full re-execution is
   required.
