# DMGSO Sensing-Topology Analysis

This directory contains the audited computational materials supporting the
reviewer-driven sensing-topology experiment reported in the revised manuscript.

## Scope

The controlled analysis compares four deterministic sensing topologies:

- `HISTORICAL`
- `AXIS_ONLY`
- `FIXED8_UNIQUE`
- `DIM_ADAPTIVE`

The experiment covers BBOB functions F1--F24 at dimensions D=20 and D=40,
using 15 official instances, five deterministic configuration indices
(`run_id` 0--4), and a function-evaluation budget of 1000 x D.

The historical DMGSO results are reused from the frozen COCO/BBOB experiment
and are not rerun. The three controlled revision topologies are provided as a
separate frozen run-level dataset. The combined analysis contains 14,400
observations (3,600 per topology).

Statistical inference is performed separately at each dimension using the
BBOB function as the inferential unit (n=24). Each function-level value is the
median corrected error across 15 instances x 5 deterministic configurations.

## Directory contents

- `execution/` -- original audited execution script and repository-portable
  reproduction entry point.
- `data/frozen/` -- frozen run-level results for the three revision topologies.
- `statistics/` -- audited statistical-analysis scripts and frozen outputs.
- `audit/` -- frozen-data integrity and combined-design audit scripts/reports.
- `provenance/` -- SHA-256 integrity manifest.

Repository-portable scripts use package-local paths and write newly generated
results under `reproduction_output/`. This directory is intentionally excluded
from version control so that reproduction runs cannot overwrite or be confused
with the frozen evidence.

The original audited scripts are retained for computational provenance.
Portability changes are limited to repository-local path/import handling; the
scientific algorithm, experimental protocol, and frozen datasets are not
modified.

## Figures

No topology-specific figure is included because this controlled experiment
supports a textual result in the revised manuscript rather than a dedicated
figure. No additional visualization artifact was introduced solely for the
repository.

## Integrity

Frozen inputs and reported statistical outputs are protected by SHA-256
fingerprints. See `provenance/SHA256SUMS.txt`.
