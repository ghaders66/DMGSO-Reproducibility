# DMGSO Ablation Studies

This directory contains the reproducibility materials for the DMGSO
ablation experiments reported in the Information Sciences revision.

The ablation study evaluates five deterministic architectural variants:

- V0: directional sensing only
- V1: V0 + adaptive step regulation
- V2: V1 + long-range sensing
- V3: V1 + memory guidance
- V4: full DMGSO architecture with adaptive step regulation,
  long-range sensing, memory guidance, and relocation

Two benchmark suites are provided independently:

- `cec2014/` — CEC2014 ablation at D=30
- `coco_bbob/` — COCO/BBOB ablation at D=2, 5, 10, 20, and 40

Both experiments use the same frozen DMGSO core implementation. The
included copies of `dmgso_core_revision.py` are byte-identical.

Historical exploratory and superseded ablation branches are intentionally
excluded from this reproducibility package. The files included here
correspond to the revision-stage experiments and the final figures and
summaries used for the revised manuscript.

Each benchmark directory contains the executable code, run-level data,
derived summaries, final publication figures, and provenance information.
