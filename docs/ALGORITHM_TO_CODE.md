# Algorithm-to-Code Consistency

This document maps Algorithm 1 in the revised manuscript to the audited DMGSO
implementation distributed with this repository.

The two benchmark packages contain byte-identical copies of the DMGSO core:

- `cec2014/execution/dmgso_core_revision.py`
- `coco_bbob/execution/dmgso_core.py`

Both files have SHA-256:

`cadf35f0ae664619527707caf2deea4a163e79eb970cf273e53f5490f9cd9191`

The mapping below refers to `cec2014/execution/dmgso_core_revision.py`.

## Core deterministic components

| Manuscript operation | Implementation | Source location |
| --- | --- | --- |
| Reflective bound handling | `reflect_bounds()` | lines 55--76 |
| Halton sequence generation | `halton_vector()` | lines 119--129 |
| Deterministic initialization | `deterministic_init()` | lines 132--135 |
| Deterministic relocation | `deterministic_relocate()` | lines 138--147 |
| Deterministic direction set | `build_directions()` | lines 155--180 |
| Favorable memory | `TopKMemory` | lines 194--209 |
| Unfavorable memory | `FIFOBuffer` | lines 211--220 |
| Memory-guidance vector | `memory_guidance_vector()` | lines 334--359 |
| Adaptive fusion weights | `directional_fusion_weights()` | lines 362--391 |
| Main optimization loop | `dmgso_optimize()` | begins at line 449 |

The direction set contains the 2D signed coordinate directions plus a fixed
number of deterministic diagonal sign-pattern directions. All directions are
normalized before use.

## Algorithm 1 to implementation mapping

| Algorithm 1 step | Audited implementation | Source location |
| --- | --- | --- |
| Construct deterministic directions | `build_directions(D, k_diag)` | line 481 |
| Initialize step size `s` | `s = cfg.s0_frac * scale` | line 485 |
| Initialize local radius `delta` | `delta = min(cfg.delta0_frac * scale, ...)` | line 494 |
| Initialize long-range radius `Delta` | `Delta = min(cfg.Delta0_frac * scale, ...)` | line 495 |
| Initialize favorable/unfavorable memories | `TopKMemory`, `FIFOBuffer` | lines 497--498 |
| Insert initial state into favorable memory | `favorable_memory.add(x, f_cur)` | line 506 |
| Initialize stagnation state | `stagnation = 0` | line 508 |
| Local directional sensing | local probing and construction of `vL` | lines 547--570 |
| Long-range directional sensing | long-range probing and construction of `vG` | lines 572--595 |
| Memory-guided sensing | construction of `vM` | lines 597--602 |
| Adaptive directional fusion | `directional_fusion_weights(...)` | lines 604--639 |
| Degenerate-direction fallback | box-center direction, then indexed direction | lines 641--660 |
| Reflective candidate generation | `reflect_bounds(x + s_eff * v_hat, ...)` | line 669 |
| Strict improvement acceptance | acceptance branch following candidate evaluation | lines 676--701 |
| Favorable-memory update | accepted state added to favorable memory | line 687 |
| Step expansion after acceptance | `s = min(s * cfg.c_plus, s_max)` | line 696 |
| Rejection/stagnation update | stagnation increment and unfavorable memory | lines 703--708 |
| Step contraction after rejection | `s = max(s * cfg.c_minus, s_min)` | line 711 |
| Stagnation-triggered relocation | deterministic relocation block | lines 714--738 |
| Reset after relocation | reset step size and stagnation | lines 735--736 |

## Notation correspondence

The manuscript and implementation use equivalent quantities with two minor
naming differences:

- manuscript long-range vector `v_O` corresponds to implementation `vG`;
- manuscript long-range fusion weight `beta` corresponds to implementation
  long-range weight `wG`.

The local and memory vectors correspond directly to `vL` and `vM`.
The local, long-range, and memory evidence strengths used by the manuscript
correspond to the vector magnitudes evaluated by the fusion logic before the
deterministic weights are assigned.

## Variant semantics

The implementation exposes the ablation structure through `_variant_flags()`:

- V0: directional baseline
- V1: V0 with adaptive step size
- V2: V1 with long-range sensing
- V3: V1 with memory guidance
- V4 / DMGSO: adaptive step, long-range sensing, memory, and relocation

V2 and V3 are parallel extensions of V1. They must therefore not be read as
a sequential V2-to-V3 single-component ablation.

## Optional refinement mode

The source also contains an experimental `DMGSO-REFINE` mode. This mode is
not part of manuscript Algorithm 1. The configuration explicitly sets
`refine_enable = False` for the main implementation. Consequently, the
refinement branch does not alter the reported main DMGSO protocol unless the
separate refinement mode is explicitly selected and enabled.

## Consistency statement

The audit found no structural inconsistency between manuscript Algorithm 1
and the main DMGSO implementation. Differences are limited to implementation
naming and low-level deterministic indexing details, including the explicit
Halton index used for relocation. These implementation details do not change
the algorithmic sequence described by Algorithm 1.
