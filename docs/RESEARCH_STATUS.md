# Research status and code integration boundaries

Status checked 2026-10-20. Real-domain status remains **not_calibrated / not_evaluated**.

## What is actually in the code

The source baseline is `1c4f4c98b92a3703aabc8c1cbdee7f71f3eafae1`. At the status check, [PR #15](https://github.com/YQGHL/roadmc/pull/15) is open and not merged. It carries the procedural distress replay, texture/sampling and native-data diagnostic source changes. A documentation PR based on that feature branch depends on #15; it must not be interpreted as already available on `master`.

Pending README text had described these source features without publishing the corresponding study guides and JSON examples. This update publishes those companions and dataset attribution; it changes no numerical algorithm or default parameter. One SUD provenance fix removes a runtime dependency on an unpublished test file. Fresh protocols must bind the new executing-source hashes; historical frozen protocols must not be edited. A regression test checks source hashing without a test directory.

## Research outside the production repository

Stage D, G-type feasibility, H1 irreversible material history, original-occupancy cavities and the subsequent A/B/C system adapter were developed in isolated research directories. They are not native production `SurfaceTile`/replay capabilities. Moving their snapshots into production requires a separate design, contracts, tests and scientific decision; this documentation update does not silently integrate them.

The controlled system matrix contains 198 configurations and 1,449,030 saved points, not independent real roads. Snapshot B can restore material under the constructed step path; history C preserves observed absence but can miss an interior DROP under coarse loading. An unwired path guard is not a demonstrated complete solution.

Smooth-boundary advantages do not extend to sharp boundaries. G-type did not establish broad superiority. These negative results remain part of the research record. Neither tests nor source publication establish real material physics, calibrated wear parameters or downstream transfer gains.

## Manuscripts and next work

The Chinese v3 manuscript is a latest candidate, not an accepted scientific version. PDF rendering and citation corrections are distinct from scientific acceptance. Remaining review includes state-rule priority, per-arm node-fraction denominators, numerical diagnostic versus certified error bounds, loading-window interpretation and claim scope. Formatting upgrades do not prove novelty.

The shortest next step is a bounded manuscript fact review and a decision on external comparison/use value. Do not rerun the full Fan audit, add training, integrate a guard, merge PRs or publish a paper merely because these materials exist. Dataset roles and licensing references are in [DATASETS.md](DATASETS.md).
