# CtrlHGen Worklogs

This directory is the audit trail for the reproduction effort.

- `requirement-gap.md`: baseline gap analysis and phased acceptance criteria.
- `phase-a-2026-08-08.md`: Phase A implementation and validation log.
- `phase-b-2026-08-09.md`: WN18RR tiny live-data end-to-end smoke log.
- `phase-c-preflight-2026-08-09.md`: local validation/checkpoint policy work
  completed before the Phase C/D live runs.
- `phase-c-2026-08-10.md`: four Phase C runs, training-scale ablation, frozen
  metrics, and the Phase D parent bake-off.
- `phase-d-2026-08-11.md`: formal epoch-45-parent GRPO run, interruption/resume,
  frozen-test metrics, and artifact audit.
- `phase-d-pilot-2026-08-11.md`: fresh RL-only data and terminal `SEP` repaired
  pilot.
- `phase-d-repaired-full-2026-08-11.md`: 13,000-step repaired GRPO confirmation.
- `phase-d-repaired-frozen-test-2026-08-12.md`: the single repaired terminal
  frozen-test and paired audit.
- `reproduction-report-2026-08-12.md`: comprehensive final report with figures.
- `report-data/`: compact DSW records and SHA256 index used to rebuild figures.

Repository worklogs record the implementation SHA that was actually validated.
The delivery commit containing the final log is reported in the task handoff because
a tracked file cannot contain the hash of the commit that contains itself.
