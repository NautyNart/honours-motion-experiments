# Execution record — 10 October 2026

## Existing project preserved

The working repository is `honours-motion-experiments`. The previous Kaggle notebook,
baseline runner and `outputs/irg-baseline-results/` were retained. The runner already
had an uncommitted post-denormalisation FP32 fix; this work did not change that file.
Original baseline asset checksums are in
`outputs/physical-pilot/baseline_source_hashes.json`; verification confirms those
source files remain unchanged.

The saved baseline used IRG Stage-2, Gemma-2-2B-IT, seed 42, greedy decoding, FP16 LLM,
FP32 VQ, Torch 2.5.1+cu121 and a Tesla T4. Five unique initial clips were imported.
Four baseline first/final pairs are identical; the squat clip contains two motion
blocks and differs. Those observations do not establish physical correctness.

The local hardware query reports **RTX 2070 SUPER, 8192 MiB**, not RTX 6000 Ada.
No local HumanML3D motion dataset, Gemma weights, full Stage-2 checkpoint, VQ checkpoint
or pretrained normalisation assets were found in this repository. PyTorch is absent
from the new CPU environment. No model/dataset downloads, GPU inference, assessor
training or human approvals were performed during this implementation.

## Implemented and executed

- Reusable `pilot/` modules, JSON configurations, CLI preparation/measurement/
  review/corruption/reconstruction/generation/refinement/report/export commands.
- Populated 19-cell `notebooks/physical_pilot.ipynb`; all nine code cells parse.
  Kaggle runtime and asset-download cells have not been executed here.
- CPU environment `.venv-pilot`; runtime requirements pass `pip check`.
- A 50-slot manifest with **five available generated clips and 45 pending slots**.
  The missing allocation is 15 originals, 20 controlled variants, 10 generated
  candidates. No candidate was falsely designated acceptable or physically failed.
- Measurements for raw/initial decoded arrays, JSON review templates and offline HTML.
  Existing generated raw/initial arrays are identical; the viewer requests only
  the initial review while retaining both measurement records.
- **22 CPU tests pass**, including known foot displacement, penetration, legitimate
  flight, swing-foot exclusion, floor preservation, corruption consistency, stale
  review rejection, matched inputs, cached-decoder mismatch, resumption and approval
  gating. Test-only fake editors verify orchestration, not IRG capability.
- NumPy coordinate recovery agrees with existing saved baseline joints: maximum
  absolute difference over the five clips is approximately **3.58e-7** in coordinate
  units. This validates the local recovery implementation against these assets.
- Headless Chrome checks pass for the initial and pair review pages, including
  scrubbing and JSON download, with no JavaScript errors. Draft downloads remain
  unreviewed. Before/after panels use common bounds and camera controls.
- Ruff checks for undefined names/unused imports pass on the new Python code.

Machine-readable checks and test/browser logs are in
`outputs/physical-pilot/verification/`.

## Local comparison smoke run

Use **`local-final`** as the final implementation smoke report. Earlier local smoke
directories are preserved development records and must not be pooled as independent
experiments. The report command with a run ID isolates that run.

Two available comparison clips were used: backward steps and a jump. Three other
saved prompts occupy calibration slots in the predeclared plan.

| Condition | What actually happened |
|---|---|
| Unchanged | Two reference outputs saved; no assessment performed |
| Semantic | Two explicit missing-checkpoint errors; no pretrained assessor/editor call |
| Rule | Two `uncertain` decisions because floor/scale/support are unreviewed; unchanged outputs saved |
| Human-verified | Two blocked records awaiting your explicit feedback approval |

Thus four records completed CPU work, two were blocked and two recorded a dependency
error. **No physical correction was demonstrated. No diagnostic accuracy or primary
success rate can be estimated without reviewed labels/outcomes.** Recorded GPU-hours
for this implementation are zero.

The largest exploratory below-Y=0 joint offset among the five initial clips is about
11.7 mm in the waving clip. Its floor and scale are not yet verified, so this is a
measurement observation, not a confirmed penetration label. Sliding/floating remain
unknown until support annotation. Thresholds remain explicitly provisional.

Available artifacts:

- `outputs/physical-pilot/manifest.json`
- `outputs/physical-pilot/review_initial/index.html`
- `outputs/physical-pilot/review_initial/annotations_template.json`
- `outputs/physical-pilot/review_pairs/index.html` — local smoke pairs, all unchanged
- `outputs/physical-pilot/reports/local-final/report.md`
- `outputs/physical-pilot/reports/local-final/comparison_per_clip.csv`
- `outputs/physical-pilot/reports/local-final/raw_reconstruction.csv`
- `outputs/physical-pilot/reports/local-final/initial_penetration.svg`
- `outputs/physical-pilot/reports/local-final/paired_penetration.svg`

## Exact remaining dependencies and next actions

1. Open the initial review page. Review nominal scale/floor, support intervals,
   physical labels and proposed feedback. Save/import your JSON. The tool will not
   fill approvals or complete labels for you.
2. Run the new notebook on Kaggle with the updated repository and saved results.
   Reuse or download Gemma, the full Stage-2 `.pth`, `ckpt/vqvae.pth`, and pretrained
   VQ `mean.npy` / `std.npy`. Saved HF revisions are copied from the original baseline.
   The GPU loader/task adapter has not yet received a pretrained execution test.
3. Supply the source-index CSV and 15 original HumanML3D feature files to complete
   the original/corruption allocation. Review acceptable parents and support before
   corruption, then review actual decoded defects. Generate the ten missing candidates.
4. Run one matched refinement round in smoke mode under a **new run ID** after your
   initial annotations are ready. Review condition-hidden outcomes. Use two rounds
   only after the first works; rule/human round two require review of the new motion.
5. Decide thresholds using the calibration subset or retain the declared provisional
   settings. Freeze them before the full comparison and report coverage/uncertainty.

The export ZIP is a portable snapshot of this work directory, including reports,
review templates and explicit pending/error states. The Python implementation and
notebook live in the repository and should accompany it when moving to Kaggle.
