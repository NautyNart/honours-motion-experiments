# Physical-defect correction pilot

Inference, measurement, human review and matched editing experiments for IRG.
No assessor training and no simulator. Start with the existing five decoded clips;
the 50-slot manifest records missing inputs honestly. See [EXECUTION.md](EXECUTION.md)
for what was actually run and [MEASUREMENTS.md](MEASUREMENTS.md) for definitions/sources.

## Start here locally

Run from the repository root using Python 3.11. The existing baseline and its results
are read only. The commands below use `python` from an activated environment.
On this PC a CPU environment has already been created at `.venv-pilot`.

```powershell
# New setup only:
python -m venv .venv-pilot
.\.venv-pilot\Scripts\Activate.ps1
python -m pip install -r requirements-pilot.txt

python -m pilot prepare --baseline outputs/irg-baseline-results/five-prompts-stage2-seed42
python -m pilot measure
python -m pilot review-export --output outputs/physical-pilot/review_initial
python -m pilot report
```

Open `outputs/physical-pilot/review_initial/index.html` directly in a browser.
It works offline and embeds the motion arrays. Scrub frames, change camera angle,
edit each JSON record, then click **Save current page** to download reviewed JSON.
Set your reviewer ID and `reviewed:true`; set feedback approval separately. Use
`labels_complete:true` with an empty label list only after confirming no target defect.
An `initial` review is needed for editing; `raw` reviews support corruption construction.
For existing generated clips raw/initial are the same decoder output, not a fresh VQ
round trip. Their pre-tokenisation motion is unavailable.

```powershell
python -m pilot review-import path/to/reviewed_annotations.json
python -m pilot measure
```

Draft records are ignored; accepted records are versioned in `review_history/` and
bound to a motion hash. The tool never creates your approval. For support annotate
ankle/toe/both during planted intervals; see the viewer's examples and measurement note.

## Kaggle

Upload/open [physical_pilot.ipynb](../notebooks/physical_pilot.ipynb). It has 19 populated
cells and uses reusable modules. Attach this updated repository and baseline results.
Use smoke mode first. The notebook creates an isolated Python 3.11/CUDA Torch 2.5.1
environment, inspects the actual GPU and chooses FP16 or native BF16 automatically.
The VQ decoder remains FP32. Do not assume two GPUs form one memory pool: this pilot
uses only GPU 0. Existing T4 inference does not establish a minimum VRAM requirement.

Model assets required:

- Gemma `google/gemma-2-2b-it`, requiring account access and an enabled Kaggle
  `HF_TOKEN` secret. Never paste tokens into the notebook, config or Git.
- Full IRG HumanML3D Stage-2 checkpoint from
  [Lymann/IRG-MotionLLM-HumanML3D](https://huggingface.co/Lymann/IRG-MotionLLM-HumanML3D).
- Motion-Agent `ckpt/vqvae.pth` and pretrained `mean.npy` / `std.npy`; the notebook
  helper uses the same public archive identifiers as the working baseline.
- GPU dependencies: `requirements-baseline.txt` plus `requirements-pilot.txt`.
  The optional browser verification tool uses Playwright; it is not a runtime dependency.

Enable `DOWNLOAD_MODELS` only when needed. Model revisions and archive hashes are
recorded. The helper does not obtain HumanML3D or install a simulator. `/kaggle/input`
stays read only; scratch code/models are under `/tmp`; exportable results are under
`/kaggle/working`. Expect multiple large model files plus the download archives and
Python environment. A 30–40 GiB scratch allowance is a planning estimate, not a
measured requirement; check the notebook's free-space output and actual asset sizes.
Fifty 200-frame 263-D+XYZ clips use roughly 13 MiB per raw/decoded/output copy;
HTML views, cached variants and archives add overhead. HumanML3D storage is separate.

## Build the 50-clip cohort

| Origin | Planned count | Purpose |
|---|---:|---|
| Original HumanML3D | 15 | Acceptable candidates; deliberately include at least five valid jumps/runs/hops. Human review decides acceptability |
| Controlled corruption | 20 | Eight sliding, six penetration, six floating; slight emphasis on support localisation |
| Generated | 15 | Fifteen diverse prompts, including the existing five; review distinguishes actual failures from acceptable results |

There are 10 calibration slots and 40 comparison slots, grouped by source family.
The planned count is unique arrays, not 50 independent recordings: 20 variants share
parents. Reports keep source IDs for clustered summaries. Generated origin is not a
defect label. Do not claim 15 genuine failures unless review finds 15; report the
observed mix. If more failure candidates are needed, predeclare a new cohort/config
and preserve the unsuccessful/acceptable candidates rather than silently replacing them.

Obtain HumanML3D through the [official workflow](https://github.com/EricGuo5513/HumanML3D).
Use already processed, unnormalised `new_joint_vecs/*.npy`. Fill
`outputs/physical-pilot/source_index_template.csv` and save a separate CSV. Each row
needs its official clip ID, source recording family (from the original index),
official partition, caption, 20 fps and feature path. Remove unfilled rows. Prefer
held-out data, but record the actual partition. Never put related mirrors/segments
in different calibration/comparison groups. Original slots 00, 04, 10 are calibration.

```bash
python -m pilot prepare --source-index path/to/source_index.csv
python -m pilot reconstruct
python -m pilot review-export --output outputs/physical-pilot/review_initial
# Review/import raw originals, especially support intervals; then:
python -m pilot corrupt
python -m pilot reconstruct
python -m pilot measure
python -m pilot review-export --output outputs/physical-pilot/review_initial
# Review/import the new reconstructed initial clips before comparisons.
python -m pilot generate   # Fill missing generated candidates on GPU; review afterward.
```

Controlled defects use smooth global translations in a reviewed support interval.
This preserves skeleton pose/rotations and updates redundant position/velocity
features consistently. It is a deliberately simple corruption family, not a model
of all naturally occurring foot errors. Changed contact channels follow the upstream
encoding convention but are never evaluation support labels. Labels and body parts
must be checked after decoding; multiple defects can coexist. VQ truncation to a
multiple of four frames is recorded. New reconstruction invalidates old annotations.

## Matched conditions and reviews

`pilot/configs/smoke.json` uses two available comparison clips; `pilot.json` has no
clip limit. Both start with one round, one candidate, greedy decoding and seed 42.
Edit model paths in a copied JSON config if necessary; relative paths resolve from
the repository. CLI `--work` selects any writable results location.

```bash
python -m pilot --config pilot/configs/smoke.json refine --run-id smoke-r1
python -m pilot --config pilot/configs/smoke.json review-export --mode pairs --run-id smoke-r1 --output outputs/physical-pilot/pairs_smoke
python -m pilot --config pilot/configs/smoke.json review-import path/to/pair_review.json --mapping outputs/physical-pilot/private/pairs_smoke_mapping.json
python -m pilot --config pilot/configs/smoke.json report --run-id smoke-r1
```

- **Unchanged:** same decoded starting motion; no assessment or editor call.
- **Semantic:** IRG's existing `refine_instruction` template receives caption and
  motion tokens without physical measurements, followed by its instructed editor.
- **Rule:** verified geometry/support plus configured thresholds produce deterministic
  localised instructions. Missing evidence produces `uncertain`; no edit is made.
- **Human:** your explicit review supplies/approves diagnosis and instruction for the
  exact input hash. It is human-verified feedback, not guaranteed perfect ground truth.

All editing conditions use the same frozen checkpoint, template, precision, candidate
budget and decoding settings. Each starts from the same initial tokens; decoding those
tokens is checked against the saved features. The original Stage-2 self-interleaving
baseline remains available in its old notebook; this new comparison uses the upstream
modular task interface so external feedback can be controlled. Its efficacy is untested.

For two rounds, set `rounds:2` **before** starting a new run. The first invocation
produces round-one outputs. Human round two waits for review. Rule round two waits
for fresh support/geometry review. Semantic feedback can proceed immediately.
Export the pair page, review/approve feedback for human round-one outputs and review
support on rule round-one outputs. Save/import that review and
resume the same run; completed inputs, code, cohort and config are immutable. A changed
review for a completed case requires a new run ID. No automatic approval is implied.

Outputs include full feedback, model text, every motion round, measurements, hashes,
runtime, GPU peaks, and session GPU-hours (including model load time). Input masks are
never copied to a differently timed edited motion. Same-seed greedy reruns are not
independent samples. No pretrained semantic evaluator is added: none was available
locally, and semantic preservation requires your explicit before/after review.

## Reports, resumption and export

The primary outcome is target resolution without new defects or loss of requested
action. It is scored only for independently reviewed defective initial clips and
reviewed outcomes; otherwise it is unknown. Condition-hidden pair pages randomise
case order and omit condition/instruction labels. They retain before/after order;
visual unchangedness can reveal the reference condition, so blinding is partial.
Keep `private/` mapping files away from reviewers until import.

Reports contain per-clip changes, diagnostic precision/recall/false alarms where
labels permit, event localisation, CSV tables and editable SVG plots. The pilot is
exploratory. Source-cluster bootstrap intervals require at least five reviewed groups;
small-sample uncertainty remains substantial. Inspect origin and subset separately.
Detection metrics evaluate the rule detector against reviewed physical labels;
semantic free text is retained verbatim, not automatically assigned physical labels.
Human feedback is an intervention, not independent ground truth. Pair reviews show
the same initial target-defect set for every condition; target resolution means
resolving that complete set. `report` without a run ID produces an inventory, and
`report --run-id ...` uses that run's frozen configuration, never pools different runs.

```bash
python -m pilot bundle --output outputs/physical-pilot-export.zip
python -m unittest discover -s pilot/tests -v
```

Export refuses to overwrite an existing ZIP. Bundles contain relative paths and the
work directory's manifests, annotations, metrics, configs, motions and HTML; no models
or environment variables. Restore into an empty writable folder, then use `--work`.
Completed cases resume by fingerprint; blocked cases retry after their dependency is
resolved. Errors go to `errors.jsonl` and per-case records. Reports never turn an error,
no-edit decision or missing approval into a successful correction.
