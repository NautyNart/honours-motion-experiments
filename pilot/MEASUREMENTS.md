# Measurement justification and operational definitions

This pilot measures kinematic ground-contact defects in single-person, level-ground
motion. These are diagnostic proxies, not certification of physical validity.
Definitions and sources were checked on 10 October 2026. No simulator is required.

## Representation established from this repository

`SFT/HumanML3D/utils/motion_process.py` recovers 22 SMPL-style joint positions from
unnormalised 263-D features. Channels are root angular velocity (1), horizontal root
velocity (2), root height (1), root-relative X/Z and absolute-height joint positions
(63), joint rotations (126), local joint velocities (66), and foot contacts (4).
The code's Y axis is vertical; X/Z span the floor. Root orientation integrates the
half-angle stored in the representation. `pilot.common.recover_xyz` reproduces this
recovery in NumPy; its output was compared with all five existing saved joint arrays.

HumanML3D uses 20 fps and 22 joints; mirroring changes both motion and caption [1].
Distances here are nominal metres in the uniform HumanML3D skeleton convention.
Verify the dataset provenance, scale and person height during calibration before
accepting metre thresholds. Normalised model inputs must be denormalised using the
checkpoint's own mean/std; dataset means must not replace those statistics.

The upstream preprocessing subtracts the sequence's minimum joint height and
initialises horizontal origin/orientation. For those canonical features the intended
floor is Y=0. The pilot retains that reference through corruption and editing:
**never re-floor an edited clip by subtracting its own minimum height**, which would
erase penetration or floating. An annotation can set another verified floor, but
before/after clips must share the same physical reference. The saved baseline has
no scene mesh, body mesh, foot sole geometry, forces or mass model. It cannot provide
surface collision depth, dynamic balance or general biomechanical validity.

Foot joints: left ankle 7, left toe 10; right ankle 8, right toe 11. All joint names
are in `pilot/common.py`. The new viewer projects X/Z ground coordinates directly,
colors anatomical left blue/right red, and keeps the floor fixed. An old GIF's
screen direction must not be treated as anatomical ground truth.

## Compact measurement specification

All event intervals use zero-based frames `[start,end)`; end is excluded. At 20 fps,
the provisional 0.15 s persistence rule means at least three consecutive samples.

| Defect / operational definition | Required inputs | Measurement and units | Provisional decision | Body part / interval | Limitations and confounds |
|---|---|---|---|---|---|
| Sliding: a supporting foot anchor moves horizontally during a reviewed planted interval | XYZ, fps, independently reviewed support and ankle/toe anchor | Horizontal step length × fps (m/s); mean supported speed and summed supported path (m) | Speed >0.075 m/s for ≥0.15 s; both adjacent frames must belong to the support interval | Selected left/right anchor; contiguous threshold crossings | Heel/toe rolling, deliberate skidding, incorrect support boundaries; path sum depends on number of anchors and support duration |
| Penetration proxy: a joint center lies below the fixed floor | XYZ, verified scale and floor | `max(floor-y,0)` (m); maximum, mean over joints/frames, fraction of frames above tolerance | Depth >0.01 m for ≥0.15 s | Each joint; ankle/toe events assigned to that foot; other joints retain their names | Joint centers are not skin or soles. A surface may penetrate while its joint remains above ground. Floor errors can dominate |
| Unintended floating: all foot anchors are above the floor while foot support is explicitly required | XYZ, verified floor/scale, required-support intervals, legitimate-flight exclusions | Lowest of ankles/toes above floor (m), affected duration (s) | Lowest foot >0.05 m for ≥0.15 s during required support | `both_feet`, contiguous required-support crossings | Assumes foot-supported actions; not handstands, crawling or object support. One raised swing foot is insufficient. Ankle/toe heights imperfectly represent soles |

Continuous summaries retain sub-threshold values. Event persistence determines the
diagnosis, so a short excursion can affect a numerical summary without producing an
event. Missing support yields null sliding/floating summaries and an unavailable
decision, never zero error. Unverified floor/scale allows exploratory penetration
numbers but prevents them from becoming actionable rule feedback.

## Published definitions versus this pilot's adaptations

**PhysDiff [2], Appendix A:** penetration/floating use the lowest body **mesh vertex**
relative to the ground with 5 mm geometry tolerance. Sliding averages horizontal
displacement of foot joints in contact across adjacent frames. We adopt adjacent-frame
contact conditioning, but replace mesh geometry with explicit joint proxies, report
speed and duration, and gate floating on required support. Our outputs are not
numerically interchangeable with its Penetrate/Float/Skate scores.

**OmniMoGen [3], Section 3.2.3 and Eq. 9:** its physical reward penalises squared
horizontal foot velocity under a contact indicator. Its contact score uses height
and speed thresholds of 0.05 m and 0.075 m/s. We use those values only as provisional
starting points for different, localised event rules. The 0.01 m joint penetration
tolerance and 0.15 s persistence rule are pilot choices, not values validated by that
paper. We do not reproduce its reward or combined contact score.

**UnderPressure [4]:** contact inference is itself a difficult problem; the paper
contrasts threshold heuristics/manual annotation with learned contact and force
estimation. This supports treating support provenance as a separate dependency.
We do not load its model or claim force-derived contact labels.

IRG's published future-work discussion motivates physical assessment beyond text
alignment [5]. This pilot supplies measurements externally and uses an explicit
assess/edit loop; it does not establish native physical understanding from tokens.

## Support annotation and calibration

1. Inspect raw and decoded clips from multiple camera angles. Use the requested
   action, foot height, leg/body motion and visible planted phase to mark support.
   Velocity may assist inspection but is not the sole criterion.
2. Annotate left/right foot and which anchor is planted. Toe-off and heel strike can
   involve different anchors; split intervals or mark ambiguous frames uncertain.
   Do not overlap annotations for the same anchor. Mark running/jumping flight
   explicitly and exclude it from required support. A raised swing foot is valid.
3. Review floor and scale. For this first pilot, exclude scenes requiring unknown
   objects, hand support or uneven ground. Document ambiguous cases rather than
   forcing a physical label.
4. The plan reserves 10 slots for calibration (three originals, their four variants,
   three generated candidates). Their source groups do not appear in the 40-slot
   comparison subset. Select original slots 00, 04 and 10 accordingly. Keep mirrors,
   alternate captions and segments of a recording in the same subset.
5. Defaults remain explicitly `provisional_not_calibrated`. Inspect severity plots
   and false alarms on reviewed calibration clips; document any chosen thresholds
   and their rationale in a new config before the comparison. Do not choose values
   by looking at which feedback condition wins. A predeclared sensitivity analysis
   can rerun measurement at 0.5×/1×/2× tolerances without choosing the best outcome.
6. Inspect again after VQ encode/decode. Corruption construction is an intended label,
   not proof that the reconstructed motion retains it. Compare `raw` and `initial`
   metrics and review both. After editing, annotate support timing again.

These checks may reveal that a measurement is unreliable. Report that finding;
the implementation does not assume the research question has already been answered.

## Primary sources

1. Chuan Guo et al., *Generating Diverse and Natural 3D Human Motions from Text*,
   CVPR 2022. [Official dataset and processing code](https://github.com/EricGuo5513/HumanML3D),
   [motion representation notebook](https://github.com/EricGuo5513/HumanML3D/blob/main/motion_representation.ipynb).
2. Ye Yuan et al., *PhysDiff: Physics-Guided Human Motion Diffusion Model*, ICCV 2023.
   [Paper, Appendix A](https://arxiv.org/html/2212.02500#A1).
3. Wendong Bu et al., *OmniMoGen: Unifying Human Motion Generation via Learning from
   Interleaved Text-Motion Instructions*, arXiv:2512.19159v1, 2025.
   [Paper](https://arxiv.org/html/2512.19159v1).
4. *UnderPressure: Deep Learning for Foot Contact Detection, Ground Reaction Force
   Estimation and Footskate Cleanup*, 2022.
   [Author paper](https://arxiv.org/abs/2208.04598).
5. *IRG-MotionLLM: Interleaving Motion Generation, Assessment and Refinement for
   Text-to-Motion Generation*, arXiv:2512.10730v2, Appendix G.
   [Paper](https://arxiv.org/html/2512.10730v2#A7),
   [official code](https://github.com/HumanMLLM/IRG-MotionLLM).
