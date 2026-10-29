# Design

Details behind the [README](../README.md): split roles, the axis reference, the
round loop rules, the planner, and the outputs. The README keeps only what is
needed to install, configure, run and read a run.

## Split roles

Datasets use the Roboflow layout (`train/`, `valid/`, `test/`).

| split | role |
|---|---|
| `train` (80% `fit_train` / 20% `fit_dev`) | builds all materials — prompts from all of `train`, tail fine-tune / rescorer / crop classifier trained on `fit_train` (a subset of it above `subsample.max_images` — below); `fit_dev` scores candidates |
| `valid` (holdout) | per-class thresholds τ_c, adoption decision, saturation |
| `test` | never read by the loop — measures the final champion once |

Without `valid/`, `test/` becomes the holdout and no unopened split remains;
the run records this as `holdout_split` in `run_summary.json`.

### fit_train subset — training time does not follow the dataset size

The three trainings (tail fine-tune, rescorer discriminator, crop classifier)
read fit_train as is, so their time grows with its size. When fit_train is
larger than `subsample.max_images` (500), only that many images are used for
training (`null` turns it off — a cap of 0 is refused by config, since it reads as
"0 images"; `fitting.subsample`, `fit_train_subset.json` in the run directory).

- **Picked in the order that covers the appearances, not at random** (greedy
  set cover). The DINOv3 embeddings of GT crops are clustered per class with
  k-center (centers = objects × `subsample.center_ratio`, 0.1), and images
  holding the most not-yet-covered centers go first. Once all are covered,
  another pass runs over the remaining images. Before that, one image per
  class goes in first (classes with fewer centers first) — the cover score does
  not tell classes apart, so if the first pass does not finish within the cap,
  a large class could take every slot. In a random sample the large clusters
  take the slots.
- Background images without GT keep their original share — they are the FP
  material the rescorer and classifier learn from. Even when background
  dominates, the GT images keep a share (one per class).
- **Only the training material shrinks.** Mining (all of train), verdicts
  (fit_dev) and adoption (holdout) are unchanged — verdict noise must not
  grow. The subset is picked within fit_train only, so material purity is
  unaffected.
- Deterministic (sorting, argmax, hashes); resume reuses the recorded subset
  (key = fit_train fingerprint, settings and embedding model; a record that no
  longer matches fit_train picks again). The training reuse keys hold the
  fit_train fingerprint, so a changed subset also breaks reuse.
- On datasets with many objects per image, 500 images can still mean many
  boxes — the cap bounds images, not objects.
- **The rescorer also caps its training rows** (`rescorer.max_train_rows`,
  100,000; `null` for no cap). A row is one detection candidate, so many detections per image can
  exceed it even after the subset. TPs are rare and the core of the ranking
  signal, so they are kept first (if TPs alone exceed half the cap they are
  thinned in class proportion, but slots the FPs cannot fill go to the TPs —
  rows are always at most the cap). FPs get a share per class proportional to
  the class's FP count, split evenly across score quartile bins — easy
  low-score FPs shrink a lot, FPs near the threshold mostly stay. Kept rows
  carry inverse-probability weights (stratum's original count ÷ kept) so the
  GBDT and the isotonic calibration see the original distribution (TP rate,
  score distribution). Under the cap it is the same uniform fit as before, and
  fit_dev rows (α grid, non-degradation check) are not thinned.

Every round, apart from the axes:

| step | split |
|---|---|
| diagnosis (planner input) | all of train inferred with the champion and split by cause (with samples); the holdout as aggregates only. Thresholds are the champion's holdout thresholds — planner input only, never fed into materials |
| screen | `fit_dev` — fixes the candidate's config, per-class thresholds and modality, and sends the top `screen.top_k` ordinary candidates to holdout (lora and classifier arms go to holdout on their own margin, outside `top_k`) |
| adoption, saturation | holdout (`valid`, or `test` when there is no `valid`) — compared with the champion at holdout thresholds; adopted only if the paired-bootstrap lower bound is ≥ `gate.margin`, Δ > 0 and Δ ≥ `gate.min_delta` (0.005) |
| final measurement | `test` (when `valid` was the holdout) — the settled champion, once, outside the loop (`scripts/predict.py --split test`) |

## Axis reference

A setting combines the 8 axes below. Every axis has an "off" value (the first
line) that equals the baseline behavior exactly.

| axis | values |
|---|---|
| `alias` — Prompt alias | `false`: class name only (manual aliases in `mine.text_aliases` are always included)<br>`vlm`: alias proposed by a multimodal LLM that looks at the crops |
| `exemplar` — Visual exemplar | `none`: text prompts only<br>`base`: GT crops as image prompts (by default chosen for diversity by embedding k-center — `mine.exemplar_selection`)<br>`fn_arm`: `base` + crops of GT objects missed when train is inferred with the `base` exemplars<br>counts `exemplar_base_k` ∈ {1, 2, 4, 8} (default 4), `exemplar_fn_k` ∈ {1, 2, 4} (default 2) |
| `text_negatives` — Negative text | `false`: no suppression phrases<br>`fp_crop`: phrases from false-positive crops of the mining pass — the mining pass infers train with the candidate's exemplars only (a GT k-center selection if it has none), without tiling or negatives<br>`gt_crop`: phrases from other classes' GT crops (suppress confusion — not valid on single-class datasets)<br>`background`: phrases from the same mining pass's false positives that overlap no GT of any class (suppress background)<br>count `text_negative_k` ∈ {1, 2, 4} (default 1) |
| `visual_negatives` — Negative visual | `false`: no suppression crops<br>`fp_crop` · `gt_crop` · `background`: crops from the same sources as `text_negatives` — mining-pass false positives, other classes' GT (not valid on single-class datasets; needs no mining pass), or false positives overlapping no GT — as negative image prompts (needs `exemplar`), emulated as IoU suppression. The suppression IoU threshold is chosen on fit_dev by screen, one config shared across classes, among a config without negatives and `search.neg_eta_grid` (0.25, 0.35, 0.5, 0.65, 0.75 — lower suppresses more)<br>count `visual_negative_k` ∈ {1, 2, 4} (default 1) |
| `rescorer` — Score calibration | `false`: detection scores as-is<br>`true`: a discriminator recalibrates scores. The per-class mixing weight α_c is chosen on fit_dev from `rescorer.alphas` (0, 0.1, 0.25, 0.5, 1, 2, 3, 4), and is non-zero only when the gain exceeds `rescorer.margin` (α_c = 0 is the original score) |
| `crop_classifier` — Re-classification | `false`: detected classes as-is<br>`true`: a SigLIP2+LoRA classifier gives class probabilities per detection box, blended with λ ∈ `crop_classifier.lambdas` (0, 0.25, 0.5, 0.75, 1): class = foreground argmax c*, score = s^(1−λ)·p(c*)^λ. λ = 0 is relabel-only (scores unchanged); a box whose argmax is background is not dropped — its score goes down and the per-class thresholds decide. Trained once per run (`<run>/classifier_cache/`, `crop_classifier.reuse`); each proposal re-scores only the λ grid on the current champion |
| `lora` — Tail fine-tune | `false`: base weights<br>`true`: fine-tunes the YOLOE tail on fit_train and interpolates with the base over `lora.lambdas` (0, 0.25, 0.5, 0.75, 1) — λ = 0 is the base. Not searched when fit_train has fewer than `lora.min_images` (20) images. The key keeps the name `lora` for history compatibility. On disk a candidate keeps only one small descriptor JSON per λ > 0 arm; the tuned tail subset (`tuned_tail.pt`) lives in the run's training cache `lora_cache/<digest>/` — arms are interpolated in memory, λ = 0 is the base itself, and the trainer's full-model checkpoints are deleted after training. Training runs once per run: its inputs (fit_train and ontology content, base weights path, `run_seed`, `lora.epochs`·`batch_size`·`lr`) do not depend on the champion, so later lora candidates (same round, carried, later rounds, after resume) rebuild the arms from the cached subset and re-score them, recording `delta_reused` ({tail: {round, id}}). An entry counts only once its `key.json` is written after training; one without it is retrained. The crop classifier is not reused — its training crops are mined from the current setting's FPs |
| `tiling` — Tiling | `false`: full image only<br>g ∈ {2, 3, 4, 5, 6}: a config that also infers the image as g×g tiles and merges (by default with SAHI GreedyNMM: only detections scoring ≥ `search.tiling_merge_min_score` (0.3) are merged, and those below it are kept minus the ones overlapping a merged result — `search.tiling_merge` also offers `nmm` and `nms`). Zoom (g−1)(1−overlap)+1 (at overlap 0.5, g=2 is 1.5×, g=3 is 2×), 1+g² forwards per image per pass. The planner picks the first grid from the missed_small sizes and the TP sizes the champion already detects (s*). If grid g went unused under the current champion, grids ≥ g are rejected as `tiling_blocked` until the champion changes (raising is only recommended, not enforced). Exception: if g lost to no tiling on fit_dev only within noise (one-sided paired-bootstrap upper bound `tiling_f.gap_ub` ≥ 0) and fell short on zoom (for the median m of the class with the most missed_small, m·zoom(g) < s* ≤ m·zoom(g+1), s* = the `search.tiling_tp_quantile` quantile of that class's champion TP sizes — m and s* are both train sizes counted at the champion's fit_dev screen thresholds, so holdout plays no part; with fewer than `search.tiling_tp_min` TPs there is no verdict), g+1 is opened — once, until a g+1 reaches screen (several g+1 candidates in the same proposal all pass). Grids ≥ `screen.prescreen_min_grid` (2) are first screened on a fit_dev subset |

An axis that is on may still go unused by the setting — screen picks whichever
is better on fit_dev (exemplar against text prompts per class, tiling against
the untiled config).

Axes not to search on a dataset go in `search.disabled_axes` by the axis keys
in the table (e.g. `[lora, tiling]`). The count keys (`exemplar_base_k`, `exemplar_fn_k`,
`text_negative_k`, `visual_negative_k`) can be disabled on their own too (the
count stays at its default or the champion's value), and
`search.disabled_values` bans single values (e.g. `{text_negatives: [gt_crop]}`).
An unknown key or value fails at the start of the run. Disabled axes and values
are left out of the planner's value list, and candidates that turn them on or
change them away from the champion's value are rejected — candidates that turn
them off or inherit the champion's value are kept. Visual negatives sit on top of
exemplars, so with `exemplar` disabled and no exemplar in the champion,
`visual_negatives` is treated as disabled too (`planner.axes.effective_disabled`,
computed per round from the champion for every candidate source). A disabled
count key is left out of the board's propose cell as well.

### Split use per axis

Every axis follows the same frame — materials and training come from train,
selection from fit_dev, adoption from the holdout gate. Nothing derived from
holdout (thresholds included) feeds materials (`test_materialize_purity` and
`test_split_roles` pin this).

- **train** is all of `fit_train` + `fit_dev`. Prompt materials (alias,
  exemplar, negatives) are mined from all of train; axes that train a model
  (rescorer, lora, crop_classifier) use only `fit_train`, so what they learn is
  measured fairly on `fit_dev`.
- **Mining pass**: train inferred with the candidate's exemplars (a GT k-center
  selection if it has none), without negatives or tiling, at the floor
  threshold. Missed GT (fn_arm) and false positives (negatives) come from it.
- Axes whose value equals the champion's are not re-mined; their materials are
  inherited from the champion.

| axis | materials / training | selection (`fit_dev`) | adoption (holdout) |
|---|---|---|---|
| `alias` | train: `mine.caption_crops_per_class` (6) GT crops per class whose short side is at least `mine.caption_min_side` px (32) — apart from the served exemplars, picked by the `mine.exemplar_selection` rule — are captioned by the VLM, and the LLM picks one alias per class. A class the captioner gave no phrase, or with no large-enough crop, gets a reason in the mining report, and the captioner warns when crops get no caption | carried in the text pass's prompts, so it counts in the per-class modality choice below | holdout gate |
| `exemplar` | train: `base` takes `exemplar_base_k` GT crops by k-center; `fn_arm` adds `exemplar_fn_k` crops of GT missed by the mining pass | one config shared by all classes from the semantic pass's config grid (max F sum), then per class a choice against the text pass of the same config | 〃 |
| `text_negatives` | train: `fp_crop` (mining-pass false positives), `gt_crop` (other classes' GT) or `background` (false positives overlapping no GT) crops are captioned by the VLM and refined by the LLM — dropping phrases that name the target itself — to at most `text_negative_k` phrases per class | carried in the text pass's prompts, so it counts in the modality choice. For a candidate without exemplars, the text pass sweeps the suppression IoU over `search.neg_eta_grid`, one config shared by all classes; with exemplars it follows the semantic config | 〃 |
| `visual_negatives` | train: `visual_negative_k` crops per class from the value's source (`fp_crop` mining-pass false positives, `gt_crop` other classes' GT, `background` false positives overlapping no GT) | from the semantic grid, one config shared by all classes among the one without negatives and the suppression IoU grid (`search.neg_eta_grid`) | 〃 |
| `tiling` | none — an inference setting | tiled and untiled configs are chosen from the same grid, shared by all classes (a text-only candidate runs the text pass per tiling variant). If untiled wins, `axis_unused`. Grids ≥ `screen.prescreen_min_grid` are first screened on a `fit_dev` subset | 〃 |
| `rescorer` | `fit_train`: discriminator training (area statistics over all of train, row cap `rescorer.max_train_rows`) | per-class α grid — identity unless the gain exceeds `rescorer.margin` | 〃 |
| `lora` | `fit_train`: YOLOE tail fine-tune (no validation split during training) | λ grid — the arm's ΔF must exceed `lora.margin` | 〃 |
| `crop_classifier` | `fit_train`: GT crops (positive) and false positives from a fit_train pass (background), mined at the `fit_dev` thresholds | the λ grid is scored on `fit_dev`; the best ΔF must exceed `crop_classifier.margin` | 〃 |

## Round loop

```
diagnose → propose → materialize → screen (fit_dev) → evaluate (holdout) → decide → record
```

The diagnosis comes first: it is of the champion at the start of the round
(reused when the champion and config are unchanged — see below), and it is the
planner's input for the same round.

- **screen** passes only the top-K ordinary candidates to holdout (lora and
  classifier arms are judged on their own margin and are not counted in `top_k`). A candidate that
  changes only tiling and raises the grid above the champion's to ≥
  `screen.prescreen_min_grid` (2 by default, i.e. every grid — grid 2 already
  costs 5 forwards per image, and a newly turned-on grid going unused is most
  common at grid 2) is first
  screened on a deterministic fit_dev subset (`screen.prescreen_fraction`,
  `prescreen_min_images`); if no class uses tiles there, it is rejected without
  the full screen. If they are used, the full fit_dev is screened as before
  (numbers unchanged). If the subset is `screen.prescreen_max_share` (0.5) of
  fit_dev or more, the full screen runs directly without a prescreen — a pass
  would be measured again, so nothing is saved. A grid below the champion's
  is not prescreened: it is cheaper, and unused tiles there do not make it
  the champion (the champion uses tiles), so it gets the full screen.
- **lost-value prescreen**: a planner candidate containing a value that clearly
  lost on fit_dev under the current champion (a candidate changing only that
  value had a (candidate − champion) fit_dev micro F(β) paired-bootstrap upper
  bound `fit_dev_cmp.ub` < 0, and no candidate containing it had an upper
  bound ≥ 0) is not blocked but measured on the same subset first (without a
  subset — fit_dev is small or the subset is `screen.prescreen_max_share` or
  more — the same comparison runs on the full fit_dev screen directly, and if
  it passes that screen is used as is). With an upper bound ≥ 0 it proceeds
  (no point-estimate win is required); otherwise it is rejected as
  `lost_value_prescreen` without the rescorer refit, full screen or holdout
  (`screen.lost_value_prescreen`, on by default). A value is the axis's main
  value (alias=vlm, exemplar=fn_arm …) and quantity axes (`exemplar_base_k`,
  `exemplar_fn_k`, …) are its variants. A combination containing any clearly-lost value
  is a variant too. The axes covered are alias, exemplar, text_negatives,
  visual_negatives and tiling — rescorer, lora and crop_classifier are left out
  because their expensive part (refit, training) comes before the comparison,
  as are values that ever changed the champion and human or carried
  candidates. Holdout results are not used for the verdict.
- **pre-holdout fit_dev verdict**: among screened candidates, those that
  clearly lost to the champion on fit_dev (`fit_dev_cmp.ub` < 0 →
  `fit_dev_lost`) and those whose per-image fit_dev (tp, fp, fn) all equal the
  champion's (`fit_dev_identical`) are not sent to holdout
  (`screen.skip_fit_dev_losers`, on by default). The same rule applies to plain
  and carried candidates and to the lora and classifier arms, and `top_k`
  counts only sendable candidates (the next candidate fills a freed slot).
  Without comparison rows nothing is filtered.
- **decide** adopts a candidate only if the one-sided paired-bootstrap lower
  bound of its holdout Δ is ≥ `gate.margin`, Δ > 0 and Δ ≥ `gate.min_delta`
  (0.005 — a noise-level gain with its lower bound at 0 does not change the
  champion and reset the ledger and fold records); otherwise the champion
  stays. Raising `gate.margin` instead would also reject real gains on small
  holdouts, where the interval is wide. Resampling runs on rows of per-image
  (tp, fp, fn) counted once (`metrics.image_counts`) — micro F(β) is set by
  their integer sums, so it equals re-matching the boxes on every resample.
  The per-class threshold search likewise matches once per class (in
  score-ordered greedy matching, the matching at τ is the score ≥ τ part of
  the full matching).
- **re-ask**: if every candidate is rejected before holdout (by the filters or
  screen: `tiling_blocked`, `axis_unused`, `same_setting`, `no_material`, …),
  the round is not closed — the planner is asked again within the same round
  (`planner.max_reasks`, default 1, 0 = off). The follow-up carries each
  candidate's skip code with an English reason and the served material;
  diagnosis and champion are reused. Re-asked candidates go through the same
  filters, screen and holdout and stay in the same round's history as
  `reask: k` — only the axis status (`axis_retired`) counts from the round's
  start, to match the re-ask prompt's axis board. If the re-ask call fails, the
  failure is kept in `proposal_reask<k>.json` and the round closes (the
  remaining `max_reasks` are not used). LLM planner only (the deterministic
  planner is not re-asked).
- **combination dedup**: an axes combination already tried in the run is
  dropped as `combo_tried`, except one that failed with `skipped: error` (OOM,
  worker failure — not a measurement): it is allowed once more
  (`planner.deterministic.ERROR_RETRIES` = 1) and blocked from its second
  failure on, so a deterministic failure does not repeat every round.
- **re-proposing the champion does not use up the re-ask**: a combination equal
  to the current champion's axes, or to the request that adopted it (the
  history row's `axes`/`requested_axes` — it can differ from the label after an
  unused-axis correction or an identity rescorer refit), is dropped as
  `champion_combo` (apart from the plain `combo_tried`). If a proposal is only
  such no-op drops (`champion_combo`, `no_change`), the planner is re-asked once
  per round (`FREE_REASKS`) without counting toward `planner.max_reasks`,
  recorded as `free_reasks` on the decision. The cap keeps a planner that
  repeats itself from never ending the round; `max_reasks: 0` turns this off
  too. The re-ask number k runs in order either way, and the verdict reads only
  the replayed proposal and history, so resume decides the same. The prompt's
  tried-combination list also names that request as the champion itself.
- **candidate ids**: the loop adds an `r<round>_` prefix before recording
  (`r2_cand_0`, re-ask `r2_reask1_0`, human candidate `r2_human_x`) — ids stay
  unique within the run even when the planner reuses them every round. Carried
  candidates are `carry_<original id>` (e.g. `carry_r1_cand_1`) — only carried
  candidates made by the loop (`source: carryover`) are exempt from the prefix.
  If the planner imitates a `carry_` id, `carry_` is stripped and the round
  prefix added (carried rows are told by source, not by id shape). The deterministic
  planner uses the same `cand_<i>` as the LLM planner. The id is the candidate's
  material and arm folder name (`settings/<id>/`) as is — it is put in
  folder-safe characters (`A-Za-z0-9_.-`) before deduping within the round, so a
  unique id is a unique folder. When every LLM candidate is invalid and the round
  is filled by the deterministic planner, candidates are `det_<i>` — they do not
  clash with the dropped LLM rows (`cand_<i>`), so proposal and history ids map
  1:1. Deterministic candidate rows carry their `remedy` (kind:axis=value, e.g.
  `missed:exemplar=base`).
- **candidate count**: the planner proposes up to `planner.max_candidates`
  (default 3) per round (and per re-ask) — the prompt, the response schema and
  the deterministic planner follow the same value. The range is 1–3 — larger
  responses can exceed the output cap (4096 tokens), and a cut response silently
  falls back to deterministic, so config rejects it.
- **saturation** stops the run after `saturation.rounds` rounds without a
  meaningful adoption. A round
  that measured nothing on holdout even after re-asking (an empty round) stops
  the run earlier after `saturation.empty_rounds` (default 2, 0 = off)
  consecutive ones (`stopped: "empty_rounds"`). Empty rounds are also stagnant
  and count toward `saturation.rounds`. A round that missed holdout only
  because of the pre-holdout fit_dev verdict (`fit_dev_lost`,
  `fit_dev_identical`) is not counted as empty (it is still stagnant) — the
  skip only saves holdout time and does not change when the run stops.
- **The planner never stops the run.** The LLM response schema has no stop
  field and asks for at least one candidate (constrained decoding). Left to the
  planner, stopping split runs on the same path by the LLM's reply, and it often
  came right after a champion change, before the detector-state axes were
  re-measured on the new champion. Useless rounds are cut by the empty-round and
  stagnation counters above. An empty reply (a server without constrained
  decoding) counts as a failed reply and falls back to deterministic (`no valid
  candidate: empty candidate list`). The deterministic planner just returns no
  candidates once its mapping is exhausted, which makes an empty round. To
  continue such a run, raise `saturation.empty_rounds` and `saturation.rounds`
  and `--resume` with `candidates.yaml` — resume restores both counters from the
  history tail, so the candidates file alone would stop it again at once. With
  `saturation.empty_rounds: 0` (off), empty rounds are cut only by the
  stagnation count.
- **diagnosis reuse**: if the champion and config equal the previous round's,
  the diagnosis (full train inference) and crop notes are not recomputed; the
  previous round's are copied (`diagnosis_key.json` — the champion's round, id,
  materials, weights, artifact, blend λ, rescorer, holdout micro F(β) and
  thresholds, and a hash of the config without `rounds` and `out_dir`).
- **final champion diagnosis**: a round's diagnosis is of the champion at the
  start of the round, so when the last round adopts, the final champion's
  residual errors would be recorded nowhere. At the end of the run the final
  champion is diagnosed once more into `rounds/final/` — if the last round
  adopted nothing this only copies its diagnosis (no inference); if it adopted,
  it is one train inference plus crop notes. The structure of residual misses
  (`residual.json` — size, crowding, border and aspect buckets, label quality
  flags) is computed from GT annotations and the diagnosis alone. Resume is
  idempotent, and continuing a finished run with more rounds lets the next
  round reuse this diagnosis. It is report-only: a failure still ends the run
  and is recorded in `run_summary.json` as `final_diagnosis.error`.
- A round adopts one candidate, the one with the largest Δ. Candidates that
  passed the gate but lost to it are marked `gate_passed_but_lost` and carried
  into the next round as `carry_<id>`: the axes they changed are applied on top
  of the **new** champion. That makes each one a single increment over the new
  champion, and holdout is not reused within the round. Carried candidates go
  through the same filters as planner and human candidates, whichever planner
  is used.
- The same setting is not evaluated twice. After screen, a candidate's served
  state (per class: config content and modality, the materials that modality
  uses, rescorer on/off, server state) is compared with the champion, its
  siblings in the round, and candidates evaluated on holdout before; a match
  is rejected without holdout evaluation. Axis labels are not compared,
  because materials can differ under the same label. If an axis the candidate
  turned on (tiling, exemplar, visual_negatives) went unused and that made it
  equal the champion, the reason is `axis_unused` (which axes in
  `unused_axes`); otherwise `same_setting`. Rescorer candidates are judged
  before the discriminator refit, the most expensive step. An evaluated
  candidate with an unused axis gets its label corrected to the measured state,
  with the request kept in `requested_axes`.
- Sizing signals are recorded in history: for tiling candidates, whether
  screen actually picked a tile config (`tiling_used`); for visual-negative
  candidates, the suppression IoU thresholds picked for semantic classes
  (`neg_eta_used`) and the grid searched (`neg_eta_grid`). The planner reads
  them before widening a count axis: tiling used on an adopted grid suggests
  g+1, negatives picked at the low (strongest) end suggest raising
  `visual_negative_k`, and picks at the high end suggest folding the axis.
  Candidates that ran a semantic pass also record each class's fit_dev F(β)
  for the semantic and text passes (`modality_f`), including candidates
  filtered before holdout, so the planner can tell whether exemplar lost
  narrowly (try another k or fn_arm) or badly. The prompt carries only a
  summary whose size does not depend on the class count (classes text won and
  the narrowest margin); the full per-class record would make the budget
  guard shrink the history window on datasets with many classes.
- lora and crop_classifier change the detector state (weights / artifact), so
  they are processed after the other candidates, and the state is restored if
  they are not adopted. Once adopted they are not turned off: the state stays
  loaded while other candidates are measured, so a candidate turning one off
  would be measured with it still on. Such candidates are rejected as
  `detector_state_off` without evaluation. Every attempt (the first proposal
  and each re-ask) starts by bringing the detector back to the champion state,
  so an arm that cleared its margin but was dropped before holdout does not
  stay loaded under the next candidates. The end-of-round restore goes through
  the same step, and it compares the artifact and λ against what is actually
  loaded rather than the round record. Each detector-state candidate also
  starts with this restore: an earlier candidate's adopted arm may still be
  loaded from its holdout, and the candidate's materials (the FP-mining train
  pass) must be mined on the champion detector before its arms run. A swap to
  the state already loaded is not sent (weights `Loop._weights_ctl`, artifact
  `Loop._adapt_ctl`), because every swap also empties the pass cache.
- **classifier blend λ**: the detector state is (weights, artifact, λ). λ is
  the loop-side blend (`infer.gateway.set_classifier_blend`), so changing only
  λ does not reload the artifact, and the pass cache keeps raw responses (with
  probabilities) so re-inference is served from it. The classifier arm keeps
  the baseline (no classifier, `lam: null`) apart from λ = 0 (relabel-only); a
  champion's `adapt_lambda` is `null` without a classifier and `0.0` for
  relabel-only. While the champion has a classifier, ordinary candidates
  re-choose λ on their own setting before the rescorer refit
  (`crop_classifier.retune_lambda`), switching only when another λ beats the
  champion's by more than `crop_classifier.margin`. Lora candidates re-choose
  it too, on the winning arm: lora changes the detections themselves, so a
  layer below it (classifier λ, then the rescorer) is refit on top, and with
  the winning arm loaded only λ changes, so the detector responses are the same
  and it costs almost nothing (if the winner is not the last arm measured, the
  winner is loaded again and fit_dev is inferred once more). The history row
  keeps the final setting (chosen λ, refit rescorer) and its fit_dev, like an
  ordinary candidate; the arm grid's best stays in the lora decision. The lora arm grid (which lora λ wins) is chosen
  on the champion's classifier λ, and the classifier λ once on that winner —
  the two grids are not swept as a product, so the number of picks on fit_dev
  matches an ordinary candidate's. The lora λ itself (weight interpolation) is
  not re-chosen after adoption: rebuilding the weights, re-inferring and
  refitting the layers below would all run again. The training cache key
  leaves the champion out — background crops come from the first training's
  champion, and the λ grid absorbs the drift. A tripwire is not cached.
  Resume and `scripts/predict.py` restore λ with the artifact; a champion with
  an artifact but no λ (from before the blend) is refused.

## Diagnosis and planner

- Every round classifies train and holdout errors into eight causes (missed,
  missed_small, below_threshold, class_confusion, localization, background,
  duplicate, tiling_artifact — the keys of `planner.deterministic.REMEDY`);
  this is the planner's only input. Holdout is exposed as aggregates only.
- Candidate axes are read as an increment over the current champion. The LLM
  planner and human candidates list only the axes they change; omitted axes
  keep the champion's value (turning one off takes its off value — `"none"` for
  exemplar, `false` for the others).
- The LLM planner (vLLM, guided JSON) only proposes — the holdout gate decides.
  Prompts and responses are saved, so `--resume` replays them without the LLM
  (same-round re-asks too, `proposal_reask<k>.json`). Response length is
  bounded by the schema (rationale, id, candidate count). A reply that does not
  parse as JSON is asked once more (one-off degeneration); if it still fails,
  the round falls back to deterministic, and the warning and the proposal's
  `error` keep the finish reason, the token count and the head and tail of the
  reply. Fallback candidates carry `source: fallback`, and `run_summary.json`
  lists the fallback rounds with their reason (`planner_fallbacks` — it should
  be empty). When the reply arrives but the sanitizer drops every candidate,
  the reason is `no valid candidate: <skip: reason; …>`, logged at ERROR like
  the other fallbacks. One call per round, plus at most one follow-up in the same
  conversation when the planner repeats carried candidates: those are dropped
  as `carried_duplicate` and the freed slots are asked for again.
- **The prompt is English only.** A candidate row's `note` is a sentence for
  people and is not carried; a skip reaches the prompt as its code and an
  English `reason` built from structured fields (`planner.llm._REASON` — the
  arm verdict numbers in `arm_gate`, the violated rule of an invalid candidate
  in `invalid`, the empty axis, the exception type, the prescreen size). A new
  skip code needs a reason there too (a test blocks it). `dataset_notes` should
  be English; the planner logs a warning otherwise. The stored key `test`
  (holdout) is carried as `held_out` (`held_out_micro_f_beta`, the error
  detail's `held_out`).
- **Prompt budget.** The budget is the planner server's context minus the
  output cap and template slack — the context the attached server reports
  (`/v1/models`), else `planner.launch.max_model_len` (32768). When the prompt is
  over, the budget ladder shrinks it in order: history window 3 → 2 → 1, then
  half the diagnosis samples, then no samples, then window 0 — the last round's
  detail is worth more than the samples, since why the last candidates lost is
  the most direct evidence for the next proposal. The step used is kept in the
  proposal (`budget`). If even the last step is over, the prompt is not sent and
  the round falls back with a `PromptOverBudget` reason.
- The other paths on the same vLLM follow the same rule: negative and alias
  phrase refinement, the VLM crop captioner and the FP/FN crop notes ask once
  more on an unparseable reply, and if that fails too each takes its own
  fallback (drop the phrases, keep the class name, skip the caption batch, no
  notes). Crop notes are a diagnostic signal only and never decide anything.
  - Each splits a system message (role, rules) from a user message (data). The
    two paths that send image batches (captioner, crop notes) upscale crops whose
    short side is below `core.coco.MIN_VLM_CROP_PX`, and the response schema asks
    for exactly as many entries as the batch has crops.
  - The captioner puts one class per batch (mixed, "same wording for look-alikes"
    carries one class's phrases over to another). Every phrase needs a type noun,
    and a hard-to-make-out crop gets a best guess — the schema rules out an
    empty list.
  - Crop notes get a glossary of the error kinds and what `class=` means (the
    predicted class for a false positive, the true class for a miss), and are told
    not to restate the kind or cite our upscaling as a cause.
  - Negative refinement states the pool's source (fp_crop, gt_crop, background)
    truthfully and blocks phrases naming the target or any other category — the
    client emulates suppression target-agnostically (a negative removes any
    overlapping positive), so a negative naming another category would suppress
    it.
    Rejections come back as rule numbers only (so a long reason cannot be cut off
    and lose the whole class), and code cuts the `text_negative_k` cap by count.
  - Category-name checks in alias and negative refinement are whole-word ('fish'
    does not block 'jellyfish'), and plurals (-s, -es) count as the same name.
- `--planner deterministic` maps error causes to remedy axes by rule; it backs
  the tests and serves as the A/B control.
- **Prompt sections.** Objective → Hard constraints (conditions whose breach
  drops a candidate without evaluation — visual_negatives without an exemplar,
  turning a detector-state axis off, turning lora and crop_classifier on
  together, a folded tiling grid, OFF axes; lines about disabled axes are left
  out. The tried-combinations line below the board states the no-repeat rule) → Rules (general rules only — the tiling
  grid, its cost and how to pick the first grid only when tiling is not
  disabled) → Budget (rounds left, stop counters) → Current champion → Axis
  board → Round history (the last `MAX_HISTORY` = 3 rounds in detail, earlier
  rounds one line each; explanations of the detail rows' fields — fit_dev_cmp,
  tiling_used, unused_axes, modality, neg_eta_used, same_as,
  gate_passed_but_lost, lost_values, wall_min, material — follow right below,
  only for fields present in some row) → Diagnosis digest (a one-line definition
  of the error kinds, `ERROR_KINDS` — every FP/FN kind in `core.metrics` — then per cause, this
  round's count in train and held-out and the axis=values that can be proposed
  now for it — left out: disabled axes, values the data blocks, values the
  champion already uses, OFF axes, values with empty material under the current
  champion, tiling grids at or below the champion's or folded, and
  visual_negatives without an exemplar champion. A cause whose remedies are all
  left out says "no axis to propose now"; only a cause with no remedy at all is
  "not addressable by a setting") → the missed-small block (only when a grid
  larger than the champion's can be proposed, listing those grids) → error
  detail tables (per-class aggregates per split — gt, miss rate, FN and FP by
  cause — and train error samples — cause, class, score or size, box size, crop
  note) → Dataset notes (when `dataset_notes` is set) → Response. The filter
  blocks every tried combination; the prompt carries only what is worth
  proposing again.
- **Axis board (`planner/board.py`).** One row per function axis gathers the
  facts for deciding, so the same fact is not repeated in other words across
  sections. Columns: status (ON · OPEN · PROBE · OFF — see axis retirement
  below — and BLOCKED or "PROBE (nothing to propose now)" when nothing can be
  proposed now), propose (the values and quantity keys that may be proposed
  now — values the data blocks, folded tiling grids, the champion's value and
  values with empty material under the current champion are left out), run (the
  whole-run record, round × axis), champion (the best fit_dev / held-out Δ and
  status per value under the current champion — `planner.ledger`; k variants of
  the champion's value and narrow losses first, numbers for the first two and
  names and status for the rest), min (median, count and max of observed
  minutes per candidate — per grid for tiling; axes not observed yet get facts
  in the note) and note (constraints of that axis alone — detector state,
  needs exemplar, folded, too little data, OFF axes only together with a non-OFF
  axis, every remaining value mined empty or untrained). The legend states the
  off value per axis and when a value with empty material may be proposed
  again. Below the table come only details the rows point to — the evidence for
  the one-time tiling exception, prescreened values, what each axis does
  (`AXIS_DOC` — exemplar, alias, rescorer, lora, crop_classifier, minus disabled
  axes), the negative-source explanation, disabled axes and values, the axis retirement rule, the combos
  tried under the current champion (those under it, and earlier ones that do not
  turn the current champion's axes off) and carried candidates. axis_status,
  ledger and gates decide; the board only presents.
- **Untried axes are suggested first (guidance only, nothing is blocked).** An
  axis whose run column says "never tried in this run" (axes that cannot be
  proposed alone now — BLOCKED, or visual_negatives without an exemplar
  champion — say "not tried" instead and are not targets) is preferred over
  another variant of an already-tried value unless the diagnosis gives a
  concrete reason (one sentence in the Rules). The basis is the axis status's
  try count (a quantity variant counts for its function axis; rows dropped
  without being measured do not count). With few candidates per round the order
  is the budget. Re-measuring a "not retried under this champion" axis is not
  such a variant (its last result is from another champion). The deterministic
  planner is unchanged.
- **Always propose.** There is no stop field. When nothing looks promising, the
  Rules say to propose the most informative remaining change — a "not retried
  under this champion" axis, then an untried one.
- If the response's `expected_gain`, `cost_min` or `differs` is missing or
  malformed only that field is dropped — the candidate is neither discarded
  nor sent to the deterministic planner. They are recorded as the candidate
  row's `plan`.

### Axis retirement — turning off axes without effect during a run

The axis ledger looks only under the current champion and resets when the
champion changes, so the same axis is retried from scratch under every
champion and an axis that lost several times on holdout keeps taking slots.
`planner/axis_status.py` decides a status per function axis (ON · OPEN · OFF ·
PROBE) from the **whole run's** history (`search.axis_retire_after`, default 2;
`null` records only and turns nothing off).

- **Each round counts once per axis** (quantity axes fold into their function
  axis). If any candidate that changed the axis that round wins, it is a
  success; if its measured outcomes are all failures, one failure — three
  variants losing in one round are still one. A tie between failure kinds takes
  a fixed order (held-out > fit_dev > arm > equal > unused), so the same history
  gives the same status.
  - Failure: the held-out gate failed, clearly lost on fit_dev (`fit_dev_lost`,
    `lost_value_prescreen`), equal to the champion (`fit_dev_identical`, and
    `same_setting` with `same_as: champion`), the arm's own verdict
    (`rescorer_identity`, `lora_gate`, `classifier_gate` — that arm axis only),
    the setting did not use the axis (`unused_axes` — the unused axis only; the
    row's other axes take the row's outcome).
  - A combination's outcome goes to every axis it changed. Which one is to
    blame cannot be told apart, but retirement needs consecutive failures, so
    an axis that wins once comes back.
  - Success (streak back to 0): adopted, or passed the gate but lost.
  - Not counted: **no material** (`no_material` and `empty_axis` — including
    the classifier and tail fine-tune tripwires; a problem of the value), a
    setting equal to a sibling's or an earlier candidate's (`same_as: sibling|tried`),
    outside screen's top K, and unmeasured drops. A candidate that only turns an
    axis off does not try it.
- After `axis_retire_after` failures in a row the axis is OFF. A candidate whose
  changed function axes are **all** OFF is recorded as `axis_retired` by the
  shared filter without evaluation; a combination with an open axis is not
  blocked.
- When the champion changes, OFF axes reopen once as PROBE — one more failure
  turns them OFF again. Axes the champion has on are ON.
- **Axes worth re-measuring.** An OPEN or PROBE axis whose last measured outcome
  was a held-out fail, a fit_dev loss or an arm gate, after which the champion
  changed and the axis has not been tried under the new champion (other
  candidates of the adopting round were measured on the old champion too; a try
  that ended without an outcome — outside screen's top K, no material, same as a
  sibling — counts as a try). The board's run column gets "not retried under
  this champion (champion changed in R<n>)" (only when a value can be proposed),
  and the Rules and the re-ask message suggest it before repeating an unused
  axis. The re-ask's board is from the round's start, so the message adds to
  leave out axes already proposed this round. Detector-state axes
  (crop_classifier, lora) tend to change with upstream changes. Unused and
  no-change are left out — they are likely the same after a champion change.
- **LLM planner only.** The deterministic planner is the A/B control and stays
  as is. Human and carried candidates are not blocked, for the same reason as
  the tiling fold, and neither are the deterministic fallback candidates of a
  round where the LLM failed (`source: fallback`) — a safety net that knows
  nothing of OFF, so blocking them would make an empty round.
- `same_as` (champion, sibling, tried) on same-setting rows reaches the prompt
  too, so the planner can read which ones counted as failures.
- `axis_retired` rows are not tried combinations (`tried_combos`, the prompt's
  combo list) — they were not measured, so the combination can be proposed again
  once the axis reopens.
- It reads held-out gate results. "What to try next" is within what the holdout
  may change (the core invariant); it never touches materials or thresholds.
- `scripts/planner_replay.py` measures the effect: for each round of a recorded
  run it asks the planner again with and without OFF (vLLM only, no detector)
  and compares the proposals on OFF axes, hygiene failures and hits on the
  adopted candidate.

## Outputs reference

```
runs/<dataset>/run/
  history.jsonl        # round ledger (per-candidate wall_min, reask, fit_dev_cmp, lost_values, plan,
                       # delta_reused, classifier_reused, classifier_lambda(_grid);
                       # per-round pass_cache counts)
  run_summary.json     # stop reason (rounds, saturation, empty_rounds), final champion,
                       # holdout_split, fit_train (training images, images before the subset,
                       # selection record), final_diagnosis (reused or not, duration)
  fit_train_subset.json  # fit_train subset (key, picked images, selection record) — above the cap only
  _subsample/gt_crops/ # GT crops embedded for the subset selection
  champion.json        # champion setting (axes, materials, weights, adapt_root, adapt_lambda,
                       # forwards_per_image,
                       # per-image fit_dev (tp, fp, fn) — the baseline of the next fit_dev_cmp)
  champion_eval.json   # champion holdout evaluation, incl. tau_test
  rounds/round_NNN/    # proposal (+ proposal_reask<k>), settings/<cand>/ materials, screen, eval,
                       # decision, diagnosis (+ diagnosis_key), crop_notes
  rounds/final/        # final champion diagnosis (diagnosis, diagnosis_key, crop_notes) + residual.json
  lora_cache/          # tail training output (tuned_tail.pt) per training key — reused within the run
  classifier_cache/    # crop classifier artifact per training key — reused within the run
  report.md            # run report (written at run end, redrawn by acs report)
  report_assets/       # report figures (chart PNGs, photo JPEGs)
```

- **Candidate ids** get an `r<round>_` prefix before recording; only carried
  candidates keep `carry_<original id>` (see the round loop). The id is optional
  in the LLM response — an empty one becomes `cand_<i>` (re-ask `reask<k>_<i>`),
  and a clash gets `_2`, `_3`….
- **History fields**: candidate rows carry `wall_min`, `reask`, `fit_dev_cmp`,
  `same_as` (same_setting rows), `arm_gate` (arm verdict numbers), `invalid`
  (the violated rule of an invalid candidate),
  `lost_values`, `plan`, `delta_reused` / `classifier_reused` (origin of a
  reused tail or classifier training) and `classifier_lambda` (the arm's best λ,
  or the λ an ordinary candidate re-chose, with `classifier_lambda_grid`),
  `source` (carryover or human), `remedy` (deterministic candidates),
  `prescreen` (the subset verdict) and `tiling_f` (tiling candidates' fit_dev
  comparison); the round's champion summary carries `classifier_lambda`; round rows carry
  `pass_cache`, and decisions `reasks`/`free_reasks`. `plan` is the LLM
  response's `expected_gain`, `cost_min` and `differs`; only a malformed field
  is dropped.
- **Rescorer refit records** (artifact only — not in the planner prompt): a
  candidate that refit the rescorer carries `rescorer_min`, the minutes per
  stage — fit_train and fit_dev inference with embeddings (`infer_train`,
  `infer_dev`), the discriminator fit (`discriminator`), the α grid (`alpha`) and
  the comparison below (`cmp`). `cmp` is record-only and kept out of `wall_min`.
  An ordinary candidate refits inside screen, but lora and classifier arms refit
  after screen closes (after the arm grid), so their refit time is not in
  `wall_min` and appears only here. In rounds where the champion has a rescorer,
  such a candidate also carries `rescorer_cmp`: micro F(β) on the same fit_dev
  predictions for the refit (`refit`), the champion's discriminator and α as they
  are (`champion_rescorer`), and the champion's discriminator with α re-picked
  (`champion_disc_alpha`). It is a record for weighing a scheme that trains the
  discriminator only when the champion changes, never used for a decision, and
  adds no inference; if it fails, only `refit` and `error` are kept and the
  candidate goes on. The discriminator (GBDT) fit runs on `rescorer.fit_threads`
  (8) threads — the entry point pins threads to 1, so they are released only
  there, and the result does not depend on the thread count.
- **Final diagnosis** (`rounds/final/`): a failure still ends the run, and the
  error is kept in `run_summary.json` as `final_diagnosis.error`.
- **Rescorer row cap**: a rescorer candidate's `rescorer_info.json` carries
  `n_train_raw` (rows before the cap), `train_rows_capped` and `wall_s` (seconds
  per stage — inference, discriminator, α).

### Report

`report.md` (`diagnose/story.py`) is a run report for readers without the
technical background. It is written automatically at the end of a run and
`acs report <run_dir> [--config c.yaml]` redraws it (no GPU, no LLM); figures go
to `report_assets/`. The order is at a glance → the data (counts, sizes,
crowding, representative photos) → attempts and the score chart by round → the
remaining error patterns with example photos → data suggestions, with a
glossary appendix at the end. Its material
(trajectory, candidate attempts, the final diagnosis bundle) is read by
`diagnose/report.py`.

- Technical terms reach it only through the `STRATEGY`, `CAUSE` and
  `_OUTCOME_GROUPS` tables. A new axis, error cause or skip reason must be added
  to them too (`test_story`).
- Scores and counts are holdout aggregates; example photos come from train only.
- Example captions are the VLM crop notes as written; without a note, a fixed
  sentence built from the cause and size is used.
- The score is the holdout score. The holdout is reused for adoption decisions,
  so the report says scores "may come out a little better than on photos never
  seen before", and warns when `holdout_split` is not valid (test as a
  stand-in).
- `rounds/final` counts as the final diagnosis only when the champion id in its
  diagnosis key matches the current champion (after a resume, a failed final
  diagnosis of the new champion leaves the old files behind). Otherwise the last
  round's diagnosis stands in. It is stale only when the last round adopted
  (otherwise it is the final champion's own diagnosis): then the summary and
  sections 3 and 4 say so, and no "final" bars are drawn. A run whose final diagnosis failed does not use
  `residual.json`. Without a holdout diagnosis, sections 3 and 4 are skipped.
- Example photos are picked only from rows of train files, and captions are
  numbered only for photos actually drawn. A photo that cannot be opened drops
  only that photo.
- The data suggestions split the remaining errors into "more photos are likely
  to reduce" (not seen, too small to see, background taken for an object, taken
  for another kind), "a decision-rule problem" (low confidence, duplicates, tile
  fragments) and "check found areas and label rules" (area off target). If the
  first is under 30% of the remaining errors, the decision rule and label rules
  come first. The number of photos needed is not stated — no learning curve is
  measured.

### Prediction

`scripts/predict.py` replays a run's champion:

```bash
.venv/bin/python scripts/predict.py runs/<ds>/run --config <run config> --split test
.venv/bin/python scripts/predict.py runs/<ds>/run --config <run config> --images /path/to/dir
```

- It uses `tau_test` from `champion_eval.json` as the threshold (not the
  `fit_dev` threshold in `champion.json`).
- `--images <dir>` predicts on images without GT instead of a split.
- Paths in `champion.json` are absolute. To replay a run directory that was
  moved, pass `--rewrite-root OLD=NEW` (repeatable).
- `--coco-map` also computes COCO mAP@[.5:.95] and AP50 on all floor
  detections before the τ cut (τ_c is only the F(β) operating point).
- `--config` is required because the run directory does not snapshot the
  config.

**The marginal contribution per component is measured by `scripts/ablate.py`.**
The loop's Δ is the increment at adoption, so it is confounded by adoption
order. ablate keeps the champion's setting fixed, removes one component at a
time (the axis and its dependents — count axes, and visual_negatives when
exemplar goes), re-measures on holdout, and reports F(champion) − F(champion \ C)
with a paired-bootstrap CI. Removing a detector-state axis takes (weights,
artifact, λ) down together by the replay rule above and empties the pass cache.
Removing a mining axis inherits the champion's materials for every other axis,
as the loop does, and the re-mining LLM refinement and VLM caption hooks only
attach to a running planner vLLM (none is launched).
