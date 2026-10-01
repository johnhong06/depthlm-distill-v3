# depthlm-distill-v3

Distilling the metric depth ability of DepthLM (Pixtral-12B) into a Qwen2.5-VL-3B student with LoRA, using only the
teacher's answers as labels — redesigned (2026-09-30) so that **every query budget has an allocation comparison**.
This repository supersedes `depthlm-distill` / `depthlm-distill-h200`; pools and teacher labels are carried over
unchanged, prior results are not. Design rationale: [paper/design_v3_corrections.md](paper/design_v3_corrections.md) (Korean).

## Study at a glance

| | |
|---|---|
| Question 1: allocation | With the number of teacher queries fixed at B, is it better to spread B over many distinct images with few pixels each, or few images with many pixels each? |
| Question 2: training signal | Cross-entropy on the teacher's single answer (`hard`) or KL to the teacher's digit distribution (`soft`)? |
| Grid | budget B ∈ {400, 1600, 6400} × pixels-per-image k ∈ {1, 4, 16}, N = B/k — 9 nested cells, every budget row is a 3-way equal-budget comparison (larger budgets can be re-added later by the fixed rule: every k with N = B/k ≤ pool size) |
| Pools | indoor (SUN RGB-D, NYUv2), driving (KITTI), mixed (50/50), 6,400 images each, order v5 (near-duplicates demoted: dHash Hamming ≤ 10 and 32×32 grayscale correlation > 0.9). Every cell with N ≤ 1,600 has only distinct images; at N = 6,400 the demoted near-duplicates come back, so driving has 5,035 and mixed 5,651 distinct images (indoor 6,400) — `tables/cell_stats.md`. Pre-registered: that cell is reported as "6,400 (5,035 / 5,651 distinct)" and allocation claims are phrased in distinct-image counts. Mixed alternates driving and indoor image by image, so every cell is 50/50 (B6400_k1 = 3,200 + 3,200) |
| Labels | DepthLM 12B answers on the pool pixels; the v3 grid and its replicate cells are fully covered by the existing 44,800 labels per pool — **zero new teacher queries** (verified by `experiments/03_build_cells_v3.py`) |
| Student input/output | **the teacher's own official DepthLM format**: the official question, then the template `<think> The point is around ` is filled in and the student answers the number (first decimal) followed by ` meters`. The teacher labels and its digit distributions were read in exactly this position, so the student learns the same conditional; image positions use Qwen2.5-VL's 3-D M-RoPE in training and evaluation alike |
| Ground truth | never used by the pure arms (`soft`, `hard`); the † arms `gt+soft` / `gt+hard` add λ = 0.1 · CE(GT) and are reported beside them but excluded from the equal-budget ranking |
| Noise floor | the B = 400 arms are re-run over training subsets (image blocks / pixel indices, free relabelings) and over seeds 0/1/2; equal-budget claims must exceed the larger spread |
| Evaluation | δ1 (max(pred/gt, gt/pred) < 1.25, identical to the official DepthLM metric) on iBims-1, NYUv2 (indoor), DDAD, nuScenes (driving), ETH3D fully held out; each pool on its own domain only |

Why this design: DepthLM's Finding 4 ("1 labeled pixel per training image … image diversity is more important than
label density", arXiv:2509.25413) rests on a sample-count scaling curve (Fig. 4c), not on a fixed-budget
images-vs-pixels comparison — the paper contains none. We run that comparison under conditions the paper never
tested: fixed budget, noisy pseudo-labels, and a small LoRA student. The allocation axis is deliberately
**distinct images, not "scene diversity"**: scene labels are capture metadata and cannot prove distinctness,
so scene counts are reported as descriptive statistics only.

## Comparability with DepthLM

**Identical to the official code.**
- Image preprocessing: the official `undistort_image` and `normalizing_focal_length` are imported from `third_party/`, with a unified focal length of 750 (the 12B teacher's setting; DepthLM's own 3B/7B used 1,000).
- Marker: the same red arrow, a 5-px shaft plus head, as in the official evaluation dataset.
- Prompt and answer: the student gets the official question (`generate_prompt_depth_sft`) and answers after the official template `<think> The point is around `.
- Metric: δ1 = max(pred/gt, gt/pred) < 1.25, as in the official `metrics.py`.
- Mixing: the mixed pool is 50/50 indoor/driving, matching DepthLM's ≈ 49 % driving.

**Deliberate differences.**
- Distance definition: every set is scored against Euclidean distance. The official indoor scripts use Euclidean distance and the official driving scripts use z-depth; z is kept for paper comparisons (`gt_z` in the evaluation files, `ground_truth_z` in `ref/`).
- Resolution: one decimal plus midpoint decoding, where the paper uses two decimals. This costs the teacher ≤ 0.006 δ1 (D-75), and the 1-vs-2-decimal ablation is planned.
- Loss: on the number tokens only, with the template filled in; the paper uses CE on the whole answer.
- Training setup: no random-crop augmentation, and LoRA on a 3B student.
- Supervision and scale: ≤ 6,400 teacher-labelled pixels, against 16 M GT images in the paper.

**Where a comparison is fair.**
- **Student against teacher on identical pixels** is the primary comparison, i.e. how much of the teacher transfers.
- **Our teacher against the paper's public Pixtral-12B row**:
  - Matches (z-depth): DDAD 0.676 against 0.670, and nuScenes 0.802 against 0.819, inside the scene-cluster CI.
  - Matches: ETH3D 0.647 (two-decimal answers) against 0.653.
  - Does not match: iBims-1 0.809 against 0.870, not explained. Our NYUv2 set is our own 200-image curation from the labelled `.mat`, not the paper's images, so its 0.881 is not comparable to the paper's 0.799.
- **DepthLM's 3B** (Qwen2.5-VL-3B, the same backbone, trained on GT with 16 M images) is a reference ceiling, not a like-for-like comparison.

## Layout

```
run.sh                             H200 entry point: MODE=grid|train|eval|baseline|smoke, CELLS / SEEDS / REPLICATES / SKIP
experiments/03_build_cells_v3.py   (B,k) grid + replicate cells + label-coverage check (run; zero new queries)
experiments/04_extract_gt.py       GT for the pool pixels (for the † arms only), validated against the teacher labels
experiments/05_import_legacy.py    maps v2 (N,k) results onto v3 cell names into results/v2_legacy/ — comparison only, never the grid
experiments/06_cell_stats.py       per-cell distinct images (dedup criterion), scene labels, domain mix → tables/cell_stats.md
experiments/08_align_pool_coords.py sets pool pixel coordinates to the coordinates the teacher labelled (fixed 41,096 mixed rows)
experiments/20_train_student.py    LoRA training: soft / hard / gtsoft / gthard (+ gt, wsoft), --seed, --decimals {1,2}
experiments/21_eval_student.py     per-cell evaluation (full, or --greedy_only for seeds and replicates)
experiments/31_grid.py, 32_decide.py   per-pool tables and the pre-registered decision rule (--floor = noise floor)
experiments/33_noise_floor.py      B = 400 spread over subsets and seeds → noise_floor_<pool>.json
experiments/34_baselines.py        teacher row (from recorded teacher outputs in ref/) + zero-shot row
pools/<pool>/                      pool.jsonl (v5 order), teacher_labels.parquet (+ gt), rows_B*_k*.parquet, rep_*.parquet, cells_v3.json, gt_coverage.md
tables/                            committed copies of the tables behind the numbers below
third_party/DepthLM_Official/      official curation/metric utilities (FAIR NC — see LICENSE, NOTICE)
paper/design_v3_corrections.md     the C-1..C-4 design decisions with evidence (Korean); later amendments in NOTES.md V-1..V-12
```

## Status

2026-09-30: implementation complete and smoke-tested on GPU (every loss, greedy/full evaluation, baseline mode, the
full `run.sh` pipeline). Teacher rows filled for all four evaluation sets. The student now uses the teacher's
official input/output format and correct image positions (NOTES V-8, V-9), so **no v2 student result enters the
grid**; the six v2 indoor soft cells are kept only as a comparison table below. Next: H200 submissions in the order
under [Order of work](#order-of-work).

## Results

Every cell is `δ1 / AbsRel` on the pool's own evaluation sets, filled in from each cell's evaluation file as it
finishes; a dash means not evaluated yet. Numbers come only from evaluation files — this repository's H200 runs
and the teacher's recorded outputs in `ref/` (pre-registration: the decision rule of `32_decide.py` is applied per
budget row; equal-budget claims must also exceed the B = 400 noise floor below). **`soft` and `hard` are both pseudo-label losses (`Loss_pseudo`)** — hard is CE on the
teacher's answer, soft is KL to the teacher's digit distribution; neither uses ground truth.
**The † rows are `Loss_gt + Loss_pseudo`**, trained alongside the pure arms in the same cells, and the pseudo term
keeps its two forms: `gt+hard` = (1−λ)·CE(teacher answer) + λ·CE(GT), `gt+soft` = (1−λ)·KL(teacher distribution)
+ λ·CE(GT), with **λ = 0.1 fixed in advance** — classic KD found the best results with a considerably lower weight
on the true-label term [arXiv:1503.02531]; the paper gives no universal value, so the common instantiation 0.1 is
pre-registered, and the weighted average keeps the effective learning rate comparable to the pure arms. † marks
budget-external information (GT pixels on top of the same teacher labels): reported in place, but **excluded from
the equal-budget ranking** (C-3b). GT is the Euclidean distance to the camera, as in the official DepthLM curation and our evaluation sets (the depth maps give z; the pool intrinsics convert it), truncated to the first decimal like the teacher labels. GT covers
84 % of the indoor rows, 45 % of mixed and **only 4.3 % of driving** (raw KITTI frames have no GT;
`pools/<pool>/gt_coverage.md`), so the driving † arms differ from the pure arms on few rows.
The `teacher` row is DepthLM 12B on the same pixels with the same midpoint decoding, computed from its recorded
outputs in `ref/` (`34_baselines.py`); like the students it is scored against Euclidean distance to the camera.
With z-depth, the paper's convention, the teacher gets DDAD 0.676 and nuScenes 0.802 (paper: 0.819).
`student zero-shot` is the untrained student. Both are baselines, not cells, and carry no loss condition. **Every evaluation pixel is scored** — an unparsable or `0.0` answer counts as the answer 0.05 m (a miss) — for students, teacher and zero-shot alike. Scene clusters for DDAD and nuScenes come from `ref/scenes_*.parquet` (built from the evaluation pack sent to H200), never from the local disk, and an unknown image stops the analysis.
All runs execute on H200; pools are filled in the order indoor → driving → mixed. The B = 25,600 extension is
dropped from the grid for now (re-added later by the fixed rule if needed).

### Indoor pool — evaluated on iBims-1, NYUv2

| Budget | N | k | Loss | iBims-1 | NYUv2 |
|---:|---:|---:|:--|:--|:--|
| **400** | 400 | 1 | soft | 0.324 / 0.340 | 0.373 / 0.297 |
| | | | hard | — | — |
| | | | gt+soft † | — | — |
| | | | gt+hard † | — | — |
| | 100 | 4 | soft | 0.315 / 0.341 | 0.388 / 0.283 |
| | | | hard | — | — |
| | | | gt+soft † | — | — |
| | | | gt+hard † | — | — |
| | 25 | 16 | soft | 0.365 / 0.322 | 0.467 / 0.265 |
| | | | hard | — | — |
| | | | gt+soft † | — | — |
| | | | gt+hard † | — | — |
| **1,600** | 1600 | 1 | soft | — | — |
| | | | hard | — | — |
| | | | gt+soft † | — | — |
| | | | gt+hard † | — | — |
| | 400 | 4 | soft | — | — |
| | | | hard | — | — |
| | | | gt+soft † | — | — |
| | | | gt+hard † | — | — |
| | 100 | 16 | soft | — | — |
| | | | hard | — | — |
| | | | gt+soft † | — | — |
| | | | gt+hard † | — | — |
| **6,400** | 6400 | 1 | soft | — | — |
| | | | hard | — | — |
| | | | gt+soft † | — | — |
| | | | gt+hard † | — | — |
| | 1600 | 4 | soft | — | — |
| | | | hard | — | — |
| | | | gt+soft † | — | — |
| | | | gt+hard † | — | — |
| | 400 | 16 | soft | — | — |
| | | | hard | — | — |
| | | | gt+soft † | — | — |
| | | | gt+hard † | — | — |
| | | | teacher | 0.809 / 0.143 | 0.881 / 0.125 |
| | | | student zero-shot | 0.054 / 0.592 | 0.060 / 0.566 |

### Driving pool — evaluated on DDAD, nuScenes (mini)

| Budget | N | k | Loss | DDAD | nuScenes |
|---:|---:|---:|:--|:--|:--|
| **400** | 400 | 1 | soft | — | — |
| | | | hard | — | — |
| | | | gt+soft † | — | — |
| | | | gt+hard † | — | — |
| | 100 | 4 | soft | — | — |
| | | | hard | — | — |
| | | | gt+soft † | — | — |
| | | | gt+hard † | — | — |
| | 25 | 16 | soft | — | — |
| | | | hard | — | — |
| | | | gt+soft † | — | — |
| | | | gt+hard † | — | — |
| **1,600** | 1600 | 1 | soft | — | — |
| | | | hard | — | — |
| | | | gt+soft † | — | — |
| | | | gt+hard † | — | — |
| | 400 | 4 | soft | — | — |
| | | | hard | — | — |
| | | | gt+soft † | — | — |
| | | | gt+hard † | — | — |
| | 100 | 16 | soft | — | — |
| | | | hard | — | — |
| | | | gt+soft † | — | — |
| | | | gt+hard † | — | — |
| **6,400** | 6400 | 1 | soft | — | — |
| | | | hard | — | — |
| | | | gt+soft † | — | — |
| | | | gt+hard † | — | — |
| | 1600 | 4 | soft | — | — |
| | | | hard | — | — |
| | | | gt+soft † | — | — |
| | | | gt+hard † | — | — |
| | 400 | 16 | soft | — | — |
| | | | hard | — | — |
| | | | gt+soft † | — | — |
| | | | gt+hard † | — | — |
| | | | teacher | 0.653 / 0.240 | 0.728 / 0.601 |
| | | | student zero-shot | 0.114 / 2.662 | 0.112 / 5.428 |

### Mixed pool (indoor 50 / driving 50) — evaluated on all four sets

| Budget | N | k | Loss | iBims-1 | NYUv2 | DDAD | nuScenes |
|---:|---:|---:|:--|:--|:--|:--|:--|
| **400** | 400 | 1 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
| | | | gt+soft † | — | — | — | — |
| | | | gt+hard † | — | — | — | — |
| | 100 | 4 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
| | | | gt+soft † | — | — | — | — |
| | | | gt+hard † | — | — | — | — |
| | 25 | 16 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
| | | | gt+soft † | — | — | — | — |
| | | | gt+hard † | — | — | — | — |
| **1,600** | 1600 | 1 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
| | | | gt+soft † | — | — | — | — |
| | | | gt+hard † | — | — | — | — |
| | 400 | 4 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
| | | | gt+soft † | — | — | — | — |
| | | | gt+hard † | — | — | — | — |
| | 100 | 16 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
| | | | gt+soft † | — | — | — | — |
| | | | gt+hard † | — | — | — | — |
| **6,400** | 6400 | 1 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
| | | | gt+soft † | — | — | — | — |
| | | | gt+hard † | — | — | — | — |
| | 1600 | 4 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
| | | | gt+soft † | — | — | — | — |
| | | | gt+hard † | — | — | — | — |
| | 400 | 16 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
| | | | gt+soft † | — | — | — | — |
| | | | gt+hard † | — | — | — | — |
| | | | teacher | 0.809 / 0.143 | 0.881 / 0.125 | 0.653 / 0.240 | 0.728 / 0.601 |
| | | | student zero-shot | 0.054 / 0.592 | 0.060 / 0.566 | 0.114 / 2.662 | 0.112 / 5.428 |

### Noise floor — B = 400 (δ1 spread per set, `33_noise_floor.py`)

Two kinds of repeat, both inside the existing labels. **Subset**: the training subset is re-drawn (image blocks
b0–b3 for N = 25 / N = 100, pixel indices p0–p3 for N = 400; b0/p0 are the main cells themselves, so they are not
re-run). **Seed**: the main cell is re-run with seeds 1 and 2, which change initialization, dropout and data order.
The spread is max − min of δ1 within an arm, and the floor per set is the largest spread. An equal-budget difference
must exceed it before it counts as a claim (`32_decide.py --floor`).

| Pool | Arm | Subset spread (4 draws) | Seed spread (3 seeds) | Loss |
|:--|:--|:--|:--|:--|
| indoor | N=25 k=16 | — | — | soft / hard |
| indoor | N=100 k=4 | — | — | soft / hard |
| indoor | N=400 k=1 | — | — | soft / hard |
| driving | N=25 k=16 | — | — | soft / hard |
| driving | N=100 k=4 | — | — | soft / hard |
| driving | N=400 k=1 | — | — | soft / hard |
| mixed | N=25 k=16 | — | — | soft / hard |
| mixed | N=100 k=4 | — | — | soft / hard |
| mixed | N=400 k=1 | — | — | soft / hard |

### Side experiments (each gated, reported separately from the main grid)

| Experiment | Condition | Cell / subset | Result |
|:--|:--|:--|:--|
| W1 pilot (C-3a) | value-space W1 vs soft KL, same cell | to be pre-registered before running | — |
| GT-only reference (C-3b) | CE on GT alone, no teacher labels | grid optimum cell per pool (TBD) | — |
| Label decimals (C-3c) | hard, 1 vs 2 decimals, same subset | fixed subset, re-labeled to 2 decimals | — |
| Recipe-mix (C-4d) | uniform mixed vs per-domain optimal recipes | gate: ≥1 claim in a domain grid | — |

### v2 pipeline, kept for comparison only — indoor soft, seed 0, same pixels

The v2 students were trained with a different question (`… Answer with only a number, for example 2.35.`), answered
the bare number, and — because the processor's `mm_token_type_ids` was not passed during training — saw 1-D instead
of 3-D image positions in training while being evaluated with 3-D positions (NOTES V-8, V-9). These six cells are the
same rows, labels and seed as the v3 cells, so once the v3 cell exists the difference measures **both fixes together**
(it cannot separate them). v2 answered "2.3" on 27–34 % of the evaluation pixels at B ≤ 1,600, against 7 % for the
teacher on the same pixels (`tables/v2_legacy/`).

| Cell | v2 iBims-1 | v2 NYUv2 | v3 iBims-1 | v3 NYUv2 |
|:--|:--|:--|:--|:--|
| B400_k1 | 0.322 / 0.364 | 0.403 / 0.307 | 0.324 / 0.340 | 0.373 / 0.297 |
| B1600_k1 | 0.461 / 0.297 | 0.539 / 0.252 | — | — |
| B1600_k4 | 0.467 / 0.293 | 0.564 / 0.246 | — | — |
| B6400_k1 | 0.602 / 0.216 | 0.724 / 0.181 | — | — |
| B6400_k4 | 0.596 / 0.218 | 0.730 / 0.180 | — | — |
| B6400_k16 | 0.600 / 0.217 | 0.703 / 0.189 | — | — |

## Order of work

All runs execute on H200: one job at a time on a whole GPU (quota `7`), submitted as a GitHub issue whose
command is one line. Settings go either as `KEY=value` arguments (lists comma-separated) or as environment
variables. The driving pool is called `outdoor` in commands.

**Evaluation (pre-registered).** Everything is evaluated on the large sets (`ref/dist_*`, all driving pixels). The
small pixel sets are not used: at ~300 px their sampling noise (~±0.03 δ1) is as large as the allocation effects
being judged (≤ 0.03 in D-27). The saving comes from *what* is computed: main cells (seed 0) get the full evaluation
including the uncertainty tree, while extra seeds and subset repeats get greedy-only evaluation (≈ 6× cheaper), which
is all a spread needs.

**The v2 job still running on H200** (old repository, `grid indoor hard`) produces v2 results, which do not enter the
grid. Stop it if the slot is needed; if it is nearly done, letting it finish only adds hard cells to the comparison
table above.

**Indoor submissions, in order** (rough estimates from the v2 H200 throughput; the † arms train about 2× slower per
step on indoor because 84 % of the rows carry a second, GT forward pass):

| # | One-line command | Fills | Est. |
|---:|:--|:--|--:|
| 1 | `bash run.sh smoke` | checks the new checkout on H200 | 0.5 h |
| 2 | `bash run.sh baseline mixed` | zero-shot row of all three tables (four sets, greedy) | 1 h |
| 3 | `bash run.sh grid indoor soft CELLS=B400_k1,B400_k4,B400_k16 SEEDS=0,1,2 REPLICATES=1` | B = 400 soft row + 2 extra seeds + 9 subset repeats (18 units) | 5–8 h |
| 4 | same as 3 with `hard` | B = 400 hard row; with 3, the indoor noise floor | 5–8 h |
| 5 | `bash run.sh grid indoor soft CELLS=B1600_k1,B1600_k4,B1600_k16` | B = 1,600 soft row | 3–4 h |
| 6 | same as 5 with `hard` | B = 1,600 hard row | 3–4 h |
| 7 | `bash run.sh grid indoor soft CELLS=B6400_k1,B6400_k4,B6400_k16` | B = 6,400 soft row | 4–6 h |
| 8 | same as 7 with `hard` | B = 6,400 hard row | 4–6 h |
| 9–10 | `bash run.sh grid indoor gtsoft CELLS=B400_k1,B400_k4,B400_k16`, then `gthard` | † rows, B = 400 | 3–4 h |
| 11–12 | as 9–10 with `CELLS=B1600_k1,B1600_k4,B1600_k16` | † rows, B = 1,600 | 3–5 h |
| 13–14 | as 9–10 with `CELLS=B6400_k1,B6400_k4,B6400_k16` | † rows, B = 6,400 | 6–9 h |

A cell counts as trained only if its adapter folder has `DONE`, written after the last step; a job that dies mid-training leaves no `DONE`, and that cell is neither evaluated nor counted. After each zip: fill its rows. After the indoor jobs: `33_noise_floor.py --pool indoor`, then
`32_decide.py --pool indoor --floor results/tables/noise_floor_indoor.json` gives the indoor verdict.

**Then driving (`outdoor`), then mixed**, the same fourteen-job pattern (after the one-time smoke and baseline).

**Last, the side experiments**, each after what it depends on: W1 pilot (needs its cell's KL result), 1-vs-2-decimal
ablation (needs a 2-decimal re-label of a fixed subset), GT-only reference (needs the grid optimum), recipe-mix (needs
at least one domain claim, C-4d).
