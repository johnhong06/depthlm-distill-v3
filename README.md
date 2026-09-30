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
| Grid | budget B ∈ {400, 1600, 6400} × pixels-per-image k ∈ {1, 4, 16}, N = B/k — 9 nested cells, every budget row is a 3-way equal-budget comparison; extension B = 25,600 with k ∈ {4, 16} (k = 1 exceeds the 6,400-image pool and is reported as missing) |
| Pools | indoor (SUN RGB-D, NYUv2), driving (KITTI), mixed (50/50), 6,400 images each, order v5 (near-duplicates demoted: dHash Hamming ≤ 10 and 32×32 grayscale correlation > 0.9) |
| Labels | DepthLM 12B answers on the pool pixels; the v3 grid and its replicate cells are fully covered by the existing 44,800 labels per pool — **zero new teacher queries** (verified by `experiments/03_build_cells_v3.py`) |
| Ground truth | never used for training in the main grid; a separate, clearly-marked `Loss_gt + Loss_pseudo` arm treats GT as budget-external information |
| Noise floor | the B = 400 arms are replicated over image blocks and pixel indices (free relabelings); equal-budget claims must exceed the replicate spread |
| Evaluation | δ1 (max(pred/gt, gt/pred) < 1.25, identical to the official DepthLM metric) on iBims-1, NYUv2 (indoor), DDAD, nuScenes (driving), ETH3D fully held out; each pool on its own domain only |

Why this design: DepthLM's Finding 4 ("1 labeled pixel per training image … image diversity is more important than
label density", arXiv:2509.25413) rests on a sample-count scaling curve (Fig. 4c), not on a fixed-budget
images-vs-pixels comparison — the paper contains none. We run that comparison under conditions the paper never
tested: fixed budget, noisy pseudo-labels, and a small LoRA student. The allocation axis is deliberately
**distinct images, not "scene diversity"**: scene labels are capture metadata and cannot prove distinctness,
so scene counts are reported as descriptive statistics only.

## Layout

```
experiments/03_build_cells_v3.py   (B,k) grid + replicate cells + label-coverage check (run; zero new queries)
experiments/20_train_student.py    LoRA training: hard / soft / wsoft / gt conditions, --decimals {1,2}
experiments/21_eval_student.py     per-cell evaluation
experiments/31_grid.py, 32_decide.py   result tables and pre-registered decision rule (comparison lists: being updated to v3 rows)
pools/<pool>/                      pool.jsonl (v5 order), teacher_labels.parquet, rows_B*_k*.parquet, rep_*.parquet, cells_v3.json
configs/                           pool and evaluation set configs
third_party/DepthLM_Official/      official curation/metric utilities (FAIR NC — see LICENSE, NOTICE)
paper/design_v3_corrections.md     the C-1..C-4 design decisions with evidence (Korean)
```

## Status

Registered 2026-09-30. Grid rows and replicate cells generated and verified for all three pools; training runs not
started. `run.sh` still targets the old grid and must not be used until updated (see NOTES.md checklist).

## Results

Every cell is `δ1 / AbsRel` on the pool's own evaluation sets, filled in from each cell's evaluation log as it
finishes; a dash means not evaluated yet. Numbers are taken only from logs of this repository (pre-registration:
the decision rule of `32_decide.py` is applied per budget row; equal-budget claims must also exceed the B = 400
replicate spread below). **`soft` and `hard` are both pseudo-label losses (`Loss_pseudo`)** — hard is CE on the
teacher's answer, soft is KL to the teacher's digit distribution; neither uses ground truth.
**`gt+pseudo` († rows) is `Loss_gt + Loss_pseudo`**, trained alongside soft/hard in the same cells: † marks that it
uses GT pixels in addition to the same teacher labels — budget-external information, so it is reported in place but
**excluded from the equal-budget ranking** (C-3b). GT coverage of the pool pixels is extracted beforehand and
reported per pool (the current label files carry no GT yet; extraction is a prerequisite).
The `teacher` row is DepthLM 12B evaluated once on the same pixels with the same midpoint decoding;
`student zero-shot` is the untrained student. Both are baselines, not cells, and carry no loss condition.
All runs execute on H200; pools are filled in the order indoor → driving → mixed, and the B = 25,600 extension
rows are **deferred** (kept empty for now).

### Indoor pool — evaluated on iBims-1, NYUv2

| Budget | N | k | Loss | iBims-1 | NYUv2 |
|---:|---:|---:|:--|:--|:--|
| **400** | 400 | 1 | soft | — | — |
| | | | hard | — | — |
| | | | gt+pseudo † | — | — |
| | 100 | 4 | soft | — | — |
| | | | hard | — | — |
| | | | gt+pseudo † | — | — |
| | 25 | 16 | soft | — | — |
| | | | hard | — | — |
| | | | gt+pseudo † | — | — |
| **1,600** | 1600 | 1 | soft | — | — |
| | | | hard | — | — |
| | | | gt+pseudo † | — | — |
| | 400 | 4 | soft | — | — |
| | | | hard | — | — |
| | | | gt+pseudo † | — | — |
| | 100 | 16 | soft | — | — |
| | | | hard | — | — |
| | | | gt+pseudo † | — | — |
| **6,400** | 6400 | 1 | soft | — | — |
| | | | hard | — | — |
| | | | gt+pseudo † | — | — |
| | 1600 | 4 | soft | — | — |
| | | | hard | — | — |
| | | | gt+pseudo † | — | — |
| | 400 | 16 | soft | — | — |
| | | | hard | — | — |
| | | | gt+pseudo † | — | — |
| **25,600** (deferred) | 6400 | 4 | soft | — | — |
| | | | hard | — | — |
| | | | gt+pseudo † | — | — |
| | 1600 | 16 | soft | — | — |
| | | | hard | — | — |
| | | | gt+pseudo † | — | — |
| | | | teacher | — | — |
| | | | student zero-shot | — | — |

### Driving pool — evaluated on DDAD, nuScenes (mini)

| Budget | N | k | Loss | DDAD | nuScenes |
|---:|---:|---:|:--|:--|:--|
| **400** | 400 | 1 | soft | — | — |
| | | | hard | — | — |
| | | | gt+pseudo † | — | — |
| | 100 | 4 | soft | — | — |
| | | | hard | — | — |
| | | | gt+pseudo † | — | — |
| | 25 | 16 | soft | — | — |
| | | | hard | — | — |
| | | | gt+pseudo † | — | — |
| **1,600** | 1600 | 1 | soft | — | — |
| | | | hard | — | — |
| | | | gt+pseudo † | — | — |
| | 400 | 4 | soft | — | — |
| | | | hard | — | — |
| | | | gt+pseudo † | — | — |
| | 100 | 16 | soft | — | — |
| | | | hard | — | — |
| | | | gt+pseudo † | — | — |
| **6,400** | 6400 | 1 | soft | — | — |
| | | | hard | — | — |
| | | | gt+pseudo † | — | — |
| | 1600 | 4 | soft | — | — |
| | | | hard | — | — |
| | | | gt+pseudo † | — | — |
| | 400 | 16 | soft | — | — |
| | | | hard | — | — |
| | | | gt+pseudo † | — | — |
| **25,600** (deferred) | 6400 | 4 | soft | — | — |
| | | | hard | — | — |
| | | | gt+pseudo † | — | — |
| | 1600 | 16 | soft | — | — |
| | | | hard | — | — |
| | | | gt+pseudo † | — | — |
| | | | teacher | — | — |
| | | | student zero-shot | — | — |

### Mixed pool (indoor 50 / driving 50) — evaluated on all four sets

| Budget | N | k | Loss | iBims-1 | NYUv2 | DDAD | nuScenes |
|---:|---:|---:|:--|:--|:--|:--|:--|
| **400** | 400 | 1 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
| | | | gt+pseudo † | — | — | — | — |
| | 100 | 4 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
| | | | gt+pseudo † | — | — | — | — |
| | 25 | 16 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
| | | | gt+pseudo † | — | — | — | — |
| **1,600** | 1600 | 1 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
| | | | gt+pseudo † | — | — | — | — |
| | 400 | 4 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
| | | | gt+pseudo † | — | — | — | — |
| | 100 | 16 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
| | | | gt+pseudo † | — | — | — | — |
| **6,400** | 6400 | 1 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
| | | | gt+pseudo † | — | — | — | — |
| | 1600 | 4 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
| | | | gt+pseudo † | — | — | — | — |
| | 400 | 16 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
| | | | gt+pseudo † | — | — | — | — |
| **25,600** (deferred) | 6400 | 4 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
| | | | gt+pseudo † | — | — | — | — |
| | 1600 | 16 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
| | | | gt+pseudo † | — | — | — | — |
| | | | teacher | — | — | — | — |
| | | | student zero-shot | — | — | — | — |

### Noise floor — B = 400 replicate cells (δ1 spread across 4 replicates, per set)

Replicates re-draw the training subset inside the existing labels (image blocks for N = 25 / N = 100, pixel indices
for N = 400). The spread (max − min of δ1 over the 4 replicates) is the floor an equal-budget difference must exceed.

| Pool | Arm | Replicates | Loss | Spread per set |
|:--|:--|:--|:--|:--|
| indoor | N=25 k=16 | image blocks ×4 | soft / hard | — |
| indoor | N=100 k=4 | image blocks ×4 | soft / hard | — |
| indoor | N=400 k=1 | pixel indices ×4 | soft / hard | — |
| driving | N=25 k=16 | image blocks ×4 | soft / hard | — |
| driving | N=100 k=4 | image blocks ×4 | soft / hard | — |
| driving | N=400 k=1 | pixel indices ×4 | soft / hard | — |
| mixed | N=25 k=16 | image blocks ×4 | soft / hard | — |
| mixed | N=100 k=4 | image blocks ×4 | soft / hard | — |
| mixed | N=400 k=1 | pixel indices ×4 | soft / hard | — |

### Side experiments (each gated, reported separately from the main grid)

| Experiment | Condition | Cell / subset | Result |
|:--|:--|:--|:--|
| W1 pilot (C-3a) | value-space W1 vs soft KL, same cell | to be pre-registered before running | — |
| GT-only reference (C-3b) | CE on GT alone, no teacher labels | grid optimum cell per pool (TBD) | — |
| Label decimals (C-3c) | hard, 1 vs 2 decimals, same subset | fixed subset, re-labeled to 2 decimals | — |
| Recipe-mix (C-4d) | uniform mixed vs per-domain optimal recipes | gate: ≥1 claim in a domain grid | — |

## Order of work

All experiments run on H200 (one job at a time on a whole GPU, as before: GitHub issue → Jenkins → container).
Pools are filled **indoor → driving → mixed**, and within a pool the cheapest rows go first so the tables fill
fast; the B = 25,600 rows are deferred. A job is one (pool, budget row, loss) with
loss ∈ {soft, hard, gt+pseudo}: nine grid jobs per pool plus one baseline job, each well under a day.

**Evaluation (pre-registered)**: everything is evaluated on the large sets (`ref/dist_*`) — the small pixel sets
are not used, since their per-replicate sampling noise (~±0.03 δ1 at ~300 px) is the same size as the allocation
effects being judged (D-27: ≤ 0.03). The economy comes from *what* is computed, not *where*: main cells (seed 0)
get the full evaluation including the uncertainty tree (CoV), while extra seeds and the 12 replicate cells get
**greedy-only** evaluation (single forward per pixel, no tree — ≈ 6× cheaper), which is all a spread estimate needs.

0. **Prerequisites (local, before any submission)**: (a) update `31_grid.py` / `32_decide.py` comparison lists to
   the v3 budget rows; (b) update `run.sh` to `cells_v3.json` / `rows_B*` / `rep_*`; (c) **extract GT for the pool
   pixels** (the label files currently carry none — SUN RGB-D/NYUv2/KITTI depth maps are on the local disk) and
   report coverage per pool; (d) add a greedy-only flag to `21_eval_student.py`; (e) smoke-test one tiny cell.
1. **Job 0 — baselines (H200, once, ≈ 4–6 h)**: teacher on all four evaluation sets (same pixels, midpoint
   decoding) + zero-shot student. Fills the two baseline rows of every table first.
2. **Indoor pool, cheapest first — submission order** (then the same pattern for driving, then mixed):
   - **A1/A2: B = 400, soft / hard (≈ 10–12 h each)** — 3 arms × 3 seeds + 12 replicate cells
     (training ≈ 1.3 h; main cells full eval, the other 18 checkpoints greedy-only). Fills the first budget row
     and the noise floor together.
   - **A3: B = 400, gt+pseudo (≈ 7 h)** — 3 arms × 1 seed, full eval (reference arm, no replicates).
   - **B1/B2/B3: B = 1,600, soft / hard / gt+pseudo (≈ 7 h each)** — 3 arms, 1 seed.
   - **C1/C2/C3: B = 6,400, soft / hard / gt+pseudo (≈ 9 h each)** — 3 arms, 1 seed.
   Each cell is evaluated inside its job and its table row is filled from the log as soon as the zip arrives.
3. **Decide per pool** (`32_decide.py`) after its jobs: pre-registered rule per budget row (soft/hard only; the
   † rows are excluded from ranking); claims only where the CI excludes zero on all of the pool's sets and the
   difference exceeds the noise floor. Then move to the next pool.
4. **Side experiments** after the main rows they depend on: W1 pilot (needs its cell's KL result), decimals
   ablation (needs the 2-decimal re-label), GT-only reference (needs the grid optimum), recipe-mix (needs ≥ 1
   domain claim).
