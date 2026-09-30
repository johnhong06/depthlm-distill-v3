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
