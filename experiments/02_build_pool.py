"""중첩 격자 풀 생성 (GT 미사용). 입력: outputs/pool_candidates.parquet (01). 출력: outputs/pool/
격자: 이미지 수 N ∈ --images, 이미지당 픽셀 k ∈ --pixels, 계단형 셀 {(N_a, k_b) : a+b < len(images)} (예산 N·k 가 상한을 넘지 않는 조합).
  · 같은 행(N 고정, k 변화)  = 이미지를 고정하고 질의 픽셀만 늘림 → 라벨 밀도의 한계효용 (다양성 교란 없음)
  · 같은 열(k 고정, N 변화)  = 픽셀 수를 고정하고 이미지를 늘림 → 이미지(장면)의 한계효용
  · 같은 예산(N·k 동일)      = 고정 예산에서의 배분 비교 (위 두 한계효용의 차이로 설명됨)
전부 중첩(nested): 이미지 순서 고정, 픽셀 인덱스 0..k-1 공유 → 셀 간 차이는 "얼마나·어떻게 폈는가" 뿐.
출력: pool.jsonl(이미지 × 최대 k 좌표), order.parquet, arms.json(셀 정의), todo_label.parquet(라벨 필요 픽셀), rows_N{N}_k{k}.parquet(셀별 학습 행)"""
import argparse, os, json, sys, zlib
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--images", type=int, nargs="+", default=[400, 1600, 6400], help="격자의 이미지 수 (오름차순)")
ap.add_argument("--pixels", type=int, nargs="+", default=[1, 4, 16], help="격자의 이미지당 픽셀 수 (오름차순)")
ap.add_argument("--cap_default", type=int, default=1, help="장면당 이미지 상한(기본: 캡처=장면인 소스)")
ap.add_argument("--cap", nargs="*", default=["sunrgbd_xtion=5", "nyuv2=3", "kitti2012=2", "kitti_dsel=20", "vkitti2=20", "diode_out=20"])
ap.add_argument("--domain_ratio", nargs="*", default=["proportional"], help="'proportional' = 상한 적용 후 도메인 후보 수 비례(결정적 인터리빙) → 모든 셀이 같은 도메인 구성")
ap.add_argument("--exclude", default="", help="제외할 이미지 경로 목록 파일(누수 검사 결과)"); ap.add_argument("--drop_source", nargs="*", default=[], help="제외할 소스 (G1 교사 신뢰도 미달 등)")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--dry", action="store_true"); ap.add_argument("--out", default="outputs/pool", help="출력 폴더"); ap.add_argument("--max_budget", type=int, default=0, help="0 = 계단형(a+b<len). 양수면 N·k ≤ max_budget 인 모든 셀"); args = ap.parse_args()
caps = {k: int(v) for k, v in (c.split("=") for c in args.cap)}
ratio = None if args.domain_ratio == ["proportional"] else {k: float(v) for k, v in (c.split("=") for c in args.domain_ratio)}
c = pd.read_parquet(os.path.join(ROOT, "outputs/pool_candidates.parquet")); rng = np.random.default_rng(args.seed)
if args.exclude and os.path.exists(args.exclude):
    ex = {l.strip() for l in open(args.exclude) if l.strip()}; n0 = len(c); c = c[~c.path.isin(ex)]; print(f"제외 {n0 - len(c)} 장")
if args.drop_source:
    n0 = len(c); c = c[~c.source.isin(args.drop_source)]; print(f"소스 제외 {args.drop_source}: {n0 - len(c)} 장")
c = c.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
c["rank_in_scene"] = c.groupby("scene").cumcount(); c["cap"] = c.source.map(lambda s: caps.get(s, args.cap_default)); c = c[c.rank_in_scene < c.cap].reset_index(drop=True)
print("장면 상한 적용 후:"); print(c.groupby(["domain", "source"]).agg(images=("path", "size"), scenes=("scene", "nunique")).to_string())
def domain_order(g):
    g = g.sample(frac=1.0, random_state=args.seed); g["r"] = g.groupby("scene").cumcount(); return g.sort_values(["r"], kind="stable")
parts = {d: domain_order(g).reset_index(drop=True) for d, g in c.groupby("domain")}; ptr = {d: 0 for d in parts}; order = []
names = list(parts); w = np.array([len(parts[d]) for d in names], float) if ratio is None else np.array([ratio.get(d, 0.0) for d in names])
print("도메인 가중:", {names[k]: round(float(w[k] / w.sum()), 3) for k in range(len(names))}, "(후보 수:", {d: len(parts[d]) for d in names}, ")")
while any(ptr[d] < len(parts[d]) for d in parts):
    avail = np.array([ptr[d] < len(parts[d]) for d in names])
    deficit = np.array([(w[k] / w.sum()) * (len(order) + 1) - ptr[names[k]] for k in range(len(names))]); deficit[~avail] = -np.inf; d = names[int(deficit.argmax())]   # 결정적: 목표 비율 대비 가장 뒤처진 도메인
    order.append(parts[d].iloc[ptr[d]]); ptr[d] += 1
o = pd.DataFrame(order).reset_index(drop=True); o["order"] = np.arange(len(o))
IM, PX = sorted(args.images), sorted(args.pixels); Nmax, kmax = IM[-1], PX[-1]
cells = [(N, k) for a, N in enumerate(IM) for b, k in enumerate(PX) if (N * k <= args.max_budget if args.max_budget else a + b < len(IM))]
if len(o) < Nmax: print(f"!!! 후보 {len(o)} 장 < 최대 이미지 수 {Nmax}"); sys.exit(1)
o = o.head(Nmax).copy()
need = np.zeros(Nmax, dtype=int)                      # 이미지 i 가 라벨링해야 할 픽셀 수 = 그 이미지를 쓰는 셀들의 최대 k
for N, k in cells: need[:N] = np.maximum(need[:N], k)
print(f"\n격자 셀 {len(cells)} 개 (예산 = N·k):")
for N, k in cells:
    g = o.head(N); print(f"  N={N:5d} k={k:2d} → 예산 {N*k:6d} px | 장면 {g.scene.nunique():5d} | 도메인 {(g.domain.value_counts(normalize=True)*100).round(1).to_dict()}")
print(f"라벨 필요 픽셀 합계: {int(need.sum())} px  (이미지 {Nmax} 장, 최대 {kmax} px/장)")
if args.dry: sys.exit(0)
out = os.path.join(ROOT, args.out); os.makedirs(out, exist_ok=True)
for f in os.listdir(out):
    if f.startswith("rows_"): os.remove(os.path.join(out, f))
recs, todo = [], []
for i, r in enumerate(o.itertuples()):
    W, H = int(r.W), int(r.H); mx, my = int(0.05 * W), int(0.05 * H); g = np.random.default_rng(zlib.crc32(r.path.encode()) & 0xFFFFFFFF)
    xs = g.integers(mx, W - mx, kmax); ys = g.integers(my, H - my, kmax)
    recs.append({"image": r.path, "source": r.source, "domain": r.domain, "scene": r.scene, "intrinsics": [float(r.fx), float(r.fy), float(r.cx), float(r.cy), W, H],
                 "pixel_coords": [[int(x), int(y)] for x, y in zip(xs, ys)], "depth": [-1.0] * kmax})
    todo += [(r.path, j) for j in range(int(need[i]))]
with open(os.path.join(out, "pool.jsonl"), "w") as f:
    for rec in recs: f.write(json.dumps(rec) + "\n")
o[["order", "path", "source", "domain", "scene"]].to_parquet(os.path.join(out, "order.parquet"), index=False)
pd.DataFrame(todo, columns=["image_id", "pixel_index"]).drop_duplicates().to_parquet(os.path.join(out, "todo_label.parquet"), index=False)
for N, k in cells:
    pd.DataFrame([(p, j) for p in o.head(N).path for j in range(k)], columns=["image_id", "pixel_index"]).to_parquet(os.path.join(out, f"rows_N{N}_k{k}.parquet"), index=False)
json.dump({"images": IM, "pixels": PX, "cells": [{"N": N, "k": k, "budget": N * k, "rows": f"rows_N{N}_k{k}.parquet", "tag": f"N{N}_k{k}"} for N, k in cells],
           "caps": caps, "seed": args.seed, "n_images": int(Nmax), "n_scenes": int(o.scene.nunique()), "n_label_px": int(need.sum())},
          open(os.path.join(out, "arms.json"), "w"), indent=1, ensure_ascii=False)
print(f"저장: {out}/pool.jsonl ({len(recs)} 장), todo_label {len(set(todo))} px, rows_*.parquet {len(cells)} 개")
