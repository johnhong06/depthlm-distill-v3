"""격자 v3 (C-2): (예산 B × 배분 k) 완전 격자 + 저예산 반복 셀. 풀 v5 순서·기존 교사 라벨 그대로 재사용 (새 질의 0 검증).
격자: B ∈ {400, 1600, 6400} × k ∈ {1, 4, 16}, N = B/k (9셀). 더 큰 예산은 고정 규칙("B 에서 N = B/k ≤ 풀 크기인
  k 전부")으로 나중에 추가 가능 — 25,600 확장은 사용자 지시로 삭제 (2026-09-30, V-4).
  같은 행(B 고정) = 등예산 배분 비교, 같은 열(k 고정) = 이미지 한계효용, 대각선(N 동일) = 밀도 한계효용.
반복 셀(잡음 바닥 전용, 본 격자와 분리): B=400 의 세 arm 을 (25,16) 이미지 블록 4개 / (100,4) 블록 4개 / (400,1) 픽셀 인덱스 0..3 으로
  반복 — 전부 기존 라벨(앞 1,600 장 ×16px) 안. 등예산 배분 차이는 반복 산포를 넘을 때만 주장한다 (paper/design_v3_corrections.md C-2).
출력(제자리): pools/<pool>/rows_B{B}_k{k}.parquet, rep_B400_*.parquet, cells_v3.json
사용: python experiments/03_build_cells_v3.py --pool mixed [--dry]
"""
import argparse, json, os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUDGETS, KS = [400, 1600, 6400], [1, 4, 16]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pool", required=True); ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(); P = os.path.join(ROOT, "pools", a.pool)
    recs = [json.loads(l) for l in open(f"{P}/pool.jsonl")]; ids = [r["image"] for r in recs]
    cells = [(B, k) for B in BUDGETS for k in KS]
    have = pd.read_parquet(f"{P}/teacher_labels.parquet")[["image_id", "pixel_index"]]
    hset = set(map(tuple, have.itertuples(index=False)))

    rows, missing = {}, 0
    for B, k in cells:
        N = B // k
        assert N <= len(ids), f"B{B}_k{k}: N={N} > 풀 {len(ids)}"
        df = pd.DataFrame([(ids[i], j) for i in range(N) for j in range(k)], columns=["image_id", "pixel_index"])
        rows[(B, k)] = df; miss = sum((t not in hset) for t in map(tuple, df.itertuples(index=False))); missing += miss
        print(f"B={B:6d} k={k:2d} N={N:5d} | 행 {len(df):6d} | 기존 라벨 밖 {miss}", flush=True)

    reps = {}   # 반복 셀: (tag) -> df. 본 셀과 겹치는 첫 블록/인덱스 0 도 포함해 4반복으로 산포를 잰다
    for b in range(4):
        reps[f"rep_B400_N25_k16_b{b}"] = pd.DataFrame([(ids[i], j) for i in range(25 * b, 25 * (b + 1)) for j in range(16)], columns=["image_id", "pixel_index"])
        reps[f"rep_B400_N100_k4_b{b}"] = pd.DataFrame([(ids[i], j) for i in range(100 * b, 100 * (b + 1)) for j in range(4)], columns=["image_id", "pixel_index"])
        reps[f"rep_B400_N400_k1_p{b}"] = pd.DataFrame([(ids[i], b) for i in range(400)], columns=["image_id", "pixel_index"])
    rmiss = sum(sum((t not in hset) for t in map(tuple, df.itertuples(index=False))) for df in reps.values())
    print(f"반복 셀 {len(reps)}개 | 기존 라벨 밖 {rmiss}", flush=True)
    print(f"합계: 본 셀 {len(cells)}개 — 새로 필요한 교사 질의 {missing + rmiss} px (0 이어야 함)", flush=True)
    if a.dry: return
    assert missing + rmiss == 0, "기존 라벨로 덮이지 않는 픽셀이 있다 — 설계 위반, 저장하지 않음"
    for (B, k), df in rows.items(): df.to_parquet(f"{P}/rows_B{B}_k{k}.parquet", index=False)
    for tag, df in reps.items(): df.to_parquet(f"{P}/{tag}.parquet", index=False)
    json.dump({"version": "v3", "budgets": BUDGETS, "ks": KS,
               "cells": [{"B": B, "k": k, "N": B // k, "rows": f"rows_B{B}_k{k}.parquet", "tag": f"B{B}_k{k}"} for B, k in cells],
               "replicates": sorted(reps), "pool_images": len(ids), "new_label_px": 0},
              open(f"{P}/cells_v3.json", "w"), indent=1, ensure_ascii=False)
    print(f"저장 → {P}/rows_B*_k*.parquet, rep_*.parquet, cells_v3.json", flush=True)

if __name__ == "__main__":
    main()
