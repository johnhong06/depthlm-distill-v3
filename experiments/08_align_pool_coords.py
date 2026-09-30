"""풀 좌표를 교사 라벨 좌표에 맞춘다 (V-10). 학생은 pool.jsonl 의 pixel_coords[j] 픽셀로 학습하고, 라벨은 교사가 라벨링 당시
실제로 본 좌표(pixel_x_orig, pixel_y_orig)의 답이다. 둘이 다르면 학생은 다른 픽셀의 답을 배운다.
mixed 풀: 풀 v4 를 만들 때 모든 이미지의 좌표가 새로 생성됐는데, 풀 v3 좌표로 만든 라벨 41,096 행이 (이미지, 픽셀 번호)가 같다는
이유로 유지됐다 → 이 스크립트가 라벨이 있는 모든 (이미지, j) 의 좌표를 라벨 좌표로 되돌린다. 라벨 없는 j 는 어느 셀도 쓰지 않는다
(03_build_cells_v3.py 가 모든 셀·반복 셀이 라벨 안이라는 것을 검증). indoor·outdoor 는 이미 일치해 바뀌는 것이 없다.
사용: python experiments/08_align_pool_coords.py --pool mixed [--dry]
"""
import argparse, json, os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pool", required=True); ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
    P = os.path.join(ROOT, "pools", a.pool)
    recs = [json.loads(l) for l in open(f"{P}/pool.jsonl")]; by = {r["image"]: r for r in recs}
    lab = pd.read_parquet(f"{P}/teacher_labels.parquet")
    changed = 0
    for r in lab.itertuples():
        xy = [int(r.pixel_x_orig), int(r.pixel_y_orig)]; c = by[r.image_id]["pixel_coords"]
        if c[int(r.pixel_index)] != xy: c[int(r.pixel_index)] = xy; changed += 1
    dup = sum(len(r["pixel_coords"]) - len({tuple(x) for x in r["pixel_coords"]}) for r in recs)   # 한 이미지 안 같은 점
    oob = sum(not (0 <= x < r["intrinsics"][4] and 0 <= y < r["intrinsics"][5]) for r in recs for x, y in r["pixel_coords"])
    print(f"{a.pool}: 라벨 {len(lab)} 행 중 좌표를 라벨 좌표로 바꾼 것 {changed} | 이미지 안 중복 점 {dup} | 이미지 밖 좌표 {oob}")
    if a.dry or not changed: return
    with open(f"{P}/pool.jsonl", "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    print(f"저장 → {P}/pool.jsonl (이미지 순서·이미지 집합은 그대로)")

if __name__ == "__main__":
    main()
