"""잡음 바닥 집계 (V-2·V-3·V-7) — B=400 행의 두 종류 반복에서 세트별 δ1 산포를 계산한다.
  부분집합 반복: 같은 arm 의 학습 부분집합 재추출 (이미지 블록 b0..b3 / 픽셀 인덱스 p0..p3).
                b0·p0 은 본 셀과 학습 행이 동일하므로 따로 돌리지 않고 본 셀(시드 0) 결과를 쓴다.
  시드 반복:    같은 본 셀을 시드 0·1·2 로 (초기화·드롭아웃·데이터 순서가 모두 바뀜).
산포 = 한 arm 안 반복들의 δ1 최대−최소. 바닥(세트별) = 모든 arm·반복 종류·손실에 대한 산포의 최댓값.
본 셀은 전체 평가, 반복은 greedy 전용이지만 δ1 은 둘 다 같은 greedy 복호값(pred_mid)에서 나온다.
출력: <out_root>/tables/noise_floor_<pool>.md + noise_floor_<pool>.json ({세트: 바닥} — 32_decide.py --floor 입력)
사용: python experiments/33_noise_floor.py --pool indoor [--root <OUT_ROOT>] [--conds soft hard] [--focal 750]
"""
import argparse, glob, json, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBSET = {"rep_B400_N25_k16": ("B400_k16", ["b1", "b2", "b3"], "b0"),   # arm → (본 셀, 반복 셀 접미사, 본 셀이 대신하는 반복)
          "rep_B400_N100_k4": ("B400_k4", ["b1", "b2", "b3"], "b0"),
          "rep_B400_N400_k1": ("B400_k1", ["p1", "p2", "p3"], "p0")}
SEEDED = {"B400_k1": ["s1", "s2"], "B400_k4": ["s1", "s2"], "B400_k16": ["s1", "s2"]}

def load(root, cond, tag, pool, focal):   # 한 파일(_large.parquet) 또는 데이터셋별 파일(_large__<ds>.parquet)
    fs = glob.glob(os.path.join(root, "eval", f"eval_{cond}_{tag}_{pool}_f{focal}_large.parquet")) + \
         glob.glob(os.path.join(root, "eval", f"eval_{cond}_{tag}_{pool}_f{focal}_large__*.parquet"))
    if not fs: return None
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).drop_duplicates(["dataset", "image_id", "pixel_index"])

def d1_by_set(d):
    out = {}
    for ds, g in d.groupby("dataset"):   # 모든 픽셀: 파싱 실패·"0.0" = 0.05 m 답 (V-12)
        p = (g["pred_mid"] if "pred_mid" in g else g["pred"] + 0.05).fillna(0.05).to_numpy(float); t = g["gt"].to_numpy(float)
        out[ds] = (float((np.maximum(p / t, t / p) < 1.25).mean()), len(g))
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pool", required=True); ap.add_argument("--root", default=os.environ.get("OUT_ROOT", "results"))
    ap.add_argument("--conds", nargs="+", default=["soft", "hard"]); ap.add_argument("--focal", default="750"); args = ap.parse_args()
    root = os.path.expanduser(args.root); root = root if os.path.isabs(root) else os.path.join(ROOT, root)
    rows = []
    for cond in args.conds:
        groups = [("부분집합", arm, [(main_rep, main)] + [(r, f"{arm}_{r}") for r in reps]) for arm, (main, reps, main_rep) in SUBSET.items()] + \
                 [("시드", main, [("s0", main)] + [(s, f"{main}_{s}") for s in seeds]) for main, seeds in SEEDED.items()]
        for kind, arm, members in groups:
            for rep, tag in members:
                d = load(root, cond, tag, args.pool, args.focal)
                if d is None: continue
                for ds, (v, n) in d1_by_set(d).items(): rows.append({"cond": cond, "종류": kind, "arm": arm, "rep": rep, "dataset": ds, "δ1": v, "n": n})
    if not rows: print("반복 평가 결과 없음 —", root); return
    df = pd.DataFrame(rows)
    sp = df.groupby(["cond", "종류", "arm", "dataset"])["δ1"].agg(["min", "max", "count"]).reset_index()
    sp["spread"] = sp["max"] - sp["min"]; sp = sp[sp["count"] >= 2]
    if not len(sp): print("반복이 2개 이상인 arm 이 아직 없음"); return
    floor = {ds: round(float(g["spread"].max()), 4) for ds, g in sp.groupby("dataset")}
    os.makedirs(os.path.join(root, "tables"), exist_ok=True)
    json.dump(floor, open(os.path.join(root, "tables", f"noise_floor_{args.pool}.json"), "w"), indent=1)
    md = [f"# 잡음 바닥 — {args.pool} B=400 (V-2·V-7)", "",
          "산포 = 같은 arm 반복들의 δ1 최대−최소 (대형 세트). 부분집합 반복 = 학습 부분집합 재추출(b0/p0 = 본 셀), 시드 반복 = 시드 0·1·2. "
          "바닥 = 전체 최댓값 — 등예산 '주장'은 |Δδ1| 이 이 값을 넘어야 한다.", "",
          sp.round(4).to_markdown(index=False), "", f"**바닥(세트별)**: {floor}", ""]
    open(os.path.join(root, "tables", f"noise_floor_{args.pool}.md"), "w").write("\n".join(md))
    print(sp.round(4).to_string(index=False)); print("바닥:", floor)
    print(f"→ 32_decide.py --pool {args.pool} --floor {os.path.join(root, 'tables', f'noise_floor_{args.pool}.json')}")

if __name__ == "__main__":
    main()
