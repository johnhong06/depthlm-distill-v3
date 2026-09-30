"""잡음 바닥 집계 (V-2·V-3) — B=400 반복 셀의 평가에서 세트별 δ1 산포를 계산한다.
반복 = 같은 arm 의 학습 부분집합 재추출 (이미지 블록 b0..b3 / 픽셀 인덱스 p0..p3), greedy 전용 평가(대형 세트).
산포 = 반복 4개의 δ1 최대−최소. 바닥(세트별) = 모든 arm·손실에 대한 산포의 최댓값.
출력: <out_root>/tables/noise_floor_<pool>.md + noise_floor_<pool>.json ({세트: 바닥} — 32_decide.py --floor 입력)
사용: python experiments/33_noise_floor.py --pool indoor [--root <OUT_ROOT>] [--conds soft hard] [--focal 750]
"""
import argparse, glob, json, os, re
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def d1(df):
    d = df[df.pred.notna() & (df.pred > 0)]
    p = (d["pred_mid"] if "pred_mid" in d else d["pred"] + 0.05).to_numpy(float); g = d["gt"].to_numpy(float)
    return float((np.maximum(p / g, g / p) < 1.25).mean()), len(d)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pool", required=True); ap.add_argument("--root", default=os.environ.get("OUT_ROOT", "results"))
    ap.add_argument("--conds", nargs="+", default=["soft", "hard"]); ap.add_argument("--focal", default="750"); args = ap.parse_args()
    root = os.path.expanduser(args.root); root = root if os.path.isabs(root) else os.path.join(ROOT, root)
    rows, floor = [], {}
    for cond in args.conds:
        pat = os.path.join(root, "eval", f"eval_{cond}_rep_B400_*_{args.pool}_f{args.focal}_large*.parquet")
        by = {}   # (arm, 반복) -> [parquet…]  (한 파일 또는 데이터셋별 파일)
        for f in glob.glob(pat):
            m = re.search(rf"eval_{cond}_(rep_B400_N\d+_k\d+)_([bp]\d)_{args.pool}_", os.path.basename(f))
            if m: by.setdefault((m.group(1), m.group(2)), []).append(f)
        for (arm, rep), fs in sorted(by.items()):
            d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).drop_duplicates(["dataset", "image_id", "pixel_index"])
            for ds, g in d.groupby("dataset"):
                v, n = d1(g); rows.append({"cond": cond, "arm": arm, "rep": rep, "dataset": ds, "δ1": v, "n": n})
    if not rows: print("반복 셀 평가 결과 없음 —", root); return
    df = pd.DataFrame(rows)
    sp = df.groupby(["cond", "arm", "dataset"])["δ1"].agg(["min", "max", "count"]).reset_index()
    sp["spread"] = sp["max"] - sp["min"]; sp = sp[sp["count"] >= 2]
    for ds, g in sp.groupby("dataset"): floor[ds] = round(float(g["spread"].max()), 4)
    os.makedirs(os.path.join(root, "tables"), exist_ok=True)
    json.dump(floor, open(os.path.join(root, "tables", f"noise_floor_{args.pool}.json"), "w"), indent=1)
    md = [f"# 잡음 바닥 — {args.pool} B=400 반복 셀 (V-2)", "",
          "산포 = 같은 arm 반복들의 δ1 최대−최소 (greedy 전용, 대형 세트). 바닥 = arm·손실 전체의 최댓값 — 등예산 '주장'은 |Δδ1| 이 이 값을 넘어야 한다.", "",
          sp.round(4).to_markdown(index=False), "", f"**바닥(세트별)**: {floor}", ""]
    open(os.path.join(root, "tables", f"noise_floor_{args.pool}.md"), "w").write("\n".join(md))
    print(sp.round(4).to_string(index=False)); print("바닥:", floor)
    print(f"→ 32_decide.py --pool {args.pool} --floor {os.path.join(root, 'tables', f'noise_floor_{args.pool}.json')}")

if __name__ == "__main__":
    main()
