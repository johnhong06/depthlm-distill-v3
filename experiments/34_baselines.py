"""베이스라인 행 (V-7) — 격자 표의 teacher / student zero-shot 두 줄.
teacher : ref/dist_<ds>.parquet 의 교사(DepthLM 12B) greedy 기록에서 계산 — GPU 불필요, 학생 평가와 같은 픽셀·같은 규약.
          픽셀 = 학생 대형 평가와 동일 (iBims-1·NYUv2 는 dist_ok 행, DDAD·nuScenes 는 jsonl 전 픽셀 = ref 전 행).
          값 = 첫째 자리 절사 + 0.05 (학생과 같은 중간값 복호). GT = 카메라 중심까지 유클리드 거리 (평가 jsonl 의 depth 와 같은 정의).
          주행 세트는 논문 비교용 z 깊이 δ1 도 따로 적는다 (논문은 z 깊이).
zero-shot: 학습 안 한 학생의 greedy 평가 <root>/eval/eval_zeroshot_f750_large__<ds>.parquet (run.sh baseline 이 만든다), 있을 때만.
출력: <root>/tables/zeroshot_baselines.md
사용: python experiments/34_baselines.py [--root results] [--datasets ibims1,nyuv2,ddad,nuscenes]
"""
import argparse, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = {"ibims1": "iBims-1", "nyuv2": "NYUv2", "ddad": "DDAD", "nuscenes": "nuScenes"}

def score(p, g):
    p, g = np.asarray(p, float), np.asarray(g, float); ok = np.isfinite(p) & (p > 0) & (g > 0); p, g = p[ok], g[ok]
    return float((np.maximum(p / g, g / p) < 1.25).mean()), float(np.mean(np.abs(p - g) / g)), int(ok.sum())

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=os.environ.get("OUT_ROOT", "results")); ap.add_argument("--datasets", default=",".join(NAMES)); a = ap.parse_args()
    root = os.path.expanduser(a.root); root = root if os.path.isabs(root) else os.path.join(ROOT, root)
    teacher, zs, zdepth = {}, {}, {}
    for ds in a.datasets.split(","):
        r = pd.read_parquet(os.path.join(ROOT, "ref", f"dist_{ds}.parquet"))
        if ds in ("ibims1", "nyuv2"): r = r[r.dist_ok == True]
        t = np.floor(r["greedy"].astype(float).to_numpy() * 10 + 1e-4) / 10 + 0.05   # 20_train 의 trunc1 과 같은 절사
        teacher[ds] = score(t, r["ground_truth_depth"])
        if "ground_truth_z" in r: zdepth[ds] = score(t, r["ground_truth_z"])
        f = os.path.join(root, "eval", f"eval_zeroshot_f750_large__{ds}.parquet")
        if os.path.exists(f):
            z = pd.read_parquet(f); zs[ds] = score(z["pred_mid"], z["gt"])
    fmt = lambda v: f"{v[0]:.3f} / {v[1]:.3f} (n={v[2]})"
    rows = [{"row": "teacher (DepthLM 12B)", **{NAMES[d]: fmt(v) for d, v in teacher.items()}},
            {"row": "student zero-shot", **{NAMES[d]: (fmt(zs[d]) if d in zs else "—") for d in teacher}}]
    md = ["# 베이스라인 행 (34_baselines.py)", "", "δ1 / AbsRel (n). 중간값 복호, 유클리드 GT. 교사는 ref/dist_* 기록에서 계산.", "",
          pd.DataFrame(rows).to_markdown(index=False), ""]
    if zdepth: md += ["논문 비교용 교사 z 깊이 δ1 / AbsRel: " + ", ".join(f"{NAMES[d]} {fmt(v)}" for d, v in zdepth.items()), ""]
    os.makedirs(os.path.join(root, "tables"), exist_ok=True); open(os.path.join(root, "tables", "zeroshot_baselines.md"), "w").write("\n".join(md))
    print("\n".join(md))

if __name__ == "__main__":
    main()
