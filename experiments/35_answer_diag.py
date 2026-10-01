"""답 분포·스케일 진단 (기술 통계 — 판정에 쓰지 않는다, V-14). 한 풀·손실의 평가 파일마다 세트별로:
  δ1, ρ (Spearman — 깊이 순서만 보는 값), 전체 스케일 편향 b = median log(pred/gt),
  스케일 하나를 보정한 δ1 (pred·e^−b — 같은 실행 안에서 스케일만 맞췄을 때, 진단 전용),
  최빈 답과 비율, "2.3" 비율, 서로 다른 답 수, 예측·GT 중앙값.
채점은 32·33 과 같다: 모든 픽셀, 파싱 실패·"0.0" = 0.05 m 답 (V-12).
출력: <root>/tables/answer_diag_<pool>_<cond>.md
사용: python experiments/35_answer_diag.py --pool indoor --cond soft [--root results]
"""
import argparse, glob, os, re
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pool", required=True); ap.add_argument("--cond", required=True)
    ap.add_argument("--root", default=os.environ.get("OUT_ROOT", "results")); ap.add_argument("--focal", default="750"); a = ap.parse_args()
    root = a.root if os.path.isabs(a.root) else os.path.join(ROOT, a.root)
    pat = re.compile(rf"eval_{a.cond}_(.+)_{a.pool}_f{a.focal}_large(__\w+)?\.parquet$")
    files = {}
    for f in sorted(glob.glob(os.path.join(root, "eval", f"eval_{a.cond}_*_large*.parquet"))):
        m = pat.search(os.path.basename(f))
        if m: files.setdefault(m.group(1), []).append(f)
    rows = []
    for tag, fs in files.items():
        d = pd.concat([pd.read_parquet(f) for f in fs]).drop_duplicates(["dataset", "image_id", "pixel_index"])
        for ds, g in d.groupby("dataset"):
            p = g["pred_mid"].fillna(0.05).to_numpy(float); t = g["gt"].to_numpy(float); ans = g["pred"].round(1); vc = ans.value_counts(normalize=True)
            b = float(np.median(np.log(p / t))); ps = p * np.exp(-b)
            rows.append({"tag": tag, "set": ds, "n": len(g), "δ1": np.mean(np.maximum(p / t, t / p) < 1.25), "ρ": spearmanr(p, t)[0], "bias": b,
                         "δ1 (scale fixed)": np.mean(np.maximum(ps / t, t / ps) < 1.25), "top answer": f"{vc.index[0]:.1f} ({vc.iloc[0]:.0%})",
                         '"2.3"': f"{(ans == 2.3).mean():.0%}", "distinct": ans.nunique(), "pred median": np.median(p), "GT median": np.median(t)})
    if not rows: print("평가 파일 없음 —", root); return
    df = pd.DataFrame(rows).sort_values(["set", "tag"]).round(3)
    md = [f"# 답 분포·스케일 진단 — {a.pool} {a.cond} (`35_answer_diag.py`, 판정에 쓰지 않음)", "",
          "bias = median log(pred/gt) (음수 = 짧게 답함). δ1 (scale fixed) = 실행마다 스케일 하나만 맞춘 δ1 — 진단 전용.", "",
          df.to_markdown(index=False), ""]
    out = os.path.join(root, "tables", f"answer_diag_{a.pool}_{a.cond}.md"); os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(md)); print(df.to_string(index=False)); print(f"→ {out}")

if __name__ == "__main__":
    main()
