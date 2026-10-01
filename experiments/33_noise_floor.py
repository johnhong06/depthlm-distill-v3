"""잡음 바닥 집계 (V-2·V-3·V-7·V-15) — 반복이 있는 예산 행(B=400·1600)마다 세트별 δ1 산포를 계산한다.
  부분집합 반복: 같은 arm 의 학습 부분집합 재추출 (k=16·4 이미지 블록 b0..b3 / k=1 픽셀 인덱스 p0..p3).
                b0·p0 은 본 셀과 학습 행이 동일하므로 따로 돌리지 않고 본 셀(시드 0) 결과를 쓴다.
  시드 반복:    같은 본 셀을 시드 0·1·2 로 (초기화·드롭아웃·데이터 순서가 모두 바뀜). B=400 에서만 돌린다 — 거기서 6개 arm×세트
                모두 부분집합 산포보다 작았다 (V-15). 결과 파일이 있으면 어느 예산이든 집계한다.
산포 = 한 arm 안 반복들의 δ1 최대−최소. 바닥(예산·세트별) = 그 예산 행의 모든 arm·반복 종류·손실에 대한 산포의 최댓값.
반복을 아직 안 돌린 예산은 바닥이 없다 — 32 는 그 아래 예산의 바닥을 쓴다. 반복이 일부만 있으면 크게 경고한다 (산포가 작게 나온다).
본 셀은 전체 평가, 반복은 greedy 전용이지만 δ1 은 둘 다 같은 greedy 복호값(pred_mid)에서 나온다.
출력: <out_root>/tables/noise_floor_<pool>.md + noise_floor_<pool>.json ({예산: {세트: 바닥}} — 32_decide.py --floor 입력)
사용: python experiments/33_noise_floor.py --pool indoor [--root <OUT_ROOT>] [--conds soft hard] [--focal 750]
"""
import argparse, glob, json, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUDGETS = (400, 1600)   # 반복 셀이 있는 예산 (03_build_cells_v3.py REP_BUDGETS)
SUBSET = {B: {f"rep_B{B}_N{B // k}_k{k}": (f"B{B}_k{k}", [f"{'p' if k == 1 else 'b'}{i}" for i in (1, 2, 3)], "p0" if k == 1 else "b0")
              for k in (16, 4, 1)} for B in BUDGETS}   # arm → (본 셀, 반복 셀 접미사, 본 셀이 대신하는 반복)
SEEDED = {B: {f"B{B}_k{k}": ["s1", "s2"] for k in (1, 4, 16)} for B in BUDGETS}

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
        for B in BUDGETS:
            groups = [("부분집합", arm, [(main_rep, main)] + [(r, f"{arm}_{r}") for r in reps]) for arm, (main, reps, main_rep) in SUBSET[B].items()] + \
                     [("시드", main, [("s0", main)] + [(s, f"{main}_{s}") for s in seeds]) for main, seeds in SEEDED[B].items()]
            for kind, arm, members in groups:
                for rep, tag in members:
                    d = load(root, cond, tag, args.pool, args.focal)
                    if d is None: continue
                    for ds, (v, n) in d1_by_set(d).items(): rows.append({"B": B, "cond": cond, "종류": kind, "arm": arm, "rep": rep, "dataset": ds, "δ1": v, "n": n, "full": len(members)})
    if not rows: print("반복 평가 결과 없음 —", root); return
    df = pd.DataFrame(rows)
    sp = df.groupby(["B", "cond", "종류", "arm", "dataset"]).agg(min=("δ1", "min"), max=("δ1", "max"), count=("δ1", "count"), full=("full", "first")).reset_index()
    sp["spread"] = sp["max"] - sp["min"]
    # 반복을 돌린 예산에서 부분집합 arm 이 4개를 못 채웠거나 시드가 일부만 있으면 산포가 작게 나와 판정이 느슨해진다 → 크게 경고
    ran = set(sp[sp["count"] >= 2].B)
    part = sp[sp.B.isin(ran) & (sp["count"] < sp["full"]) & ((sp["종류"] == "부분집합") | (sp["count"] >= 2))]
    for B in sorted(set(part.B)): print(f"!!! B={B}: 반복 일부만 있음 ({len(part[part.B == B])}개 arm×세트) — 실패한 단위를 다시 돌린 뒤 다시 계산할 것 (지금 바닥은 하한)", flush=True)
    sp = sp[sp["count"] >= 2].drop(columns="full")
    if not len(sp): print("반복이 2개 이상인 arm 이 아직 없음"); return
    floor = {str(B): {ds: round(float(g2["spread"].max()), 4) for ds, g2 in g.groupby("dataset")} for B, g in sp.groupby("B")}
    os.makedirs(os.path.join(root, "tables"), exist_ok=True)
    json.dump(floor, open(os.path.join(root, "tables", f"noise_floor_{args.pool}.json"), "w"), indent=1)
    md = [f"# 잡음 바닥 — {args.pool} (V-2·V-7·V-15)", "",
          "산포 = 같은 arm 반복들의 δ1 최대−최소 (대형 세트). 부분집합 반복 = 학습 부분집합 재추출(b0/p0 = 본 셀), 시드 반복 = 시드 0·1·2 (B=400). "
          "바닥 = 예산 행마다 최댓값 — 그 예산의 등예산 '주장'은 |Δδ1| 이 이 값을 넘어야 한다. B=6400 은 B=1600 바닥을 쓴다 (32_decide).", "",
          sp.round(4).to_markdown(index=False), "", f"**바닥(예산·세트별)**: {floor}", ""]
    open(os.path.join(root, "tables", f"noise_floor_{args.pool}.md"), "w").write("\n".join(md))
    print(sp.round(4).to_string(index=False)); print("바닥:", floor)
    print(f"→ 32_decide.py --pool {args.pool} --floor {os.path.join(root, 'tables', f'noise_floor_{args.pool}.json')}")

if __name__ == "__main__":
    main()
