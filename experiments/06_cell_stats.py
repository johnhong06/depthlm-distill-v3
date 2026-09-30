"""셀별 기술 통계 (규칙 5, C-1) — 배분 축 "서로 다른 이미지 수"의 근거 표.
각 풀의 격자 N ∈ {25, 100, 400, 1600, 6400} (앞 N 장)마다:
  서로 다른 이미지 = 앞에서부터 훑으며, 이미 남긴 이미지 중 dHash(9×8) 해밍 ≤ 10 이면서 32×32 회색조 상관 > 0.9 인 것이 없는 이미지 수
                    (07_dedup_pool_order.py 와 같은 기준·같은 특징 함수)
  장면 라벨 수·도메인 비율·소스 구성 — 기술 통계일 뿐, 장면 다양성을 주장하는 근거로 쓰지 않는다 (C-1).
출력: tables/cell_stats.md
사용: python experiments/06_cell_stats.py --data_root $DATA_ROOT
"""
import argparse, importlib.util, json, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("dedup", os.path.join(ROOT, "experiments", "07_dedup_pool_order.py"))
dedup = importlib.util.module_from_spec(spec); spec.loader.exec_module(dedup)
NS = [25, 100, 400, 1600, 6400]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data_root", required=True); ap.add_argument("--ham", type=int, default=10); ap.add_argument("--corr", type=float, default=0.9)
    a = ap.parse_args(); rows = []
    for pool in ["indoor", "outdoor", "mixed"]:
        recs = [json.loads(l) for l in open(os.path.join(ROOT, "pools", pool, "pool.jsonl"))]
        F = [dedup.feats(os.path.join(a.data_root, r["image"])) for r in recs]
        H = np.stack([f[0] for f in F]); V = np.stack([f[1] for f in F])
        kept, first_dup = [], {}
        for i in range(len(recs)):   # 앞에서부터 한 번 훑으면 모든 N 의 "앞 N 장 안 서로 다른 수" 가 나온다 (접두사 성질)
            if kept:
                K = np.array(kept); cand = K[np.unpackbits(H[K] ^ H[i], axis=1).sum(1) <= a.ham]
                if len(cand) and (V[cand] @ V[i] / V.shape[1]).max() > a.corr: continue
            kept.append(i)
        ks = np.array(kept)
        for N in NS:
            sub = recs[:N]; dom = pd.Series([r["domain"] for r in sub]).value_counts()
            rows.append({"pool": pool, "N": N, "서로 다른 이미지": int((ks < N).sum()), "비율": f"{100 * (ks < N).sum() / N:.1f} %",
                         "장면 라벨 수": len({r["scene"] for r in sub}),
                         "도메인": ", ".join(f"{d} {100 * c / N:.0f} %" for d, c in dom.items()),
                         "소스 수": len({r["source"] for r in sub})})
        print(pool, "완료", flush=True)
    df = pd.DataFrame(rows)
    cells = "셀과의 대응: N=25 → B400_k16 · N=100 → B400_k4, B1600_k16 · N=400 → B400_k1, B1600_k4, B6400_k16 · N=1600 → B1600_k1, B6400_k4 · N=6400 → B6400_k1"
    md = ["# 셀별 기술 통계 (06_cell_stats.py, 규칙 5)", "",
          f"서로 다른 이미지 = 07_dedup 기준(dHash 해밍 ≤ {a.ham} 이고 32×32 상관 > {a.corr} 인 앞 이미지가 없음). 장면 수·도메인은 기술 통계일 뿐이다.",
          "", cells, "", df.to_markdown(index=False), ""]
    open(os.path.join(ROOT, "tables", "cell_stats.md"), "w").write("\n".join(md)); print("\n".join(md))

if __name__ == "__main__":
    main()
