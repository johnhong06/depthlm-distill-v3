"""옛 격자(v2, N×k) 결과를 v3 태그로 매핑해 들여온다 (V-6). 중첩 설계라 학습 행이 동일한 셀만 매핑한다:
  N400_k1→B400_k1, N1600_k1→B1600_k1, N400_k4→B1600_k4, N6400_k1→B6400_k1, N1600_k4→B6400_k4, N400_k16→B6400_k16
  (같은 접두사 이미지 × 같은 픽셀 인덱스 × 같은 라벨·시드·하이퍼파라미터 — 셀 이름만 다르다. N1600_k16·N6400_k4 는
  삭제된 예산 25,600 이라 제외.)
들여오는 것: eval/eval_<cond>_<old>_<pool>_f<focal>_large*.parquet → <out>/eval/ (v3 태그로 개명),
  checkpoints/<cond>_<old>_<pool>_f<focal>/ → <out>/checkpoints/ (재평가용). 출처·매핑은 <out>/import_legacy_*.md 에 기록.
사용: python experiments/05_import_legacy.py --zip ~/Downloads/results_soft_indoor.zip --pool indoor --cond soft [--out results]
"""
import argparse, hashlib, os, shutil, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP = {"N400_k1": "B400_k1", "N1600_k1": "B1600_k1", "N400_k4": "B1600_k4",
       "N6400_k1": "B6400_k1", "N1600_k4": "B6400_k4", "N400_k16": "B6400_k16"}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--zip", required=True); ap.add_argument("--pool", required=True); ap.add_argument("--cond", required=True)
    ap.add_argument("--focal", default="750"); ap.add_argument("--out", default=os.environ.get("OUT_ROOT", "results")); a = ap.parse_args()
    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out); zp = os.path.expanduser(a.zip)
    zf = zipfile.ZipFile(zp); done, skipped = [], []
    for old, new in MAP.items():
        o_suf, n_suf = f"{a.cond}_{old}_{a.pool}_f{a.focal}", f"{a.cond}_{new}_{a.pool}_f{a.focal}"
        for n in zf.namelist():
            b = os.path.basename(n)
            if n.startswith("eval/") and b.startswith(f"eval_{o_suf}_large"):   # _large.parquet 또는 _large__<ds>.parquet
                dst = os.path.join(out, "eval", b.replace(o_suf, n_suf)); os.makedirs(os.path.dirname(dst), exist_ok=True)
                with zf.open(n) as f, open(dst, "wb") as g: shutil.copyfileobj(f, g)
                done.append(f"{n} → eval/{os.path.basename(dst)}")
            elif n.startswith(f"checkpoints/{o_suf}/") and not n.endswith("/"):
                dst = os.path.join(out, "checkpoints", n_suf, os.path.relpath(n, f"checkpoints/{o_suf}")); os.makedirs(os.path.dirname(dst), exist_ok=True)
                with zf.open(n) as f, open(dst, "wb") as g: shutil.copyfileobj(f, g)
                if b == "adapter_model.safetensors": done.append(f"{n} → checkpoints/{n_suf}/")
    for old in ("N1600_k16", "N6400_k4"):
        if any(f"_{old}_" in n for n in zf.namelist()): skipped.append(old)
    h = hashlib.sha256(open(zp, "rb").read()).hexdigest()[:16]
    md = [f"# 옛 격자 결과 들여옴 — {a.pool} {a.cond} (05_import_legacy.py, V-6)", "",
          f"출처: `{zp}` (sha256 앞 16: {h})", f"매핑: {MAP}", f"제외(삭제된 예산 25,600): {skipped}", "", "들여온 파일:"] + [f"- {d}" for d in done]
    os.makedirs(out, exist_ok=True); open(os.path.join(out, f"import_legacy_{a.pool}_{a.cond}.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md[-len(done):] if done else ["들여온 파일 없음 — zip 구조 확인"])); print(f"→ {out} ({len(done)}건)")

if __name__ == "__main__":
    main()
