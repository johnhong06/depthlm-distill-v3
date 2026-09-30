"""교사(DepthLM 12B) 기준선 — 새 평가 세트(KITTI-HO, DDAD)의 모든 질의 픽셀에서 greedy 답을 모아 ref/dist_<name>.parquet 를 만든다.

기존 세트의 ref/dist_*.parquet 는 옛 프로젝트의 11_collect_dist.py(greedy + 자릿수 트리)가 만들었다. 새 세트는 기준선 δ1·AbsRel 만 필요해 greedy 만 돈다.
dist_ok = greedy 답이 숫자로 파싱됨 (학생 평가가 이 행만 쓴다). 교사는 소수 둘째 자리로 답하므로 채점은 +0.005 중간값 (README 기준선 절과 같은 규칙).
GT 는 유클리드 거리(ground_truth_depth)와 z 깊이(ground_truth_z, 교사 논문값 비교용)를 함께 저장한다. 재개 가능(25 픽셀마다 저장).
사용: TEACHER_MODEL=$DATA_ROOT/models/DepthLM python experiments/12_teacher_eval.py --config configs/kitti_ho.yaml
"""
from __future__ import annotations
import argparse, json, os, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, pandas as pd, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from depthlm_uncertainty.depthlm_data import DepthLMJsonl, build_problem_prompt, draw_marker
from depthlm_uncertainty.inference import DepthLMRunner
from depthlm_uncertainty.metrics import parse_depth_answer
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def resolve(p): p = os.path.expandvars(os.path.expanduser(p)); return p if os.path.isabs(p) else os.path.join(ROOT, p)

def score(pred, gt):
    p = pred + 0.005; return float((np.maximum(p / gt, gt / p) < 1.25).mean()), float((np.abs(p - gt) / gt).mean())

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    cfg = yaml.safe_load(open(resolve(a.config))); name = cfg["dataset"]; jl = resolve(cfg["jsonl"])
    ds = DepthLMJsonl(jl, os.path.expandvars(cfg["image_folder"]), name)
    recs = [json.loads(l) for l in open(jl)]; zmap = {(r["image"], j): z for r in recs for j, z in enumerate(r.get("depth_z", []))}
    out = resolve(f"ref/dist_{name}.parquet"); rows = pd.read_parquet(out).to_dict("records") if os.path.exists(out) else []
    done = {(r["image_id"], r["pixel_index"]) for r in rows}
    todo = [(i, s) for i in range(len(ds)) for j in range(ds.num_pixels(i)) if (s := ds.get_sample(i, j)) is not None and (s.image_id, j) not in done]
    if a.limit: todo = todo[: a.limit]
    print(f"{name}: 완료 {len(done)}, 남은 {len(todo)} 픽셀", flush=True)
    if todo:
        runner = DepthLMRunner(os.environ.get("TEACHER_MODEL", cfg["model_path"]), max_new_tokens=cfg.get("max_new_tokens", 128)); problem = build_problem_prompt(); t0 = time.time()
        for n, (i, s) in enumerate(todo, 1):
            g = runner.generate_greedy([draw_marker(ds.get_context(i).base_image, *s.pixel_xy)], problem)[0]; v = parse_depth_answer(g.text)
            rows.append({"image_id": s.image_id, "dataset": name, "pixel_index": s.pixel_index, "pixel_x": s.pixel_xy[0], "pixel_y": s.pixel_xy[1],
                         "ground_truth_depth": s.depth_gt, "ground_truth_z": zmap.get((s.image_id, s.pixel_index), np.nan),
                         "greedy": v, "greedy_text": g.text, "dist_ok": v is not None})
            if n % 25 == 0 or n == len(todo):
                pd.DataFrame(rows).to_parquet(out + ".tmp", index=False); os.replace(out + ".tmp", out)
                print(f"  {n}/{len(todo)}  {(time.time() - t0) / n:.2f} s/px", flush=True)
    d = pd.DataFrame(rows); ok = d[d.dist_ok == True]
    print(f"{name}: {len(ok)}/{len(d)} 파싱  δ1/AbsRel (유클리드) {score(ok.greedy.values, ok.ground_truth_depth.values)}  (z) {score(ok.greedy.values, ok.ground_truth_z.values)}  → {out}", flush=True)

if __name__ == "__main__":
    main()
