"""풀 픽셀 GT 추출 (V-3) — gt+hard·gt+soft arm 의 L_gt 용. 라벨 parquet 의 gt 열을 제자리에서 채운다.
GT 는 절대 규칙 1 에 따라 본 격자 학습에 쓰지 않는다: † arm 과 채점·점검 전용.
GT = 카메라 중심까지의 **유클리드 거리** √(X²+Y²+Z²) (V-12) — DepthLM 공식 curate(curate_sunRGBD.py·curate_NYU.py 의
euclidean_distances)와 우리 평가 세트의 depth 와 같은 정의. 깊이맵은 z 이므로 풀의 intrinsics 로 X=(u−cx)z/fx, Y=(v−cy)z/fy 를 만든다.
(kitti_dsel 은 크롭이라 주점을 W/2, H/2 로 근사 — 01_count_scenes 의 근사 intrinsics 와 같다.)
소스별 깊이맵(z) 출처·규약:
  sunrgbd_*  : 같은 캡처 폴더의 depth/*.png, SUN RGB-D 툴박스 규약 bitshift(v,-3)|bitshift(v,13) 후 /1000 m, 유효 0.005–25.
               공식 curate_sunRGBD.py 의 /10000 은 툴박스와 1.25배 어긋나며, 추출 검증에서 교사 log(t/g)+0.20 편향으로
               드러나 툴박스 규약을 채택 (NYUv2 는 편향 −0.03 으로 정상 — V-5 검증 기록)
  nyuv2      : nyu_depth_v2_labeled.mat 의 depths[idx] (m)
  kitti_dsel : groundtruth_depth/*.png (…_sync_image_… → …_sync_groundtruth_depth_…), uint16/256 m, 0 = 없음
  kitti2012  : training/disp_occ/<이름>.png (…_10 만 GT 존재), depth = fx·0.54/disp, disp = uint16/256
  kitti2015  : training/disp_occ_0/<이름>.png (…_10 만), 동일 변환 (fx 721.5377)
  kitti_raw  : GT 없음 (velodyne 투영 미구현 — 커버리지 0 으로 보고)
사용: python experiments/04_extract_gt.py --pool indoor --data ~/data [--dry]
"""
import argparse, json, os
import numpy as np, pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = 0.54   # KITTI 스테레오 기선 (m) — 프레임별 calib 대신 표준값 (근사, NOTES V-5)

def depth_map(src, img, data):
    """이미지 경로(pool/…) → 원본 GT 깊이맵 (H,W) [m], 없으면 None."""
    if src.startswith("sunrgbd"):
        d = os.path.join(data, img[len("pool/"):].replace("/image/", "/depth/"))
        d = os.path.dirname(d)
        if not os.path.isdir(d): return None
        fs = [f for f in os.listdir(d) if f.endswith(".png")]
        if not fs: return None
        raw = np.asarray(Image.open(os.path.join(d, fs[0])), np.uint16)
        v = (np.bitwise_or(np.right_shift(raw, 3), np.left_shift(raw, 13)).astype(np.float32)) / 1000.0
        v[(v <= 0.005) | (v >= 25)] = 0.0; return v
    if src == "nyuv2":
        idx = int(os.path.basename(img).split("_")[1].split(".")[0])
        return NYU[idx]
    if src == "kitti_dsel":
        p = os.path.join(data, img[len("pool/"):].replace("/image/", "/groundtruth_depth/").replace("_sync_image_", "_sync_groundtruth_depth_"))
        if not os.path.exists(p): return None
        return np.asarray(Image.open(p), np.float32) / 256.0
    if src in ("kitti2012", "kitti2015"):
        name = os.path.basename(img).replace("kitti_", "")
        if not name.endswith("_10.png"): return None   # GT 는 _10 프레임만
        sub = ("kitti2012/training/disp_occ" if src == "kitti2012" else "kitti2015/training/disp_occ_0")
        p = os.path.join(data, "kitti", sub, name)
        if not os.path.exists(p): return None
        disp = np.asarray(Image.open(p), np.float32) / 256.0
        fx = 707.0912 if src == "kitti2012" else 721.5377
        with np.errstate(divide="ignore"): z = np.where(disp > 0, fx * BASELINE / np.maximum(disp, 1e-6), 0.0)
        return z.astype(np.float32)
    return None   # kitti_raw

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pool", required=True); ap.add_argument("--data", default=os.path.expanduser("~/data")); ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(); P = os.path.join(ROOT, "pools", a.pool); a.data = os.path.expanduser(a.data)
    lab = pd.read_parquet(f"{P}/teacher_labels.parquet")
    INTR = {r["image"]: r["intrinsics"] for r in map(json.loads, open(f"{P}/pool.jsonl"))}   # [fx, fy, cx, cy, W, H] 원본 해상도
    global NYU
    NYU = None
    if (lab.source == "nyuv2").any():
        import h5py
        f = h5py.File(os.path.join(a.data, "nyuv2/nyu_depth_v2_labeled.mat"), "r")
        NYU = {i: np.asarray(f["depths"][i], np.float32).T for i in
               {int(os.path.basename(x).split("_")[1].split(".")[0]) for x in lab[lab.source == "nyuv2"].image_id}}
    gt = lab["gt"].to_numpy(float).copy(); n_img, n_nomap = 0, 0
    for img, g in lab.groupby("image_id", sort=False):
        src = g["source"].iloc[0]; dm = depth_map(src, img, a.data); n_img += 1
        if dm is None: n_nomap += 1; continue
        H, W = dm.shape
        x = g["pixel_x_orig"].to_numpy(float).round().astype(int).clip(0, W - 1)
        y = g["pixel_y_orig"].to_numpy(float).round().astype(int).clip(0, H - 1)
        z = dm[y, x].astype(np.float64); fx, fy, cx, cy = INTR[img][:4]
        gt[g.index] = np.where(z > 0, np.sqrt(((x - cx) * z / fx) ** 2 + ((y - cy) * z / fy) ** 2 + z ** 2), 0.0)   # z → 유클리드 거리
    lab["gt"] = gt
    cov = lab.assign(has=lab["gt"] > 0).groupby("source")["has"].agg(["sum", "count"])
    cov["pct"] = (100 * cov["sum"] / cov["count"]).round(1)
    print(f"{a.pool}: 이미지 {n_img} (깊이맵 없음 {n_nomap}) | GT 픽셀 {int((lab['gt'] > 0).sum())}/{len(lab)}")
    print(cov.to_string())
    if a.dry: return
    lab.to_parquet(f"{P}/teacher_labels.parquet", index=False)
    open(f"{P}/gt_coverage.md", "w").write(f"# GT 커버리지 ({a.pool}, 04_extract_gt.py)\n\n" + cov.to_markdown() + "\n\n규약은 스크립트 docstring 참조. GT 는 † arm 전용 (절대 규칙 1).\n")
    print(f"저장 → {P}/teacher_labels.parquet, gt_coverage.md")

if __name__ == "__main__":
    main()
