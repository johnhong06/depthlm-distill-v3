"""G0 — 풀 후보 이미지 열거 + 장면(스캔·시퀀스) 집계. GT 는 읽지 않는다. 출력: outputs/pool_candidates.parquet, outputs/scene_table.md
장면 정의: SUN RGB-D kinect2/realsense/b3do = 캡처 1장 = 1장면(같은 방의 재촬영은 구분 불가 — 한계로 기록), sun3ddata = 시퀀스 폴더;
NYUv2 = .mat scene 이름; KITTI 2012 = 프레임 쌍(000000_10/_11) 번호; KITTI depth_selection = 드라이브; vKITTI2 = SceneXX; DIODE = scene/scan.
intrinsics: SUN RGB-D 이미지별 3×3, NYUv2 표준, KITTI 2012 calib(707.09), depth_selection 은 근사(707.09, 크롭 — 초점 정규화만 쓰므로 ≤2 % 오차), vKITTI2 725.0087, DIODE [886.81, 927.06, 512, 384]."""
import os, sys, glob, json, re
import numpy as np, pandas as pd, h5py
from PIL import Image
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D = os.path.expanduser("~/data"); rows = []
def add(path, source, domain, scene, fx, fy, cx, cy, W, H): rows.append(dict(path=path, source=source, domain=domain, scene=scene, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H))
# 1) SUN RGB-D (NYUdata 제외)
for sub, tag in [("kv2/kinect2data", "sunrgbd_kv2"), ("xtion/sun3ddata", "sunrgbd_xtion"), ("realsense", "sunrgbd_rs"), ("kv1/b3dodata", "sunrgbd_b3do")]:
    base = os.path.join(D, "sunrgbd/SUNRGBD", sub)
    for ip in glob.glob(os.path.join(base, "**", "intrinsics.txt"), recursive=True):
        d = os.path.dirname(ip)
        if os.path.basename(d) == "fullres": continue
        imgs = sorted(glob.glob(os.path.join(d, "image", "*.jpg")) + glob.glob(os.path.join(d, "image", "*.png")))
        if not imgs: continue
        K = np.loadtxt(ip).reshape(3, 3); W, H = Image.open(imgs[0]).size
        scene = os.path.relpath(d, base).split(os.sep)[0] if tag == "sunrgbd_xtion" else os.path.relpath(d, base)
        add(imgs[0], tag, "indoor", f"{tag}/{scene}", float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), W, H)
# 2) NYUv2 labeled − 평가 200 (v1 풀의 PNG 재사용, 장면 이름은 .mat)
eval_ids = {json.loads(l)["image"] for l in open(os.path.join(D, "nyuv2_depthlm/nyuv2_val.jsonl"))}
f = h5py.File(os.path.join(D, "nyuv2/nyu_depth_v2_labeled.mat"), "r"); sc = f["scenes"]
for idx in range(sc.shape[1]):
    name = f"rgb/nyu_{idx:04d}.png"
    if name in eval_ids: continue
    p = os.path.join(D, "distill_pool", name)
    if not os.path.exists(p): continue
    scene = "".join(chr(c[0]) for c in f[sc[0][idx]][:])
    add(p, "nyuv2", "indoor", f"nyuv2/{scene}", 518.8579, 519.4696, 325.5824, 253.7362, 640, 480)
# 3) KITTI 2012 (v1 풀 PNG), depth_selection (드라이브별), vKITTI2 (v1 풀 JPG)
for l in open(os.path.join(D, "distill_pool/pool.jsonl")):
    r = json.loads(l); fx, fy, cx, cy, W, H = r["intrinsics"]; p = os.path.join(D, "distill_pool", r["image"]); b = os.path.basename(r["image"])
    if r["source"] == "kitti": add(p, "kitti2012", "driving", "kitti2012/" + b.split("_")[1], fx, fy, cx, cy, W, H)
    elif r["source"] == "vkitti2": add(p, "vkitti2", "driving", "vkitti2/" + b.split("_")[1], fx, fy, cx, cy, W, H)
k15 = os.path.join(D, "kitti/kitti2015/training/image_2")   # KITTI 2015: 200 개의 서로 다른 장면 × 2 프레임 (calib zip 미보유 → 표준 P2 근사, 초점 정규화만 사용)
for p in sorted(glob.glob(os.path.join(k15, "*.png"))):
    b = os.path.basename(p); W, H = Image.open(p).size
    add(p, "kitti2015", "driving", "kitti2015/" + b.split("_")[0], 721.5377, 721.5377, 609.5593, 172.854, W, H)
mp = os.path.join(D, "kitti/raw/manifest.parquet")   # KITTI raw 추가 드라이브: 장면 = 드라이브 + 300프레임(30 s) 구간
if os.path.exists(mp):
    for r in pd.read_parquet(mp).itertuples(): add(r.path, "kitti_raw", "driving", f"kitti_raw/{r.drive}/seg{r.frame // 300}", r.fx, r.fy, r.cx, r.cy, int(r.W), int(r.H))
ds = os.path.join(D, "kitti/depth_selection/val_selection_cropped/image")
for p in sorted(glob.glob(os.path.join(ds, "*.png"))):
    m = re.match(r"(\d{4}_\d{2}_\d{2}_drive_\d{4})", os.path.basename(p)); W, H = Image.open(p).size
    add(p, "kitti_dsel", "driving", "kitti_dsel/" + m.group(1), 707.0912, 707.0912, W / 2, H / 2, W, H)   # 근사 intrinsics (크롭)
# 4) DIODE train outdoor (추출돼 있으면)
do = os.path.join(D, "diode/train/outdoor")
if os.path.isdir(do):
    for p in sorted(glob.glob(os.path.join(do, "scene_*", "scan_*", "*.png"))):
        scan = os.path.relpath(os.path.dirname(p), do).replace(os.sep, "/"); add(p, "diode_out", "outdoor_static", "diode_out/" + scan, 886.81, 927.06, 512.0, 384.0, 1024, 768)
else: print("DIODE train/outdoor 아직 없음 — 추출 후 재실행", flush=True)
df = pd.DataFrame(rows); os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True); df.to_parquet(os.path.join(ROOT, "outputs/pool_candidates.parquet"), index=False)
g = df.groupby(["domain", "source"]).agg(images=("path", "size"), scenes=("scene", "nunique")).reset_index(); g["img_per_scene"] = (g.images / g.scenes).round(1)
tot = pd.DataFrame([{"domain": "TOTAL", "source": "", "images": len(df), "scenes": df.scene.nunique(), "img_per_scene": round(len(df) / max(df.scene.nunique(), 1), 1)}])
md = "## G0 풀 후보 장면 집계 (GT 미사용)\n\n" + pd.concat([g, tot]).to_markdown(index=False) + "\n\n장면 정의·intrinsics 출처는 스크립트 docstring 참조. 평가 이미지(NYUv2 200, iBims-1, ETH3D)는 포함되지 않음(NYUv2 는 이름 단위 제외, 나머지는 소스에 없음).\n"
open(os.path.join(ROOT, "outputs/scene_table.md"), "w").write(md); print(md)
