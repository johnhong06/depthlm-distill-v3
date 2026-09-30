"""평가 픽셀의 부트스트랩 군집 — 같은 군집의 픽셀은 함께 다시 뽑는다 (NOTES D-24).

주행 세트(DDAD, nuScenes)는 장면당 여러 장(DDAD 5장, nuScenes 약 25장)이라 같은 장면의 사진끼리 상관이 있어 장면을 군집으로 쓴다.
실내 세트(iBims-1, NYUv2 — 사진 한 장이 사실상 장면 하나)와 그 밖의 세트는 사진을 군집으로 쓴다.
장면은 저장소에 고정한 ref/scenes_<name>.parquet 에서 읽는다 — H200 에 보낸 주행 평가 팩의 jsonl 에서 뽑은 것 (V-12).
예전에는 $DATA_ROOT/eval/<name>/<name>_val.jsonl 을 읽고, 파일이 없거나 이미지가 목록에 없으면 조용히 사진 단위로 물러났다.
로컬에서는 그 경로가 없거나 옛 nuScenes(trainval)를 가리켜 구간이 절반 이하로 좁아졌다. 이제 장면을 못 찾으면 멈춘다.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

SCENE_CLUSTERED = ("ddad", "nuscenes")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cache: dict[str, dict[str, str]] = {}

def scene_map(name: str, data_root: str | None = None) -> dict[str, str] | None:
    """평가 세트의 이미지 → 장면. 장면 단위 세트가 아니면 None. data_root 는 예전 호출 호환용으로만 받는다."""
    if name not in SCENE_CLUSTERED: return None
    if name not in _cache:
        d = pd.read_parquet(os.path.join(ROOT, "ref", f"scenes_{name}.parquet"))
        _cache[name] = dict(zip(d.image_id.astype(str), d.scene.astype(str)))
    return _cache[name]

def clusters(name: str, image_ids, data_root: str | None = None) -> np.ndarray:
    """픽셀마다 군집 id (장면 단위 세트는 장면, 나머지는 사진). 장면 단위 세트에서 장면을 모르는 이미지가 있으면 멈춘다."""
    ids = np.asarray(image_ids).astype(str); m = scene_map(name, data_root)
    if m is None: return ids
    miss = sorted({i for i in ids if i not in m})
    if miss: raise KeyError(f"[clusters] {name}: 장면을 모르는 이미지 {len(miss)}개 (예: {miss[0]}) — 다른 평가 세트 버전이 섞였다")
    return np.array([m[i] for i in ids])

def boot_ci(values, groups, n: int = 2000, seed: int = 0, stat=np.mean) -> tuple[float, float]:
    """군집 부트스트랩 95 % 구간: 군집을 복원 추출하고 그 군집의 픽셀을 모두 넣는다."""
    v = np.asarray(values, float); g = np.asarray(groups); u, inv = np.unique(g, return_inverse=True)
    rng = np.random.default_rng(seed); sums = np.bincount(inv, weights=v, minlength=len(u)); cnts = np.bincount(inv, minlength=len(u))
    draws = rng.integers(0, len(u), size=(n, len(u)))
    est = sums[draws].sum(1) / np.maximum(cnts[draws].sum(1), 1) if stat is np.mean else np.array([stat(np.concatenate([v[inv == k] for k in d])) for d in draws])
    lo, hi = np.percentile(est, [2.5, 97.5]); return float(lo), float(hi)
