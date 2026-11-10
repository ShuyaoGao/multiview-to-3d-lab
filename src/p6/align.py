"""把“坐标系未知”的预测网格对齐到真值（路线 A/B 需要；路线 C/D 本来就在同一坐标系）。

步骤：包围盒中心+最长边归一 → 24 种轴对齐旋转里挑 Chamfer 最小 → 相似变换 ICP 微调。
注意：对齐用到了真值，所以是“乐观上界”式的评估，报告里要写明 aligned=True。
"""
from __future__ import annotations

import itertools

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .views import NORM_EXTENT


def normalize(mesh: trimesh.Trimesh, extent: float = NORM_EXTENT) -> trimesh.Trimesh:
    m = mesh.copy()
    lo, hi = m.bounds
    m.apply_translation(-(lo + hi) / 2)
    s = extent / float((hi - lo).max())
    m.apply_scale(s)
    return m


def _rotations24() -> list[np.ndarray]:
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3))
            for i in range(3):
                R[i, perm[i]] = signs[i]
            if np.linalg.det(R) > 0:
                mats.append(R)
    return mats


def _cd(a: np.ndarray, b: np.ndarray, tb: cKDTree | None = None) -> float:
    tb = tb or cKDTree(b)
    d1, _ = tb.query(a)
    d2, _ = cKDTree(a).query(b)
    return 0.5 * (d1.mean() + d2.mean())


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / xs.var(0).sum() if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def align_to_gt(pred: trimesh.Trimesh, gt: trimesh.Trimesh, n: int = 6000, icp_iters: int = 40,
                allow_rot: bool = True) -> tuple[trimesh.Trimesh, dict]:
    p = normalize(pred, float((gt.bounds[1] - gt.bounds[0]).max()))
    p.apply_translation((gt.bounds[0] + gt.bounds[1]) / 2)
    gpts = np.asarray(trimesh.sample.sample_surface(gt, n, seed=5)[0])
    tg = cKDTree(gpts)
    ppts = np.asarray(trimesh.sample.sample_surface(p, n, seed=6)[0])
    c = ppts.mean(0)
    best = (np.inf, np.eye(3))
    for R in (_rotations24() if allow_rot else [np.eye(3)]):
        q = (ppts - c) @ R.T + c
        cd = _cd(q, gpts, tg)
        if cd < best[0]:
            best = (cd, R)
    R0 = best[1]
    T = np.eye(4)
    T[:3, :3] = R0
    T[:3, 3] = c - R0 @ c
    q = trimesh.transform_points(ppts, T)
    # 相似变换 ICP（点到点，双向截断）
    for _ in range(icp_iters):
        d, j = tg.query(q)
        keep = d < np.percentile(d, 90)
        s, R, t = umeyama(q[keep], gpts[j[keep]], True)
        Ti = np.eye(4)
        Ti[:3, :3] = s * R
        Ti[:3, 3] = t
        q = trimesh.transform_points(q, Ti)
        T = Ti @ T
    out = p.copy()
    out.apply_transform(T)
    info = {"init_rot_cd": float(best[0]), "final_cd_sample": float(_cd(q, gpts, tg)), "transform": T.tolist()}
    return out, info
