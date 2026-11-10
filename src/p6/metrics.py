"""几何指标：Chamfer 距离、F-score、体积 IoU、三视图轮廓 IoU。

全部在“归一化世界坐标”里算：真值最长边 = 1.6，视图窗口 [-1,1]。
  - chamfer：对称平均最近邻欧氏距离 0.5*(mean d(P→Q) + mean d(Q→P))，越小越好
  - fscore@τ：τ 取 0.02 / 0.05（约为物体尺寸的 1.25% / 3%）
  - vol_iou：128^3 体素占据（表面体素化 + 孔洞填充）的 IoU
  - sil_iou_*：三个正交视图的轮廓 IoU，以及均值
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree

from .views import HALF, VIEWS, mask_iou, silhouettes

TAUS = (0.02, 0.05)


def load_mesh(path) -> trimesh.Trimesh:
    m = trimesh.load(str(path), force="mesh", process=True)
    if not isinstance(m, trimesh.Trimesh):
        raise ValueError(f"无法读为网格: {path}")
    return m


def sample_surface(mesh: trimesh.Trimesh, n: int, seed: int = 0) -> np.ndarray:
    if len(mesh.faces) == 0:
        return np.zeros((0, 3))
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(pts)


def chamfer_fscore(gt: trimesh.Trimesh, pred: trimesh.Trimesh, n: int = 30000, taus=TAUS) -> dict:
    p_gt = sample_surface(gt, n, 1)
    p_pr = sample_surface(pred, n, 2)
    if len(p_pr) == 0:
        return {"chamfer": None, **{f"fscore@{t}": 0.0 for t in taus}}
    d_gp, _ = cKDTree(p_pr).query(p_gt)   # 完整性（recall）
    d_pg, _ = cKDTree(p_gt).query(p_pr)   # 准确性（precision）
    out = {"chamfer": float(0.5 * (d_gp.mean() + d_pg.mean())),
           "acc_mean": float(d_pg.mean()), "comp_mean": float(d_gp.mean())}
    for t in taus:
        prec = float((d_pg < t).mean())
        rec = float((d_gp < t).mean())
        out[f"fscore@{t}"] = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return out


def occupancy(mesh: trimesh.Trimesh, res: int = 128, half: float = HALF) -> np.ndarray:
    """沿 X/Y/Z 三个方向射线奇偶计数，三票取二（对小破洞、非严格水密网格较稳）。"""
    c = (np.arange(res) + 0.5) / res * 2 * half - half
    votes = np.zeros((res, res, res), np.uint8)
    for k in range(3):
        i, j = [a for a in range(3) if a != k]
        I, J = np.meshgrid(c, c, indexing="ij")
        org = np.zeros((res * res, 3))
        org[:, i] = I.ravel()
        org[:, j] = J.ravel()
        org[:, k] = -half - 0.5
        d = np.zeros_like(org)
        d[:, k] = 1.0
        locs, ridx, _ = mesh.ray.intersects_location(org, d, multiple_hits=True)
        inside = np.zeros((res * res, res), bool)
        if len(ridx):
            t = locs[:, k]
            order = np.lexsort((t, ridx))
            ridx, t = ridx[order], t[order]
            # 去掉同一射线上几乎重合的重复命中（共享边/顶点）
            keep = np.ones(len(t), bool)
            keep[1:] = ~((ridx[1:] == ridx[:-1]) & (np.abs(t[1:] - t[:-1]) < 1e-7))
            ridx, t = ridx[keep], t[keep]
            starts = np.searchsorted(ridx, np.arange(res * res))
            ends = np.searchsorted(ridx, np.arange(res * res), side="right")
            for r in np.nonzero(ends > starts)[0]:
                cnt = np.searchsorted(t[starts[r]:ends[r]], c)
                inside[r] = (cnt % 2) == 1
        inside = inside.reshape(res, res, res)  # [i, j, k]
        votes += np.moveaxis(inside, (0, 1, 2), (i, j, k)).astype(np.uint8)
    return votes >= 2


def occupancy_surface_fill(mesh: trimesh.Trimesh, res: int = 128, half: float = HALF) -> np.ndarray:
    """备用：表面采样体素化 + 孔洞填充（不依赖射线，但表面采样有缝时会填充失败）。"""
    pitch = 2 * half / res
    area = float(mesh.area)
    n = int(np.clip(area / (0.35 * pitch) ** 2, 50000, 4_000_000))
    pts = sample_surface(mesh, n, 3)
    pts = np.concatenate([pts, mesh.vertices], 0)
    idx = np.floor((pts + half) / pitch).astype(int)
    ok = np.all((idx >= 0) & (idx < res), axis=1)
    idx = idx[ok]
    grid = np.zeros((res, res, res), bool)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return ndimage.binary_fill_holes(grid)


def volume_iou(gt: trimesh.Trimesh, pred: trimesh.Trimesh, res: int = 128) -> float:
    a = occupancy(gt, res)
    b = occupancy(pred, res)
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def silhouette_ious(gt_masks: dict, pred: trimesh.Trimesh) -> dict:
    pm = silhouettes(np.asarray(pred.vertices), np.asarray(pred.faces))
    out = {f"sil_iou_{v}": mask_iou(gt_masks[v], pm[v]) for v in VIEWS}
    out["sil_iou_mean"] = float(np.mean([out[f"sil_iou_{v}"] for v in VIEWS]))
    return out


def evaluate(gt: trimesh.Trimesh, pred: trimesh.Trimesh, gt_masks: dict | None = None) -> dict:
    if gt_masks is None:
        gt_masks = silhouettes(np.asarray(gt.vertices), np.asarray(gt.faces))
    r = {}
    r.update(chamfer_fscore(gt, pred))
    r["vol_iou"] = volume_iou(gt, pred)
    r.update(silhouette_ious(gt_masks, pred))
    r["pred_faces"] = int(len(pred.faces))
    r["pred_watertight"] = bool(pred.is_watertight)
    return r
