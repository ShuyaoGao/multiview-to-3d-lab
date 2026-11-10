"""三视图约定 + 纯 numpy/cv2 的正交轮廓渲染。

坐标系（与 Blender 一致，Z 朝上）：
  front 正视图：相机在 -Y 向 +Y 看，图像 右=+X，上=+Z
  side  侧视图：相机在 +X 向 -X 看（右视图），图像 右=+Y，上=+Z
  top   顶视图：相机在 +Z 向下看，图像 右=+X，上=+Y
统一尺度：每张图覆盖世界坐标 [-HALF, HALF]^2（HALF=1.0），分辨率 RES。
真值网格被归一化到最长边 = 1.6、包围盒中心在原点，所以三张图同尺度、可直接互相对应。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

HALF = 1.0
RES = 512
NORM_EXTENT = 1.6
VIEWS = ("front", "side", "top")
# 每个视图：图像 u 轴、v 轴分别取世界坐标的哪一维
VIEW_AXES = {"front": (0, 2), "side": (1, 2), "top": (0, 1)}
# 视线方向（从相机指向物体），用于深度/遮挡说明
VIEW_DIRS = {"front": (0, 1, 0), "side": (-1, 0, 0), "top": (0, 0, -1)}


def world_to_pix(u: np.ndarray, v: np.ndarray, res: int = RES, half: float = HALF):
    col = (u + half) / (2 * half) * res
    row = (half - v) / (2 * half) * res
    return col, row


def pix_to_world(col: np.ndarray, row: np.ndarray, res: int = RES, half: float = HALF):
    u = (col + 0.5) / res * (2 * half) - half
    v = half - (row + 0.5) / res * (2 * half)
    return u, v


def silhouette(vertices: np.ndarray, faces: np.ndarray, view: str, res: int = RES, half: float = HALF,
               ss: int = 1) -> np.ndarray:
    """返回 bool 轮廓 (res,res)。ss>1 时超采样后再下采样（多数表决）。"""
    a, b = VIEW_AXES[view]
    r = res * ss
    col, row = world_to_pix(vertices[:, a], vertices[:, b], r, half)
    pts = np.stack([col, row], -1)
    tris = pts[faces]  # (F,3,2)
    img = np.zeros((r, r), np.uint8)
    # 逐个三角形 fillConvexPoly（fillPoly 一次传多个多边形会按奇偶规则互相抵消）
    # 投影面积≈0 的侧立面三角形跳过，否则会在孔洞边缘画出 1px 的线
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    area2 = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    tris = tris[area2 > 1e-6]
    shift = 4
    tri_i = np.round((tris - 0.5) * (1 << shift)).astype(np.int32)
    for t in tri_i:
        cv2.fillConvexPoly(img, t, 1, lineType=cv2.LINE_8, shift=shift)
    if ss > 1:
        img = img.reshape(res, ss, res, ss).mean(axis=(1, 3)) > 0.5
    return img.astype(bool)


def silhouettes(vertices: np.ndarray, faces: np.ndarray, res: int = RES, ss: int = 4) -> dict[str, np.ndarray]:
    return {v: silhouette(vertices, faces, v, res, HALF, ss) for v in VIEWS}


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return float(inter / uni) if uni else 1.0


def save_mask(path: Path, m: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.where(m, 0, 255).astype(np.uint8))  # 黑物体白背景


def load_mask(path: Path, thresh: int = 250) -> np.ndarray:
    """从 PNG 读轮廓：RGBA 用 alpha，否则把“非白色背景”当物体。"""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3 and img.shape[2] == 4:
        return img[:, :, 3] > 127
    if img.ndim == 3:
        gray = img.min(axis=2)
    else:
        gray = img
    return gray < thresh


def diff_image(target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """可视化差异：白=都空，黑=都有，红=目标有而预测缺(漏)，蓝=预测多出(溢)。BGR。"""
    out = np.full(target.shape + (3,), 255, np.uint8)
    both = target & pred
    miss = target & ~pred
    extra = ~target & pred
    out[both] = (60, 60, 60)
    out[miss] = (0, 0, 255)
    out[extra] = (255, 120, 0)
    return out


def mask_bbox_world(m: np.ndarray, view: str, res: int = RES, half: float = HALF) -> dict | None:
    """轮廓在世界坐标里的包围盒（给 LLM 反馈用的数字）。"""
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    u0, v1 = pix_to_world(xs.min(), ys.min(), res, half)
    u1, v0 = pix_to_world(xs.max(), ys.max(), res, half)
    a, b = VIEW_AXES[view]
    names = "XYZ"
    return {f"{names[a]}_min": round(float(u0), 3), f"{names[a]}_max": round(float(u1), 3),
            f"{names[b]}_min": round(float(v0), 3), f"{names[b]}_max": round(float(v1), 3)}


def load_view_mask(path: Path, thresh: int = 245, erode_px: int = 1) -> np.ndarray:
    """从“白底灰模+黑色描边”的视图图片提取轮廓：非白即物体，再腐蚀掉约 1px 的外描边。

    对 Blender workbench 渲染（描边画在轮廓外侧 ~1px）实测把与真值轮廓的 IoU 从 0.88–0.98 提到 0.97–0.996。
    生成式模型出的图走 src/p6/views_post 的抠图流程，得到干净 alpha 后用 load_mask。
    """
    m = load_mask(path, thresh)
    if erode_px > 0:
        k = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
        m = cv2.erode(m.astype(np.uint8), k).astype(bool)
    return m
