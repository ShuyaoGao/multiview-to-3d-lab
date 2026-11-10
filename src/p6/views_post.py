"""三视图后处理：切分 → 去背景 → 对齐 → 统一尺度 → 输出符合 views.py 约定的标准三视图。

约定 = 第三角投影（ANSI）：正视图看 XZ，右视图（相机在 +X）看 YZ 且图像右侧=+Y（物体背面），
顶视图图像上方=+Y。三张图共用一个尺度，物体最长边 = 1.6 世界单位，画布 [-1,1]^2、512px。

一致性约束（正交三视图必须满足，生成模型经常违反）：
  正视图高 = 侧视图高（Z）；正视图宽 = 顶视图宽（X）；侧视图宽 = 顶视图高（Y）。
consistency() 给出三对尺寸的相对误差；canonicalize() 取两两平均后把每张图重采样到统一尺度。

  python -m p6.views_post --sheet sheet.png --layout front,side,top --out dir
  python -m p6.views_post --front f.png --side s.png --top t.png --out dir
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import views

RES = views.RES


def bg_mask(img: np.ndarray, tol: int = 18) -> np.ndarray:
    """估计背景色（四边像素中位数），与背景色差 > tol 的像素为前景。返回前景 bool。"""
    if img.ndim == 3 and img.shape[2] == 4:
        return img[:, :, 3] > 127
    rgb = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(border, axis=0)
    d = np.abs(rgb.astype(np.int32) - bg[None, None, :]).max(axis=2)
    m = d > tol
    k = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    # 填内部小孔（灰模高光可能接近背景色）；真实通孔面积大，保留
    n, lab, stats, _ = cv2.connectedComponentsWithStats((1 - m).astype(np.uint8), connectivity=4)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        touches = x == 0 or y == 0 or x + w >= m.shape[1] or y + h >= m.shape[0]
        if not touches and area < 0.002 * m.size:
            m[lab == i] = 1
    # 只保留面积足够大的连通块
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m)
    if n > 1:
        big = stats[1:, cv2.CC_STAT_AREA].max()
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= 0.02 * big:
                keep[lab == i] = 1
    return keep.astype(bool)


def split_sheet(img: np.ndarray, n: int) -> list[tuple[int, int]]:
    """按前景的列投影找 n 个水平排列的面板，返回每个面板的列区间。"""
    m = bg_mask(img)
    col = m.any(axis=0).astype(np.int32)
    runs, start = [], None
    for x, v in enumerate(np.append(col, 0)):
        if v and start is None:
            start = x
        elif not v and start is not None:
            runs.append([start, x])
            start = None
    # 合并过近的碎片，直到剩 n 段
    while len(runs) > n:
        gaps = [runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)]
        i = int(np.argmin(gaps))
        runs[i] = [runs[i][0], runs[i + 1][1]]
        del runs[i + 1]
    if len(runs) < n:
        # 退化：平均切
        W = img.shape[1]
        runs = [[int(W * k / n), int(W * (k + 1) / n)] for k in range(n)]
    pad = 6
    return [(max(0, a - pad), min(img.shape[1], b + pad)) for a, b in runs]


def _bbox(m):
    ys, xs = np.nonzero(m)
    return xs.min(), xs.max() + 1, ys.min(), ys.max() + 1


def consistency(masks: dict[str, np.ndarray]) -> dict:
    """三视图尺寸一致性（像素单位，各图需同一缩放——仅对同一张拼图里的面板有意义）。"""
    b = {v: _bbox(m) for v, m in masks.items()}
    w = {v: b[v][1] - b[v][0] for v in b}
    h = {v: b[v][3] - b[v][2] for v in b}
    pairs = {"Z(front.h vs side.h)": (h["front"], h["side"]),
             "X(front.w vs top.w)": (w["front"], w["top"]),
             "Y(side.w vs top.h)": (w["side"], h["top"])}
    out = {k: round(abs(a - c) / max(a, c), 4) for k, (a, c) in pairs.items()}
    out["max_rel_err"] = max(out.values())
    out["raw_px"] = {k: [int(a), int(c)] for k, (a, c) in pairs.items()}
    return out


def canonicalize(imgs: dict[str, np.ndarray], masks: dict[str, np.ndarray], same_scale: bool = True) -> tuple[dict, dict, dict]:
    """把三张图裁到前景包围盒后，按统一的世界尺寸重采样到标准画布。

    same_scale=True：三张图来自同一张拼图（像素尺度可比），世界尺寸 = 两两平均；
    否则各图自己的宽高比可信，但绝对尺度不可比，用“共享边长度相等”求每张图的缩放。
    """
    b = {v: _bbox(m) for v, m in masks.items()}
    w = {v: float(b[v][1] - b[v][0]) for v in b}
    h = {v: float(b[v][3] - b[v][2]) for v in b}
    if same_scale:
        X = (w["front"] + w["top"]) / 2
        Z = (h["front"] + h["side"]) / 2
        Y = (w["side"] + h["top"]) / 2
    else:
        # 以正视图像素为基准，侧视图按高度对齐 Z，顶视图按宽度对齐 X
        s_side = h["front"] / h["side"]
        s_top = w["front"] / w["top"]
        X = (w["front"] + w["top"] * s_top) / 2
        Z = (h["front"] + h["side"] * s_side) / 2
        Y = (w["side"] * s_side + h["top"] * s_top) / 2
    k = views.NORM_EXTENT / max(X, Y, Z)
    ext = {"X": X * k, "Y": Y * k, "Z": Z * k}
    target = {"front": (ext["X"], ext["Z"]), "side": (ext["Y"], ext["Z"]), "top": (ext["X"], ext["Y"])}
    out_img, out_mask = {}, {}
    for v in views.VIEWS:
        x0, x1, y0, y1 = b[v]
        crop = imgs[v][y0:y1, x0:x1]
        cm = masks[v][y0:y1, x0:x1].astype(np.uint8) * 255
        tw = max(1, int(round(target[v][0] / 2 * RES)))
        th = max(1, int(round(target[v][1] / 2 * RES)))
        crop = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
        cm = cv2.resize(cm, (tw, th), interpolation=cv2.INTER_LINEAR) > 127
        canvas = np.full((RES, RES, 3), 255, np.uint8)
        mcan = np.zeros((RES, RES), bool)
        ox, oy = (RES - tw) // 2, (RES - th) // 2
        c3 = crop if crop.ndim == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        c3 = c3[:, :, :3].copy()
        c3[~cm] = 255
        canvas[oy:oy + th, ox:ox + tw] = c3
        mcan[oy:oy + th, ox:ox + tw] = cm
        out_img[v], out_mask[v] = canvas, mcan
    return out_img, out_mask, {"extent_world": ext, "scale_px_to_world": k}


def hull_self_consistency(masks: dict[str, np.ndarray], res: int = 192) -> dict:
    """用三张轮廓雕刻出 visual hull 再投影回去：与输入轮廓的 IoU。<1 说明三视图互相矛盾。"""
    from .routes.visual_hull import carve, occ_to_mesh

    occ = carve(masks, res)
    m = occ_to_mesh(occ)
    if len(m.faces) == 0:
        return {v: 0.0 for v in views.VIEWS} | {"mean": 0.0}
    pm = views.silhouettes(np.asarray(m.vertices), np.asarray(m.faces), ss=2)
    r = {v: round(views.mask_iou(masks[v], pm[v]), 4) for v in views.VIEWS}
    r["mean"] = round(float(np.mean([r[v] for v in views.VIEWS])), 4)
    return r


def process(imgs: dict[str, np.ndarray], out: Path, same_scale: bool) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    masks = {v: bg_mask(imgs[v]) for v in views.VIEWS}
    rep = {"consistency_before": consistency(masks) if same_scale else None}
    ci, cm, info = canonicalize(imgs, masks, same_scale)
    for v in views.VIEWS:
        cv2.imwrite(str(out / f"{v}.png"), ci[v])
        views.save_mask(out / f"mask_{v}.png", cm[v])
    rep.update(info)
    rep["hull_self_consistency"] = hull_self_consistency(cm)
    (out / "views_post.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet")
    ap.add_argument("--layout", default="front,side,top", help="拼图里从左到右的视图名")
    ap.add_argument("--front")
    ap.add_argument("--side")
    ap.add_argument("--top")
    ap.add_argument("--flip-side", action="store_true", help="侧视图是从 -X 看的（左视图）时水平翻转")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.sheet:
        img = cv2.imread(a.sheet, cv2.IMREAD_COLOR)
        names = a.layout.split(",")
        spans = split_sheet(img, len(names))
        panels = {n: img[:, s0:s1] for n, (s0, s1) in zip(names, spans)}
        imgs = {v: panels[v] for v in views.VIEWS}
        same = True
    else:
        imgs = {v: cv2.imread(getattr(a, v), cv2.IMREAD_COLOR) for v in views.VIEWS}
        same = False
    if a.flip_side:
        imgs["side"] = imgs["side"][:, ::-1].copy()
    print(json.dumps(process(imgs, Path(a.out), same), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
