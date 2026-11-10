"""路线 D（无 LLM 基线）：三视图轮廓体素雕刻（visual hull / space carving）+ marching cubes。

体素 (x,y,z) 被保留 ⇔ 正视图(x,z)、侧视图(y,z)、顶视图(x,y) 三张轮廓都覆盖它。
这是“只靠三张正交轮廓”能恢复的最大形体：凸出部分全对，凹陷（杯子内腔、门洞）一律被填实。

  python -m p6.routes.visual_hull --cases all [--res 256] [--source mask|render]
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import trimesh
from skimage import measure

from .. import views
from ..evaluate import eval_case, list_cases, summarize
from ..paths import BENCH_DATA
from ..stamp import new_run_dir, stamp_run, write_json


def carve(masks: dict[str, np.ndarray], res: int = 256, half: float = views.HALF) -> np.ndarray:
    """返回 occupancy[x,y,z] (res^3, bool)。masks 为 views 约定下的 (H,W) 轮廓。"""
    c = (np.arange(res) + 0.5) / res * 2 * half - half  # 体素中心
    H = masks["front"].shape[0]

    def u2col(u):
        return np.clip(((u + half) / (2 * half) * H).astype(int), 0, H - 1)

    def v2row(v):
        return np.clip(((half - v) / (2 * half) * H).astype(int), 0, H - 1)

    cx, cy, cz = u2col(c), u2col(c), v2row(c)  # 列索引用 x/y，行索引用 z/y
    ry = v2row(c)
    F = masks["front"][np.ix_(cz, cx)]   # [z, x]
    S = masks["side"][np.ix_(cz, cy)]    # [z, y]
    T = masks["top"][np.ix_(ry, cx)]     # [y, x]
    occ = F.T[:, None, :] & S.T[None, :, :] & T.T[:, :, None]
    return occ


def occ_to_mesh(occ: np.ndarray, half: float = views.HALF) -> trimesh.Trimesh:
    res = occ.shape[0]
    pad = np.pad(occ.astype(np.float32), 1)
    if pad.max() == 0:
        return trimesh.Trimesh()
    pitch = 2 * half / res
    v, f, _, _ = measure.marching_cubes(pad, 0.5, spacing=(pitch, pitch, pitch))
    v = v - pitch + (-half + pitch / 2)
    m = trimesh.Trimesh(v, f[:, ::-1], process=True)
    return m


def masks_for_case(case: str, source: str) -> dict[str, np.ndarray]:
    d = BENCH_DATA / case
    if source == "mask":
        return {v: views.load_mask(d / f"mask_{v}.png") for v in views.VIEWS}
    # 从灰模渲染图按“非白背景”抠出轮廓（模拟真实图片输入）
    return {v: views.load_view_mask(d / "views" / f"{v}.png") for v in views.VIEWS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="all")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--source", default="render", choices=["mask", "render"])
    ap.add_argument("--views-dir", default=None, help="用外部三视图（如 Qwen 生成并规范化后的）：<dir>/<case>/{front,side,top}.png")
    a = ap.parse_args()
    run = new_run_dir("D_visual_hull", f"res{a.res}_{'gen' if a.views_dir else a.source}")
    stamp_run(run, {"route": "D_visual_hull", "res": a.res, "source": a.source})
    rows = {}
    for case in list_cases(a.cases):
        t0 = time.time()
        cd = run / case
        cd.mkdir(parents=True, exist_ok=True)
        if a.views_dir:
            import cv2

            from ..views_post import bg_mask

            vd = Path(a.views_dir) / case
            if not vd.exists():
                continue
            masks = {v: bg_mask(cv2.imread(str(vd / f"{v}.png"))) for v in views.VIEWS}
            for v in views.VIEWS:
                shutil.copy(vd / f"{v}.png", cd / f"input_{v}.png")
        else:
            masks = masks_for_case(case, a.source)
            for v in views.VIEWS:  # 输入留痕
                src = BENCH_DATA / case / ("views" if a.source == "render" else "") / (f"{v}.png" if a.source == "render" else f"mask_{v}.png")
                shutil.copy(src, cd / f"input_{v}.png")
        occ = carve(masks, a.res)
        m = occ_to_mesh(occ)
        m.export(cd / "pred.obj")
        r = eval_case(case, cd / "pred.obj")
        r["seconds"] = round(time.time() - t0, 2)
        write_json(cd / "result.json", r)
        rows[case] = r
        print(f"{case:14s} CD={r['chamfer']:.4f} F@.02={r['fscore@0.02']:.3f} F@.05={r['fscore@0.05']:.3f} "
              f"vIoU={r['vol_iou']:.3f} silIoU={r['sil_iou_mean']:.3f}  {r['seconds']}s")
    res = {"route": "D_visual_hull", "params": vars(a), "per_case": rows, "mean": summarize(rows)}
    write_json(run / "result.json", res)
    print("mean", res["mean"], "\n->", run)


if __name__ == "__main__":
    main()
