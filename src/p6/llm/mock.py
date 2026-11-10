"""规则 mock provider —— 不是 LLM！

没有任何 API key 时用来把“三视图 → bpy 脚本 → Blender → 渲染比对 → 反馈 → 修正”闭环完整跑通。
它直接读目标三视图（请求里附带的 target_*.png），按轮次给出越来越精细的 bpy 脚本：
  第 1 轮：三视图包围盒 → 一个长方体
  第 2 轮：正视图轮廓挤出 ∩ 包围盒
  第 3 轮起：正/侧/顶三张轮廓各自挤出后求交（等价于 CAD 形式的 visual hull，含通孔）
因此它的上限 = 路线 D 轮廓雕刻；它的意义是验证闭环工程本身，而不是代表 LLM 能力。
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from .. import views


def _find_targets(messages) -> dict[str, Path]:
    out = {}
    for m in messages:
        for p in m.get("images", []):
            mm = re.search(r"target_(front|side|top)\.png$", str(p))
            if mm:
                out[mm.group(1)] = Path(p)
    return out


def _contours(mask: np.ndarray, eps_px: float = 0.8):
    m = mask.astype(np.uint8) * 255
    cs, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    outers, holes = [], []
    if hier is None:
        return outers, holes
    for c, h in zip(cs, hier[0]):
        if cv2.contourArea(c) < 6:
            continue
        a = cv2.approxPolyDP(c, eps_px, True)[:, 0, :].astype(float)
        if len(a) < 3:
            continue
        u, v = views.pix_to_world(a[:, 0], a[:, 1])
        poly = [(round(float(x), 4), round(float(y), 4)) for x, y in zip(u, v)]
        (holes if h[3] >= 0 else outers).append(poly)
    return outers, holes


def _prism_code(var: str, mask: np.ndarray, plane: str) -> list[str]:
    outers, holes = _contours(mask)
    L = []
    parts = []
    for i, poly in enumerate(outers):
        L.append(f"{var}_o{i} = prism({poly}, 4.0, plane='{plane}')")
        parts.append(f"{var}_o{i}")
    if not parts:
        return [f"{var} = box((2,2,2))"]
    L.append(f"{var} = union({', '.join(parts)})" if len(parts) > 1 else f"{var} = {parts[0]}")
    hs = []
    for i, poly in enumerate(holes):
        L.append(f"{var}_h{i} = prism({poly}, 4.2, plane='{plane}')")
        hs.append(f"{var}_h{i}")
    if hs:
        L.append(f"{var} = difference({var}, {', '.join(hs)})")
    return L


def _bbox(masks):
    bf = views.mask_bbox_world(masks["front"], "front")
    bs = views.mask_bbox_world(masks["side"], "side")
    x0, x1, z0, z1 = bf["X_min"], bf["X_max"], bf["Z_min"], bf["Z_max"]
    y0, y1 = bs["Y_min"], bs["Y_max"]
    return x0, x1, y0, y1, z0, z1


class MockProvider:
    def __init__(self, system: str, **_):
        self.system = system
        self.turn = 0
        self.name = "mock:rule-based(not-an-LLM)"

    def chat(self, messages) -> str:
        self.turn += 1
        t = _find_targets(messages)
        masks = {v: views.load_view_mask(t[v]) for v in views.VIEWS}
        x0, x1, y0, y1, z0, z1 = _bbox(masks)
        L = [f"# mock 第 {self.turn} 轮"]
        if self.turn == 1:
            L.append(f"box(({x1 - x0:.4f}, {y1 - y0:.4f}, {z1 - z0:.4f}), "
                     f"({(x0 + x1) / 2:.4f}, {(y0 + y1) / 2:.4f}, {(z0 + z1) / 2:.4f}))")
            why = "先用三视图包围盒占位。"
        elif self.turn == 2:
            L += _prism_code("F", masks["front"], "XZ")
            L.append(f"bb = box(({x1 - x0:.4f}, {y1 - y0:.4f}, {z1 - z0 + 0.1:.4f}), "
                     f"({(x0 + x1) / 2:.4f}, {(y0 + y1) / 2:.4f}, {(z0 + z1) / 2:.4f}))")
            L.append("intersect(F, bb)")
            why = "正视图轮廓挤出，再用包围盒截断深度。"
        else:
            L += _prism_code("F", masks["front"], "XZ")
            L += _prism_code("S", masks["side"], "YZ")
            L += _prism_code("T", masks["top"], "XY")
            L.append("FS = intersect(F, S)")
            L.append("intersect(FS, T)")
            why = "三张轮廓分别挤出后求交（CAD 形式的 visual hull）。"
        return f"规则 mock：{why}\n\n```python\n" + "\n".join(L) + "\n```\n"
