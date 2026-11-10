"""构建评测基准：bench/shapes/*.py → bench/data/<case>/{gt.obj, masks, 灰模三视图, meta.json}

用法：
  .venv/Scripts/python bench/build.py                # 构建全部
  .venv/Scripts/python bench/build.py --only gear mug
  .venv/Scripts/python bench/build.py --blind 3      # 另外生成 3 个“盲测”随机零件（参数不打印，供 LLM 测试防泄漏）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from p6 import blender, views  # noqa: E402
from p6.align import normalize  # noqa: E402
from p6.metrics import load_mesh  # noqa: E402
from p6.paths import BENCH, BENCH_DATA  # noqa: E402
from p6.stamp import env_stamp, write_json  # noqa: E402

CASES = {
    "bracket": ("hard-surface", "程序化生成（本项目，CC0）"),
    "flange_shaft": ("hard-surface", "程序化生成（本项目，CC0）"),
    "gear": ("hard-surface", "程序化生成（本项目，CC0）"),
    "mug": ("hard-surface/concave", "程序化生成（本项目，CC0）"),
    "table": ("furniture", "程序化生成（本项目，CC0）"),
    "chair": ("furniture", "程序化生成（本项目，CC0）"),
    "house": ("architecture", "程序化生成（本项目，CC0）"),
    "figure": ("character", "程序化生成（本项目，CC0）"),
    "spot": ("organic", "Keenan Crane 3D Model Repository, CC0 1.0"),
    "bob": ("organic", "Keenan Crane 3D Model Repository, CC0 1.0"),
}


def blind_script(seed: int) -> str:
    """随机机械零件：底座 + 若干凸台/筋 + 若干孔。参数只写进脚本文件，不打印。"""
    rnd = random.Random(seed)
    L = []
    bx, by, bz = rnd.uniform(1.0, 1.6), rnd.uniform(0.6, 1.2), rnd.uniform(0.12, 0.3)
    L.append(f"base = box(({bx:.3f}, {by:.3f}, {bz:.3f}), (0, 0, {bz/2:.3f}))")
    parts = []
    for k in range(rnd.randint(2, 4)):
        kind = rnd.choice(["box", "cyl", "prism"])
        x, y = rnd.uniform(-bx / 2 + 0.15, bx / 2 - 0.15), rnd.uniform(-by / 2 + 0.15, by / 2 - 0.15)
        h = rnd.uniform(0.2, 0.8)
        if kind == "box":
            sx, sy = rnd.uniform(0.1, 0.5), rnd.uniform(0.1, 0.5)
            L.append(f"p{k} = box(({sx:.3f}, {sy:.3f}, {h:.3f}), ({x:.3f}, {y:.3f}, {bz + h/2:.3f}))")
        elif kind == "cyl":
            r = rnd.uniform(0.08, 0.25)
            L.append(f"p{k} = cylinder({r:.3f}, {h:.3f}, ({x:.3f}, {y:.3f}, {bz + h/2:.3f}))")
        else:
            w = rnd.uniform(0.2, 0.5)
            L.append(f"p{k} = prism([({x - w/2:.3f}, {bz:.3f}), ({x + w/2:.3f}, {bz:.3f}), ({x:.3f}, {bz + h:.3f})], "
                     f"{rnd.uniform(0.08, 0.3):.3f}, plane='XZ', offset={y:.3f})")
        parts.append(f"p{k}")
    L.append(f"part = union(base, {', '.join(parts)})")
    holes = []
    for k in range(rnd.randint(1, 3)):
        r = rnd.uniform(0.05, 0.12)
        ax = rnd.choice(["Z", "Z", "Y"])
        x = rnd.uniform(-bx / 2 + 0.15, bx / 2 - 0.15)
        if ax == "Z":
            y = rnd.uniform(-by / 2 + 0.15, by / 2 - 0.15)
            L.append(f"h{k} = cylinder({r:.3f}, 3.0, ({x:.3f}, {y:.3f}, 0), axis='Z')")
        else:
            z = rnd.uniform(0.04, bz - 0.04) if bz > 0.15 else bz / 2
            r = min(r, bz / 2 - 0.02)
            L.append(f"h{k} = cylinder({r:.3f}, 3.0, ({x:.3f}, 0, {z:.3f}), axis='Y')")
        holes.append(f"h{k}")
    L.append(f"difference(part, {', '.join(holes)})")
    return "# 盲测随机零件（seed 固定，参数请勿给 LLM 看）\n" + "\n".join(L) + "\n"


def build_case(name: str, script: Path, category: str, source: str, res: int = 512) -> dict:
    d = BENCH_DATA / name
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    raw = d / "raw.obj"
    meta = blender.build_mesh(script, raw)
    if not meta.get("ok"):
        raise RuntimeError(f"{name}: {meta.get('error')}")
    m = normalize(load_mesh(raw))
    gt = d / "gt.obj"
    m.export(gt)
    raw.unlink()
    masks = views.silhouettes(np.asarray(m.vertices), np.asarray(m.faces), res)
    for v, mk in masks.items():
        views.save_mask(d / f"mask_{v}.png", mk)
    imgs = blender.render_ortho3(gt, d / "views", res)
    # 一致性自检：Blender 渲染的 alpha 与 numpy 光栅化轮廓必须几乎重合
    check = {}
    for v in views.VIEWS:
        a = views.load_mask(d / "views" / f"{v}_rgba.png")
        check[v] = round(views.mask_iou(a, masks[v]), 4)
    info = {
        "case": name, "category": category, "source": source,
        "script": str(script.relative_to(BENCH.parent)),
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "gt_sha256": hashlib.sha256(gt.read_bytes()).hexdigest(),
        "verts": int(len(m.vertices)), "faces": int(len(m.faces)), "watertight": bool(m.is_watertight),
        "volume": float(m.volume) if m.is_watertight else None,
        "extent": (m.bounds[1] - m.bounds[0]).round(4).tolist(),
        "render_vs_raster_iou": check, "seconds": round(time.time() - t0, 1),
    }
    write_json(d / "meta.json", info)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--blind", type=int, default=0)
    a = ap.parse_args()
    BENCH_DATA.mkdir(parents=True, exist_ok=True)
    results = {}
    names = a.only or list(CASES)
    for n in names:
        if n not in CASES:
            continue
        cat, src = CASES[n]
        info = build_case(n, BENCH / "shapes" / f"{n}.py", cat, src)
        results[n] = info
        print(n, info["faces"], info["watertight"], info["render_vs_raster_iou"], f"{info['seconds']}s")
    for i in range(a.blind):
        n = f"blind_{i:02d}"
        sp = BENCH_DATA / n / "blind_shape.py"
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(blind_script(1000 + i), encoding="utf-8")
        info = build_case(n, sp, "hard-surface/blind", "程序化随机生成（本项目，CC0）")
        results[n] = info
        print(n, info["faces"], info["render_vs_raster_iou"])
    write_json(BENCH_DATA / "_env_stamp.json", env_stamp({"blender": blender.blender_version()}))
    idx = BENCH_DATA / "index.json"
    old = json.loads(idx.read_text(encoding="utf-8")) if idx.exists() else {}
    old.update(results)
    write_json(idx, old)


if __name__ == "__main__":
    main()
