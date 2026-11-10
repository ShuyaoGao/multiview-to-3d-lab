"""用 ComfyUI + Qwen-Image-2.1 生成三视图，并量化“三视图准不准”。

两个实验：
  t2i ：文字描述 → 一张 [正|右|顶] 拼图 → 切分规范化 → 一致性（三对共享尺寸的相对误差 + visual hull 自洽 IoU）
  edit：基准用例的真值正视图 → 让模型补出侧视图和顶视图 → 与真值轮廓比 IoU（可直接量化“准确率”）
        → 规范化后的三视图可再交给路线 C/D/A 建模（--export-dir）

  python -m p6.gen_views t2i --objects "a desk lamp" "a toy robot" --seeds 0 1
  python -m p6.gen_views edit --cases bracket,spot --seeds 0
需要先启动 ComfyUI：python -m p6.comfy.client
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import views, views_post
from .comfy import workflows as W
from .comfy.client import ComfyClient
from .evaluate import list_cases
from .paths import BENCH_DATA
from .stamp import new_run_dir, stamp_run, write_json

T2I_PROMPT = (
    "Orthographic three-view technical drawing of {obj}. Exactly three views side by side on a pure white background, "
    "from left to right: FRONT view, RIGHT SIDE view, TOP view. Every view is a strict orthographic projection with no "
    "perspective, all at the same scale and aligned: the front view and the side view have the same height and share the "
    "same bottom line, the top view has exactly the same width as the front view. Plain medium grey matte clay render with clear shading, "
    "studio lighting, crisp dark outlines, generous white space between the views, no shadows, no ground plane, no text, "
    "no labels, no dimension lines."
)
EDIT_PROMPT = (
    "The input image is the FRONT orthographic view of an object (grey clay render on white). Draw an orthographic "
    "three-view sheet of exactly this object on a pure white background, from left to right: the same FRONT view, the "
    "RIGHT SIDE view (seen from the right of the front view), and the TOP view (seen from directly above, with the front "
    "of the object at the bottom of the image). Strict orthographic projection, identical scale in all three views, the "
    "front and side views have the same height and the same bottom line, the top view has the same width as the front "
    "view. Same grey clay material and thin dark outlines, generous white space between the views, no text, no shadows."
)
NEGATIVE = "perspective, isometric, 3/4 view, text, watermark, labels, shadow, floor, multiple objects, colorful background"


def _gen(c: ComfyClient, wf: dict, out: Path) -> tuple[Path, dict]:
    files, st = c.run(wf, out, timeout=1800)
    pngs = [f for f in files if f.suffix.lower() == ".png"]
    if not pngs:
        raise RuntimeError(f"没有输出图片: {st}")
    return pngs[0], st


def run_t2i(c, run: Path, objects: list[str], seeds: list[int], steps: int, lora: float, w: int, h: int) -> dict:
    rows = {}
    for oi, obj in enumerate(objects):
        for s in seeds:
            key = f"obj{oi:02d}_s{s}"
            d = run / key
            d.mkdir(parents=True, exist_ok=True)
            prompt = T2I_PROMPT.format(obj=obj)
            wf = W.qwen21_t2i(prompt, NEGATIVE, w, h, s, steps, 1.0, lora, prefix=f"p6_t2i_{key}")
            write_json(d / "workflow_api.json", wf)
            (d / "prompt.txt").write_text(prompt, encoding="utf-8")
            t0 = time.time()
            img_p, st = _gen(c, wf, d)
            img = cv2.imread(str(img_p))
            spans = views_post.split_sheet(img, 3)
            imgs = {v: img[:, a:b] for v, (a, b) in zip(views.VIEWS, spans)}
            try:
                rep = views_post.process(imgs, d / "views", same_scale=True)
            except Exception as e:  # noqa  抠图得到空前景（例如图太淡）
                rows[key] = {"object": obj, "seed": s, "failed": True, "error": repr(e)[:300]}
                write_json(d / "result.json", rows[key])
                print(f"{key} {obj[:30]:30s} 后处理失败（前景为空）")
                continue
            rows[key] = {"object": obj, "seed": s, "seconds": round(time.time() - t0, 1), "comfy_seconds": st.get("seconds"),
                         "consistency": rep["consistency_before"], "hull_self_consistency": rep["hull_self_consistency"],
                         "spans": spans}
            write_json(d / "result.json", rows[key])
            print(f"{key} {obj[:30]:30s} max_rel_err={rep['consistency_before']['max_rel_err']:.3f} "
                  f"hull_self={rep['hull_self_consistency']['mean']:.3f} {rows[key]['seconds']}s")
    return rows


def _best_flip_iou(gt: np.ndarray, pred: np.ndarray) -> tuple[float, bool]:
    a = views.mask_iou(gt, pred)
    b = views.mask_iou(gt, pred[:, ::-1])
    return (a, False) if a >= b else (b, True)


def run_edit(c, run: Path, cases: list[str], seeds: list[int], steps: int, lora: float, w: int, h: int,
             export_dir: Path | None) -> dict:
    rows = {}
    for case in cases:
        for s in seeds:
            key = f"{case}_s{s}"
            d = run / key
            d.mkdir(parents=True, exist_ok=True)
            up = c.upload(BENCH_DATA / case / "views" / "front.png", f"p6_edit_{case}_front.png")
            wf = W.qwen21_edit(up, EDIT_PROMPT, NEGATIVE, w, h, s, steps, 1.0, lora, prefix=f"p6_edit_{key}")
            write_json(d / "workflow_api.json", wf)
            t0 = time.time()
            img_p, st = _gen(c, wf, d)
            img = cv2.imread(str(img_p))
            spans = views_post.split_sheet(img, 3)
            imgs = {v: img[:, a:b] for v, (a, b) in zip(views.VIEWS, spans)}
            try:
                rep = views_post.process(imgs, d / "views", same_scale=True)
            except Exception as e:  # noqa
                rows[key] = {"case": case, "seed": s, "failed": True, "error": repr(e)[:300]}
                write_json(d / "result.json", rows[key])
                print(f"{key:16s} 后处理失败（前景为空）")
                continue
            gt = {v: views.load_mask(BENCH_DATA / case / f"mask_{v}.png") for v in views.VIEWS}
            pm = {v: views.load_mask(d / "views" / f"mask_{v}.png") for v in views.VIEWS}
            iou = {}
            for v in views.VIEWS:
                val, flipped = _best_flip_iou(gt[v], pm[v])
                iou[v] = {"iou": round(val, 4), "best_with_hflip": flipped, "iou_noflip": round(views.mask_iou(gt[v], pm[v]), 4)}
            rows[key] = {"case": case, "seed": s, "seconds": round(time.time() - t0, 1),
                         "consistency": rep["consistency_before"], "hull_self_consistency": rep["hull_self_consistency"],
                         "sil_iou_vs_gt": iou, "extent_world": rep["extent_world"]}
            write_json(d / "result.json", rows[key])
            if export_dir:
                ed = export_dir / case
                ed.mkdir(parents=True, exist_ok=True)
                for v in views.VIEWS:
                    im = cv2.imread(str(d / "views" / f"{v}.png"))
                    if v == "side" and iou["side"]["best_with_hflip"]:
                        im = im[:, ::-1]
                    cv2.imwrite(str(ed / f"{v}.png"), im)
            print(f"{key:16s} IoU vs GT: F {iou['front']['iou']:.3f} S {iou['side']['iou']:.3f}"
                  f"{'(flip)' if iou['side']['best_with_hflip'] else ''} T {iou['top']['iou']:.3f}"
                  f"{'(flip)' if iou['top']['best_with_hflip'] else ''}  max_rel_err={rep['consistency_before']['max_rel_err']:.3f}"
                  f"  {rows[key]['seconds']}s")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["t2i", "edit"])
    ap.add_argument("--objects", nargs="*", default=[
        "a cast-iron pipe flange with six bolt holes", "a simple wooden chair", "a ceramic coffee mug with a handle",
        "a cartoon cow toy", "a small two-storey house with a chimney"])
    ap.add_argument("--cases", default="named")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--lora", type=float, default=1.0, help="Fun-Acc-4Step LoRA 强度；0=不用（此时建议 --steps 25）")
    ap.add_argument("--width", type=int, default=1536)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--export-dir", default=None)
    ap.add_argument("--url", default="http://127.0.0.1:8196")
    a = ap.parse_args()
    c = ComfyClient(a.url)
    run = new_run_dir("G_qwen_views", f"{a.mode}_steps{a.steps}_lora{a.lora}")
    stamp_run(run, {"route": "G_qwen_views", "args": vars(a), "comfy_system_stats": c.system_stats(),
                    "models": [W.QWEN_UNET, W.QWEN_TE, W.QWEN_VAE] + ([W.QWEN_ACC_LORA] if a.lora else [])})
    if a.mode == "t2i":
        rows = run_t2i(c, run, a.objects, a.seeds, a.steps, a.lora, a.width, a.height)
        ok = [r for r in rows.values() if not r.get("failed")]
        mean = {"max_rel_err": float(np.mean([r["consistency"]["max_rel_err"] for r in ok])) if ok else None,
                "hull_self_consistency": float(np.mean([r["hull_self_consistency"]["mean"] for r in ok])) if ok else None,
                "n_ok": len(ok), "n_failed": len(rows) - len(ok)}
    else:
        rows = run_edit(c, run, list_cases(a.cases), a.seeds, a.steps, a.lora, a.width, a.height,
                        Path(a.export_dir) if a.export_dir else None)
        ok = [r for r in rows.values() if not r.get("failed")]
        mean = {v: float(np.mean([r["sil_iou_vs_gt"][v]["iou"] for r in ok])) if ok else None for v in views.VIEWS}
        mean["max_rel_err"] = float(np.mean([r["consistency"]["max_rel_err"] for r in ok])) if ok else None
        mean["n_ok"], mean["n_failed"] = len(ok), len(rows) - len(ok)
    write_json(run / "result.json", {"mode": a.mode, "args": vars(a), "per_item": rows, "mean": mean})
    print("mean", mean, "\n->", run)


if __name__ == "__main__":
    main()
