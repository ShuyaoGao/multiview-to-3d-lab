"""统一评测入口：给定基准用例名和预测网格，算全部指标。

  python -m p6.evaluate --case gear --pred runs/.../pred.obj [--align]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import views
from .align import align_to_gt
from .metrics import evaluate, load_mesh
from .paths import BENCH_DATA


def list_cases(spec: str | list[str] | None = "all") -> list[str]:
    idx = json.loads((BENCH_DATA / "index.json").read_text(encoding="utf-8"))
    if spec in (None, "all") or spec == ["all"]:
        return list(idx)
    if isinstance(spec, str):
        spec = spec.split(",")
    out = []
    for s in spec:
        if s == "named":
            out += [k for k in idx if not k.startswith("blind")]
        elif s == "blind":
            out += [k for k in idx if k.startswith("blind")]
        else:
            out.append(s)
    return out


def gt_masks(case: str) -> dict[str, np.ndarray]:
    return {v: views.load_mask(BENCH_DATA / case / f"mask_{v}.png") for v in views.VIEWS}


def eval_case(case: str, pred_path: Path, align: bool = False, save_aligned: Path | None = None,
              allow_rot: bool = True) -> dict:
    gt = load_mesh(BENCH_DATA / case / "gt.obj")
    pred = load_mesh(pred_path)
    info = {}
    if align:
        pred, ainfo = align_to_gt(pred, gt, allow_rot=allow_rot)
        info["align"] = {k: v for k, v in ainfo.items() if k != "transform"}
        info["align_transform"] = ainfo["transform"]
        if save_aligned:
            pred.export(save_aligned)
    r = evaluate(gt, pred, gt_masks(case))
    r["aligned"] = align
    r["align_rot_search"] = allow_rot if align else None
    r.update(info)
    return r


def summarize(rows: dict[str, dict], keys=("chamfer", "fscore@0.02", "fscore@0.05", "vol_iou", "sil_iou_mean")) -> dict:
    out = {}
    for k in keys:
        vals = [r[k] for r in rows.values() if r.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    out["n"] = len(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--align", action="store_true")
    a = ap.parse_args()
    print(json.dumps(eval_case(a.case, Path(a.pred), a.align), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
