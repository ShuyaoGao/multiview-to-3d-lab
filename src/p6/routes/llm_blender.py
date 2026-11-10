"""路线 C：三视图 → LLM 写 bpy 脚本 → 无头 Blender 建模 → 渲染三视图 → 轮廓 IoU → 差异反馈 → 修正。

  # 基准用例（有真值，最后算 3D 指标）
  python -m p6.routes.llm_blender --provider mock --cases all --iters 4
  python -m p6.routes.llm_blender --provider anthropic --cases blind --iters 6
  python -m p6.routes.llm_blender --provider file --cases blind_00     # 人/外部代理充当 LLM
  # 任意三张图（无真值，只看轮廓 IoU）
  python -m p6.routes.llm_blender --provider anthropic --images front.png side.png top.png

闭环只用“目标图片”里能得到的信息（轮廓、包围盒）；真值网格只在最后评测时使用。
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from .. import blender, views
from ..evaluate import eval_case, list_cases, summarize
from ..llm import prompts
from ..llm.providers import make_provider
from ..metrics import load_mesh
from ..paths import BENCH_DATA
from ..stamp import new_run_dir, stamp_run, write_json


def _measure(masks):
    return {v: views.mask_bbox_world(masks[v], v) for v in views.VIEWS}


def run_loop(target_imgs: dict[str, Path], out: Path, provider_kind: str, iters: int = 5,
             stop_iou: float = 0.985, provider_kw: dict | None = None, note: str = "") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    tg = {}
    for v in views.VIEWS:
        tg[v] = out / f"target_{v}.png"
        shutil.copy(target_imgs[v], tg[v])
    tmask = {v: views.load_view_mask(tg[v]) for v in views.VIEWS}
    tmeas = _measure(tmask)
    system = prompts.system_prompt()
    prompts.write_prompt_snapshot(out)
    kw = dict(provider_kw or {})
    if provider_kind == "file":
        kw.setdefault("exchange_dir", out / "exchange")
    prov = make_provider(provider_kind, system, **kw)
    history = [{"role": "user", "text": prompts.first_message(tmeas, note), "images": [tg[v] for v in views.VIEWS]}]
    log, best = [], None
    for it in range(1, iters + 1):
        d = out / f"iter_{it:02d}"
        d.mkdir(exist_ok=True)
        # 只保留第一条（目标图）和最新一条反馈的图片，控制上下文体积
        msgs = []
        for k, m in enumerate(history):
            mm = dict(m)
            if m["role"] == "user" and 0 < k < len(history) - 1:
                mm["images"] = []
            msgs.append(mm)
        t0 = time.time()
        try:
            reply = prov.chat(msgs)
        except Exception as e:  # noqa
            log.append({"iter": it, "error": f"provider: {e!r}"})
            break
        t_llm = time.time() - t0
        (d / "response.md").write_text(reply, encoding="utf-8")
        code = prompts.extract_code(reply)
        rec = {"iter": it, "llm_seconds": round(t_llm, 1)}
        if code is None:
            fb = "Your reply contained no ```python code block. Send the complete script in one code block."
            rec["error"] = "no code block"
        else:
            sp = d / "script.py"
            sp.write_text(code, encoding="utf-8")
            meta = blender.build_mesh(sp, d / "mesh.obj")
            rec["blender"] = {k: meta.get(k) for k in ("ok", "tris", "wall_seconds", "stage", "union_note")}
            if not meta.get("ok"):
                rec["error"] = meta.get("error", "")[-1500:]
                fb = prompts.feedback_message(it, {}, {}, {}, {}, err=meta.get("error", "unknown"))
            else:
                m = load_mesh(d / "mesh.obj")
                pm = views.silhouettes(np.asarray(m.vertices), np.asarray(m.faces))
                ious = {v: views.mask_iou(tmask[v], pm[v]) for v in views.VIEWS}
                ious["mean"] = float(np.mean([ious[v] for v in views.VIEWS]))
                px = {v: {"missing": int((tmask[v] & ~pm[v]).sum()), "extra": int((~tmask[v] & pm[v]).sum())}
                      for v in views.VIEWS}
                for v in views.VIEWS:
                    cv2.imwrite(str(d / f"diff_{v}.png"), views.diff_image(tmask[v], pm[v]))
                rend = blender.render_ortho3(d / "mesh.obj", d / "render")
                rec.update({"sil_iou": ious, "px": px})
                fb = prompts.feedback_message(it, ious, tmeas, _measure(pm), px)
                if best is None or ious["mean"] > best["sil_iou"]["mean"]:
                    best = {"iter": it, "sil_iou": ious, "mesh": d / "mesh.obj"}
                imgs = [d / f"diff_{v}.png" for v in views.VIEWS] + [rend[v] for v in views.VIEWS]
                history.append({"role": "assistant", "text": reply})
                history.append({"role": "user", "text": fb, "images": imgs})
                (d / "feedback.md").write_text(fb, encoding="utf-8")
                log.append(rec)
                print(f"  iter {it}: silIoU mean={ious['mean']:.3f} (F {ious['front']:.3f} S {ious['side']:.3f} "
                      f"T {ious['top']:.3f})  llm {t_llm:.0f}s")
                if ious["mean"] >= stop_iou:
                    break
                continue
        history.append({"role": "assistant", "text": reply})
        history.append({"role": "user", "text": fb, "images": []})
        (d / "feedback.md").write_text(fb, encoding="utf-8")
        log.append(rec)
        print(f"  iter {it}: ERROR {rec.get('error', '')[:120]!r}")
    res = {"provider": prov.name, "iterations": log, "best": best}
    if hasattr(prov, "usage"):
        res["usage"] = prov.usage
    if best:
        shutil.copy(best["mesh"], out / "pred.obj")
    write_json(out / "loop.json", res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="mock", choices=["mock", "anthropic", "openai", "file"])
    ap.add_argument("--cases", default=None)
    ap.add_argument("--images", nargs=3, metavar=("FRONT", "SIDE", "TOP"))
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--stop-iou", type=float, default=0.985)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    run = new_run_dir(f"C_llm_blender", f"{a.provider}{'_' + a.tag if a.tag else ''}")
    stamp_run(run, {"route": "C_llm_blender", "args": vars(a), "blender": blender.blender_version()})
    if a.images:
        tgt = dict(zip(views.VIEWS, map(Path, a.images)))
        r = run_loop(tgt, run / "custom", a.provider, a.iters, a.stop_iou)
        write_json(run / "result.json", {"route": "C_llm_blender", "custom": r})
        return
    rows = {}
    for case in list_cases(a.cases or "all"):
        print(f"[{case}]")
        t0 = time.time()
        tgt = {v: BENCH_DATA / case / "views" / f"{v}.png" for v in views.VIEWS}
        cd = run / case
        r = run_loop(tgt, cd, a.provider, a.iters, a.stop_iou)
        if (cd / "pred.obj").exists():
            ev = eval_case(case, cd / "pred.obj")
        else:
            ev = {"chamfer": None, "fscore@0.02": 0.0, "fscore@0.05": 0.0, "vol_iou": 0.0, "sil_iou_mean": 0.0,
                  "failed": True}
        ev["best_iter"] = r["best"]["iter"] if r["best"] else None
        ev["n_iters"] = len(r["iterations"])
        ev["seconds"] = round(time.time() - t0, 1)
        write_json(cd / "result.json", ev)
        rows[case] = ev
        print(f"  => CD={ev['chamfer']} F@.02={ev['fscore@0.02']:.3f} vIoU={ev['vol_iou']:.3f} silIoU={ev['sil_iou_mean']:.3f}")
    write_json(run / "result.json", {"route": "C_llm_blender", "provider": a.provider, "args": vars(a),
                                     "per_case": rows, "mean": summarize(rows)})
    print("mean", summarize(rows), "\n->", run)


if __name__ == "__main__":
    main()
