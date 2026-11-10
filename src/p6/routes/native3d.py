"""路线 A：原生图像/多视图 → 3D（经 ComfyUI 原生节点，免编译 CUDA 扩展）。

  python -m p6.routes.native3d --method pixal3d_mv --cases all
  python -m p6.routes.native3d --method hunyuan_mv --cases spot --side-as left
  python -m p6.routes.native3d --method trellis2 --cases all        # 只用正视图

输入：基准用例的灰模正视图/侧视图（顶视图这些模型都不接受）。
  --side-as：我们的侧视图（相机在 +X）作为模型的 left 还是 right 输入，见 README 的约定说明。
输出 GLB → 转 OBJ → 与真值相似变换对齐（aligned=True）→ 指标。需要先启动 ComfyUI：python -m p6.comfy.client
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import trimesh

from .. import blender
from ..comfy import workflows as W
from ..comfy.client import ComfyClient, gpu_free_mb
from ..evaluate import eval_case, list_cases, summarize
from ..paths import BENCH_DATA
from ..stamp import new_run_dir, stamp_run, write_json


def glb_to_obj(glb: Path, obj: Path) -> trimesh.Trimesh:
    sc = trimesh.load(str(glb), force="scene")
    m = sc.dump(concatenate=True) if isinstance(sc, trimesh.Scene) else sc
    m.export(obj)
    return m


def prep_rig(src: Path, dst: Path, mask_from: str = "view", pad: float = 1.1, size: int = 1024) -> Path:
    """把标准视图（同尺度、白底）转成 Pixal3D/TRELLIS 节点要求的“装配”格式：
    黑底、所有视图同一尺度、物体最长边约占画面 1/pad。只依赖输入图片本身（不看真值）。"""
    import cv2
    import numpy as np

    from .. import views
    from ..views_post import bg_mask

    img = cv2.imread(str(src))
    m = views.load_view_mask(src) if mask_from == "view" else bg_mask(img)
    half_px = views.RES / 2
    ext_px = views.NORM_EXTENT / 2 * views.RES / 2        # 物体最长边在 512 图里占的半宽（像素）
    crop_half = int(round(ext_px * pad))
    c = int(half_px)
    out = np.zeros_like(img)
    out[m] = img[m]
    out = out[c - crop_half:c + crop_half, c - crop_half:c + crop_half]
    out = cv2.resize(out, (size, size), interpolation=cv2.INTER_LANCZOS4)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), out)
    return dst


def build(method: str, c: ComfyClient, inputs: dict[str, Path], seed: int, prefix: str, side_as: str,
          upsample: int = 1536) -> dict:
    up = {k: c.upload(p, f"{prefix}_{k}.png") for k, p in inputs.items()}
    rig = all(str(p).endswith("_rig.png") for p in inputs.values())
    if method == "hunyuan_mv":
        v = {"front": up["front"], side_as: up["side"]}
        return W.hunyuan3d_mv(v, seed=seed, prefix=prefix)
    if method == "pixal3d_mv":
        v = {"front": up["front"], side_as: up["side"]}
        return W.pixal3d_mv(v, seed=seed, upsample_res=upsample, prefix=prefix, bg_crop=not rig)
    if method == "trellis2":
        return W.trellis2_image(up["front"], seed=seed, upsample_res=upsample, prefix=prefix, bg_crop=not rig)
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["hunyuan_mv", "pixal3d_mv", "trellis2"])
    ap.add_argument("--cases", default="all")
    ap.add_argument("--side-as", default="left", choices=["left", "right"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prep", default="rig", choices=["rig", "template"],
                    help="rig=我们自己做同尺度黑底预处理（符合节点说明）；template=官方模板的逐视图抠图裁剪")
    ap.add_argument("--rot-search", action="store_true",
                    help="评测对齐时做 24 种轴向旋转搜索。默认关闭：这些模型按正视图定姿，只需相似变换 ICP")
    ap.add_argument("--upsample", type=int, default=1024, help="TRELLIS.2 系上采样分辨率（官方模板 1536；1024 省时省显存）")
    ap.add_argument("--url", default="http://127.0.0.1:8196")
    ap.add_argument("--views-dir", default=None, help="用别处的三视图（如 Qwen 生成后规范化的）代替真值灰模渲染：<dir>/<case>/{front,side}.png")
    a = ap.parse_args()
    c = ComfyClient(a.url)
    run = new_run_dir("A_native3d", f"{a.method}_{a.side_as}_{a.prep}" + ("_gen" if a.views_dir else ""))
    stamp_run(run, {"route": "A_native3d", "args": vars(a), "gpu_free_mb_at_start": gpu_free_mb()[0],
                    "comfy_system_stats": c.system_stats()})
    rows = {}
    for case in list_cases(a.cases):
        cd = run / case
        cd.mkdir(parents=True, exist_ok=True)
        src = Path(a.views_dir) / case if a.views_dir else BENCH_DATA / case / "views"
        inputs = {"front": src / "front.png", "side": src / "side.png"}
        if a.prep == "rig" and a.method != "hunyuan_mv":
            mf = "bg" if a.views_dir else "view"
            inputs = {k: prep_rig(p, cd / f"input_{k}_rig.png", mf) for k, p in inputs.items()}
        prefix = f"p6A_{a.method}_{case}"
        wf = build(a.method, c, inputs, a.seed, prefix, a.side_as, a.upsample)
        write_json(cd / "workflow_api.json", wf)
        t0 = time.time()
        try:
            files, status = c.run(wf, cd / "comfy_out", timeout=3600)
        except Exception as e:  # noqa
            rows[case] = {"failed": True, "error": repr(e)[:2000], "chamfer": None, "fscore@0.02": 0.0,
                          "fscore@0.05": 0.0, "vol_iou": 0.0, "sil_iou_mean": 0.0}
            write_json(cd / "result.json", rows[case])
            print(case, "FAILED", repr(e)[:300])
            continue
        glbs = [f for f in files if f.suffix.lower() == ".glb"]
        if not glbs:
            rows[case] = {"failed": True, "error": f"no glb; status={status}", "chamfer": None, "fscore@0.02": 0.0,
                          "fscore@0.05": 0.0, "vol_iou": 0.0, "sil_iou_mean": 0.0}
            write_json(cd / "result.json", rows[case])
            print(case, "NO GLB", json.dumps(status)[:300])
            continue
        dinfo = blender.decimate(glbs[0], cd / "pred_raw.obj", faces=200000)   # GLB 是 glTF Y-up，导入时 Blender 转为 Z-up
        m = trimesh.load(cd / "pred_raw.obj", force="mesh")
        ev = eval_case(case, cd / "pred_raw.obj", align=True, save_aligned=cd / "pred_aligned.obj", allow_rot=a.rot_search)
        ev.update({"seconds": round(time.time() - t0, 1), "comfy_seconds": status.get("seconds"),
                   "raw_faces": dinfo["faces_in"], "decimated_tris": dinfo["tris_out"]})
        write_json(cd / "result.json", ev)
        rows[case] = ev
        (cd / "pred_aligned.obj").unlink(missing_ok=True)   # 体积大，评测后只保留简化后的 pred_raw.obj
        print(f"{case:14s} CD={ev['chamfer']:.4f} F@.02={ev['fscore@0.02']:.3f} vIoU={ev['vol_iou']:.3f} "
              f"silIoU={ev['sil_iou_mean']:.3f}  {ev['seconds']}s")
    write_json(run / "result.json", {"route": "A_native3d", "args": vars(a), "per_case": rows, "mean": summarize(rows)})
    print("mean", summarize(rows), "\n->", run)


if __name__ == "__main__":
    main()
