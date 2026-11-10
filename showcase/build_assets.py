"""从 runs/ 收集展示素材 → showcase/assets/（data.js、meshes.js、图片）。

  .venv/Scripts/python showcase/build_assets.py
- 所有数字取自 report_runs.json 指向的各 run 的 result.json（以及 runs/report/summary.json）。
- 网格用 Blender 简化到约 FACES 面，路线 A 用 result.json 里的对齐变换放到真值坐标系；GLB 以 base64 写进 meshes.js，
  这样 index.html 直接双击（file://）也能离线打开。
"""
from __future__ import annotations

import base64
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p6 import blender, views  # noqa: E402
from p6.metrics import load_mesh  # noqa: E402

OUT = ROOT / "showcase" / "assets"
SPEC = json.loads((ROOT / "report_runs.json").read_text(encoding="utf-8"))
ENT = {e["key"]: e for e in SPEC["entries"]}
VIEW_CASES = ["bracket", "gear", "chair", "mug", "house", "spot", "figure", "blind_01", "blind_05"]
VIEW_ROUTES = ["gt", "D", "C_claude", "C_mock", "A_pixal", "A_hunyuan", "A_trellis", "B"]
FACES = 3000


def rjson(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def per_case(key: str) -> dict:
    e = ENT.get(key)
    if not e or e["run"].startswith("__"):
        return {}
    return rjson(ROOT / e["run"] / "result.json").get("per_case", {})


def img_small(src: Path, dst: Path, size: int = 256, quality: int = 85) -> str:
    img = cv2.imread(str(src))
    h, w = img.shape[:2]
    s = size / max(h, w)
    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.suffix == ".jpg":
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(str(dst), img)
    return dst.relative_to(OUT.parent).as_posix()


def pred_mesh(key: str, case: str) -> trimesh.Trimesh | None:
    if key == "gt":
        return load_mesh(ROOT / "bench/data" / case / "gt.obj")
    e = ENT.get(key)
    if not e or e["run"].startswith("__"):
        return None
    d = ROOT / e["run"] / case
    rp = d / "result.json"
    if not rp.exists():
        return None
    r = rjson(rp)
    if r.get("source_run"):
        d = ROOT / r["source_run"] / case
    if (d / "pred_aligned.obj").exists():
        return load_mesh(d / "pred_aligned.obj")
    src = d / "pred.obj" if (d / "pred.obj").exists() else d / "pred_raw.obj"
    if not src.exists():
        return None
    m = load_mesh(src)
    if r.get("align_transform"):
        from p6.align import normalize

        gt = load_mesh(ROOT / "bench/data" / case / "gt.obj")
        m = normalize(m, float((gt.bounds[1] - gt.bounds[0]).max()))
        m.apply_translation((gt.bounds[0] + gt.bounds[1]) / 2)
        m.apply_transform(np.array(r["align_transform"]))
    return m


def cluster_simplify(m: trimesh.Trimesh, target: int) -> trimesh.Trimesh:
    """顶点聚类简化：把顶点吸附到网格单元中心并合并，逐步加粗网格直到面数达标。"""
    ext = float((m.bounds[1] - m.bounds[0]).max())
    for cells in (160, 120, 96, 72, 56, 44, 34):
        q = np.round((m.vertices - m.bounds[0]) / (ext / cells))
        _, inv = np.unique(q, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        verts = np.zeros((inv.max() + 1, 3))
        np.add.at(verts, inv, m.vertices)
        verts /= np.bincount(inv)[:, None]
        f = inv[m.faces]
        f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
        if len(f):
            _, keep = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
            f = f[np.sort(keep)]
        out = trimesh.Trimesh(verts, f, process=True)
        if len(out.faces) <= 3 * target:
            return out
    return out


def glb_b64(m: trimesh.Trimesh, tmpdir: Path) -> str:
    tmpdir.mkdir(parents=True, exist_ok=True)
    src, dec = tmpdir / "in.obj", tmpdir / "out.obj"
    m.export(src)
    if len(m.faces) > FACES:
        blender.decimate(src, dec, FACES)
        m = trimesh.load(dec, force="mesh")
    if len(m.faces) > 3 * FACES:
        # 碎片化网格（TRELLIS/Pixal 输出常见）Blender collapse 降不下来：退回顶点聚类
        m = cluster_simplify(m, FACES)
    m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))  # Z-up → glTF Y-up
    m.visual = trimesh.visual.ColorVisuals(m)
    return base64.b64encode(m.export(file_type="glb")).decode()


def summary_rows(group_title: str) -> list[dict]:
    s = rjson(ROOT / "runs/report/summary.json")["summary"][group_title]
    out = []
    for e in SPEC["entries"]:
        if e["label"] in s:
            row = {"label": e["label"], "note": e.get("note", ""), **s[e["label"]]}
            row["dim"] = e["key"].startswith("G")
            out.append(row)
    return out


def main():
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    groups = [g["title"] for g in SPEC["groups"]]
    data: dict = {"n_cases": len(rjson(ROOT / "bench/data/index.json"))}
    data["summary"] = {"blind": summary_rows(groups[0]), "named": summary_rows(groups[1])}

    # 逐物体 CD 表
    idx = rjson(ROOT / "bench/data/index.json")
    tbl_keys = ["D", "C_mock", "C_claude", "A_pixal", "A_hunyuan", "A_trellis", "B", "G_D"]
    pcs = {k: per_case(k) for k in tbl_keys}
    data["case_table"] = {
        "routes": [{"key": k, "label": ENT[k]["label"]} for k in tbl_keys if pcs[k]],
        "rows": [{"case": c, "category": idx[c]["category"],
                  "cd": {k: (pcs[k].get(c) or {}).get("chamfer") for k in tbl_keys if pcs[k]}} for c in idx],
    }

    # 查看器
    meshes, viewer = {}, {"routes": [], "cases": {}}
    labels = {"gt": "真值"} | {k: ENT[k]["label"] for k in VIEW_ROUTES if k in ENT}
    viewer["routes"] = [{"key": k, "label": labels[k]} for k in VIEW_ROUTES if k == "gt" or (k in ENT and not ENT[k]["run"].startswith("__"))]
    tmp = ROOT / "runs" / "showcase_tmp"
    for c in VIEW_CASES:
        vi = {"views": {}, "metrics": {}}
        for v in views.VIEWS:
            vi["views"][v] = img_small(ROOT / "bench/data" / c / "views" / f"{v}.png", OUT / "cases" / c / f"{v}.png", 220)
        for r in viewer["routes"]:
            m = pred_mesh(r["key"], c)
            if m is None:
                continue
            meshes[f"{c}/{r['key']}"] = glb_b64(m, tmp)
            if r["key"] != "gt":
                pr = per_case(r["key"]).get(c, {})
                vi["metrics"][r["key"]] = {k: pr.get(k) for k in ("chamfer", "fscore@0.02", "vol_iou")}
            print(c, r["key"], "ok", flush=True)
        viewer["cases"][c] = vi
    data["viewer"] = viewer
    shutil.rmtree(tmp, ignore_errors=True)

    # 生成视图（图 4）
    G = SPEC["gen"]
    ge = rjson(ROOT / G["edit_run"] / "result.json")
    rows = []
    for c in G["edit_show"]:
        key = f"{c}_s{G['edit_seed']}"
        it = ge["per_item"][key]
        d = ROOT / G["edit_run"] / key / "views"
        rows.append({"case": c, "iou": {v: it["sil_iou_vs_gt"][v]["iou"] for v in views.VIEWS},
                     "gt": {v: img_small(ROOT / "bench/data" / c / "views" / f"{v}.png", OUT / "gen" / c / f"gt_{v}.png", 200) for v in views.VIEWS},
                     "gen": {v: img_small(d / f"{v}.png", OUT / "gen" / c / f"gen_{v}.png", 200) for v in views.VIEWS}})
    t2i = []
    tr = rjson(ROOT / G["t2i_run"] / "result.json")
    for key in G["t2i_show"]:
        it = tr["per_item"][key]
        src = sorted((ROOT / G["t2i_run"] / key).glob("10_*.png"))[0]
        t2i.append({"prompt": it["object"], "err": it["consistency"]["max_rel_err"],
                    "img": img_small(src, OUT / "img" / f"t2i_{key}.jpg", 768, 82)})
    m = ge["mean"]
    data["gen"] = {"rows": rows, "t2i": t2i,
                   "summary": (f"给 Qwen-Image-2.1 一张真值正视图，让它补出侧视图和顶视图，再与真值轮廓比较（{m['n_ok']} 张，10 个物体 × 2 个种子）："
                               f"正视图 IoU {m['front']:.3f}，侧视图 {m['side']:.3f}，顶视图 {m['top']:.3f}"
                               f"（侧/顶视图允许水平翻转后取较好者，属于偏乐观的口径）。三对共享尺寸的平均最大相对误差 {m['max_rel_err']:.3f}。"
                               "画面看起来像三视图，但几何经常不对：顶视图照抄正视图、圆杯画成方杯、L 形支架的侧面画成薄板。")}

    # 闭环（图 5）
    L = SPEC["loop"]
    lj = rjson(ROOT / L["run"] / L["case"] / "loop.json")
    data["loop"] = []
    for it in lj["iterations"]:
        d = ROOT / L["run"] / L["case"] / f"iter_{it['iter']:02d}"
        data["loop"].append({"iter": it["iter"], "iou": it["sil_iou"]["mean"],
                             "diff": {v: img_small(d / f"diff_{v}.png", OUT / "loop" / f"it{it['iter']}_{v}.png", 220) for v in views.VIEWS}})

    # 叙述文字里的 {占位符} 全部用 result.json 的数字填
    num = {}
    summ = rjson(ROOT / "runs/report/summary.json")["summary"]
    for gname, tag in ((groups[0], "blind"), (groups[1], "named"), (groups[2], "spot")):
        for e in SPEC["entries"]:
            if e["label"] in summ.get(gname, {}):
                num[f"{tag}_{e['key']}"] = summ[gname][e["label"]]["chamfer"]
    allb = rjson(ROOT / ENT["B"]["run"] / "result.json")["mean"]["chamfer"]
    num["all_B"] = allb
    num["gen_side"], num["gen_top"] = m["side"], m["top"]
    num["loop1"], num["loop2"] = data["loop"][0]["iou"], data["loop"][-1]["iou"]
    data["loop_note"] = L["note"].format(**num)

    def fmt(x):
        if isinstance(x, str):
            return x.format(**num) if "{" in x else x
        if isinstance(x, list):
            return [fmt(i) for i in x]
        if isinstance(x, dict):
            return {k: fmt(v) for k, v in x.items()}
        return x

    for k in ("findings", "routes_story", "compare", "train", "decide"):
        data[k] = fmt(SPEC["story"][k])
    data["numbers"] = num
    shutil.copy(ROOT / "docs/img/dataset_demo.png", OUT / "img" / "dataset_demo.png")

    (OUT / "data.js").write_text("window.P6DATA = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    (OUT / "meshes.js").write_text("window.P6MESH = " + json.dumps(meshes) + ";\n", encoding="utf-8")
    tot = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"meshes.js {(OUT / 'meshes.js').stat().st_size / 1e6:.2f} MB；assets 合计 {tot / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
