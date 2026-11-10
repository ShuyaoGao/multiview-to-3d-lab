"""汇总各路线 run 的 result.json → Markdown 表 + 图（REPORT.md 的数字都由它生成）。

  python -m p6.report --runs report_runs.json
report_runs.json: [{"label": "D 轮廓雕刻", "run": "runs/D_visual_hull/...", "note": "..."}, ...]
输出：runs/report/summary.json、runs/report/tables.md、docs/img/route_compare.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .paths import ROOT

KEYS = [("chamfer", "CD↓", "{:.4f}"), ("fscore@0.02", "F@0.02↑", "{:.3f}"), ("fscore@0.05", "F@0.05↑", "{:.3f}"),
        ("vol_iou", "体积IoU↑", "{:.3f}"), ("sil_iou_mean", "轮廓IoU↑", "{:.3f}")]


def load(run: str) -> dict:
    r = json.loads((ROOT / run / "result.json").read_text(encoding="utf-8"))
    return r.get("per_case", {})


def mean_over(rows: dict, cases: list[str], key: str):
    vals = []
    for c in cases:
        r = rows.get(c)
        if r is None:
            return None
        v = r.get(key)
        if v is None:  # 失败按最差处理
            v = {"chamfer": np.nan}.get(key, 0.0)
        vals.append(v)
    vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else None


def table(entries: list[dict], cases: list[str], title: str) -> str:
    L = [f"### {title}（n={len(cases)}：{', '.join(cases)}）", "",
         "| 路线 | " + " | ".join(k[1] for k in KEYS) + " | 说明 |", "|---" * (len(KEYS) + 2) + "|"]
    for e in entries:
        rows = load(e["run"])
        if not all(c in rows for c in cases):
            continue
        cells = []
        for k, _, fmt in KEYS:
            v = mean_over(rows, cases, k)
            cells.append("—" if v is None else fmt.format(v))
        L.append(f"| {e['label']} | " + " | ".join(cells) + f" | {e.get('note', '')} |")
    return "\n".join(L) + "\n"


def per_case_table(entries: list[dict], cases: list[str], key: str = "chamfer") -> str:
    labs = [e["label"] for e in entries]
    data = [load(e["run"]) for e in entries]
    L = [f"### 逐用例 {key}", "", "| 用例 | " + " | ".join(labs) + " |", "|---" * (len(labs) + 1) + "|"]
    for c in cases:
        cells = []
        for rows in data:
            r = rows.get(c)
            v = None if r is None else r.get(key)
            cells.append("—" if v is None else (f"{v:.4f}" if key == "chamfer" else f"{v:.3f}"))
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def plot(entries: list[dict], cases: list[str], out: Path, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    labs, f2, viou = [], [], []
    for e in entries:
        rows = load(e["run"])
        if not all(c in rows for c in cases):
            continue
        labs.append(e["label"])
        f2.append(mean_over(rows, cases, "fscore@0.02"))
        viou.append(mean_over(rows, cases, "vol_iou"))
    y = np.arange(len(labs))
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(labs) + 1.2))
    ax.barh(y - 0.2, f2, 0.4, label="F-score@0.02", color="#3b6ea8")
    ax.barh(y + 0.2, viou, 0.4, label="体积 IoU", color="#c9873a")
    ax.set_yticks(y, labs)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{title}（n={len(cases)}）", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "report_runs.json"))
    a = ap.parse_args()
    spec = json.loads(Path(a.runs).read_text(encoding="utf-8"))
    groups = spec["groups"]
    entries = spec["entries"]
    md = []
    summary = {}
    for g in groups:
        sel = [e for e in entries if e["label"] in g.get("labels", [e["label"] for e in entries])]
        md.append(table(sel, g["cases"], g["title"]))
        if g.get("per_case"):
            md.append(per_case_table(sel, g["cases"], g.get("per_case_key", "chamfer")))
        summary[g["title"]] = {e["label"]: {k: mean_over(load(e["run"]), g["cases"], k) for k, _, _ in KEYS}
                               for e in sel if all(c in load(e["run"]) for c in g["cases"])}
        if g.get("plot"):
            plot(sel, g["cases"], ROOT / g["plot"], g["title"])
    out = ROOT / "runs" / "report"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"spec": spec, "summary": summary}, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------- 网格与插图
def route_mesh(run: str, case: str):
    """取某个 run 里某用例的预测网格，并放到真值坐标系（与评测时的对齐一致）。"""
    import trimesh  # noqa

    from .align import normalize
    from .metrics import load_mesh
    from .paths import BENCH_DATA

    if run == "gt":
        return load_mesh(BENCH_DATA / case / "gt.obj")
    d = ROOT / run / case
    rp = d / "result.json"
    if rp.exists():
        r = json.loads(rp.read_text(encoding="utf-8"))
    else:  # 合并型 run：只有顶层 result.json，per_case 里记着 source_run
        top = json.loads((ROOT / run / "result.json").read_text(encoding="utf-8")).get("per_case", {})
        if case not in top:
            return None
        r = top[case]
    if r.get("source_run"):
        d = ROOT / r["source_run"] / case
    if (d / "pred_aligned.obj").exists():
        return load_mesh(d / "pred_aligned.obj")
    src = d / "pred.obj" if (d / "pred.obj").exists() else d / "pred_raw.obj"
    if not src.exists():
        return None
    m = load_mesh(src)
    if r.get("align_transform"):
        gt = load_mesh(BENCH_DATA / case / "gt.obj")
        m = normalize(m, float((gt.bounds[1] - gt.bounds[0]).max()))
        m.apply_translation((gt.bounds[0] + gt.bounds[1]) / 2)
        m.apply_transform(np.array(r["align_transform"]))
    return m


EN_LABEL = {"gt": "GT", "D": "D hull|(3-view carving)", "C_mock": "C mock|(rule, no LLM)",
            "C_claude": "C Claude agent|(manual LLM)", "A_pixal": "A Pixal3D-MV", "A_hunyuan": "A Hunyuan3D-2mv",
            "A_trellis": "A TRELLIS.2|(front only)", "B": "B turntable|+VGGT+TSDF", "G_D": "G Qwen views|-> D hull",
            "G_A": "G Qwen views|-> Pixal3D"}


def qualitative(cases: list[str], keys: list[str], out_dir: Path) -> list[Path]:
    """每个用例一张图：行 = 路线，列 = 正/侧/顶灰模渲染（真值坐标系）。"""
    from . import blender, sheet

    spec = json.loads((ROOT / "report_runs.json").read_text(encoding="utf-8"))
    ent = {e["key"]: e for e in spec["entries"]}
    tmp = ROOT / "runs" / "report" / "qual"
    outs = []
    for c in cases:
        rows = []
        for k in ["gt"] + keys:
            run = "gt" if k == "gt" else ent[k]["run"]
            if run.startswith("__"):
                continue
            m = route_mesh(run, c)
            if m is None:
                continue
            d = tmp / c / k
            d.mkdir(parents=True, exist_ok=True)
            m.export(d / "m.obj")
            r = blender.render_ortho3(d / "m.obj", d)
            label = "真值 GT" if k == "gt" else ent[k]["label"]
            rows.append((label.replace(" ", "\n", 1), [r[v] for v in ("front", "side", "top")]))
        p = out_dir / f"qual_{c}.png"
        sheet.save(p, sheet.grid(rows, ["front", "side", "top"], size=170, label_w=150))
        outs.append(p)
    return outs
