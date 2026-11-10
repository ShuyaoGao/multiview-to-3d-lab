"""为一键演示页 demo/index.html 收集素材 → demo/assets/（demo_data.js、loop_meshes.js、loop/*.png）。

  $env:PYTHONPATH = "src"; .venv\\Scripts\\python demo\\build_demo_assets.py

只读 runs/ 和 bench/data/ 里已有的结果，不加载任何模型、不启动 ComfyUI。
- 轮廓掩码：bench/data/<case>/mask_*.png 下采样到 N×N，页面里用 JS 现场做轮廓雕刻（visual hull）动画。
- 闭环回放：runs/C_llm_blender/<run>/<case>/iter_NN/ 的 script.py、response.md、render/*.png、diff_*.png、mesh.obj。
其余数字（各路线指标、Qwen 生成对比）直接复用 showcase/assets/data.js。
"""
from __future__ import annotations

import base64
import json
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "showcase"))

import build_assets as SB  # noqa: E402  (复用 showcase 的简化与 GLB 编码)
from p6.metrics import load_mesh  # noqa: E402

OUT = ROOT / "demo" / "assets"
N = 80  # 掩码分辨率：80×80 覆盖世界 [-1,1]²
MASK_CASES = ["bracket", "gear", "chair", "mug", "house", "spot", "figure", "blind_01", "blind_05"]
CL = "runs/C_llm_blender/"
LOOPS = [
    {"id": "claude_blind05", "run": CL + "20261007-233505_file_claude-in-loop-b2", "case": "blind_05",
     "who": "Claude 代理（手动扮演 LLM）", "result_run": CL + "merged_claude_in_loop"},
    {"id": "claude_blind01", "run": CL + "20261007-231151_file_claude-in-loop", "case": "blind_01",
     "who": "Claude 代理（手动扮演 LLM）", "result_run": CL + "merged_claude_in_loop"},
    {"id": "claude_spot", "run": CL + "20261007-231547_file_claude-in-loop-organic", "case": "spot",
     "who": "Claude 代理（手动扮演 LLM）", "result_run": CL + "merged_claude_in_loop"},
    {"id": "mock_blind05", "run": CL + "20261008-005043_mock_all16", "case": "blind_05",
     "who": "规则 mock（不是 LLM）", "result_run": CL + "20261008-005043_mock_all16"},
]
SCRIPT_MAX = 1400  # mock 的多边形坐标很长，展示时截断


def rjson(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def mask_bits(case: str, view: str) -> str:
    m = cv2.imread(str(ROOT / "bench/data" / case / f"mask_{view}.png"), cv2.IMREAD_GRAYSCALE)
    obj = (m < 128).astype(np.float32)  # save_mask：黑 = 物体
    small = cv2.resize(obj, (N, N), interpolation=cv2.INTER_AREA) > 0.5
    return "".join("1" if b else "0" for b in small.reshape(-1))  # 行优先，第 0 行 = 图像顶部


def plan_text(resp: Path) -> str:
    if not resp.exists():
        return ""
    t = resp.read_text(encoding="utf-8")
    t = t.split("```")[0].strip()
    return t[:600]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SB.OUT = OUT  # img_small 返回相对 OUT.parent（= demo/）的路径
    data: dict = {"N": N, "masks": {}, "loops": []}
    for c in MASK_CASES:
        data["masks"][c] = {v: mask_bits(c, v) for v in ("front", "side", "top")}

    meshes = {}
    tmp = ROOT / "runs" / "demo_tmp"
    for L in LOOPS:
        base = ROOT / L["run"] / L["case"]
        lj = rjson(base / "loop.json")
        pc = rjson(ROOT / L["result_run"] / "result.json")["per_case"][L["case"]]
        d = OUT / "loop" / L["id"]
        entry = {"id": L["id"], "case": L["case"], "who": L["who"], "run": L["run"],
                 "final": {k: pc.get(k) for k in ("chamfer", "fscore@0.02", "vol_iou", "sil_iou_mean", "n_iters")},
                 "target": {v: SB.img_small(base / f"target_{v}.png", d / f"target_{v}.png", 200)
                            for v in ("front", "side", "top")} if (base / "target_front.png").exists() else None,
                 "iters": []}
        for it in lj["iterations"]:
            k = it["iter"]
            idir = base / f"iter_{k:02d}"
            script = (idir / "script.py").read_text(encoding="utf-8")
            trunc = len(script) > SCRIPT_MAX
            if trunc:
                script = script[:SCRIPT_MAX].rstrip() + "\n# …（多边形坐标很长，已截断）"
            e = {"iter": k, "iou": it["sil_iou"], "px": it.get("px"),
                 "llm_seconds": it.get("llm_seconds"), "blender_seconds": (it.get("blender") or {}).get("wall_seconds"),
                 "script": script, "script_lines": script.count("\n") + 1, "plan": plan_text(idir / "response.md"),
                 "render": {v: SB.img_small(idir / "render" / f"{v}.png", d / f"it{k}_render_{v}.png", 200) for v in ("front", "side", "top")},
                 "diff": {v: SB.img_small(idir / f"diff_{v}.png", d / f"it{k}_diff_{v}.png", 200) for v in ("front", "side", "top")}}
            mo = idir / "mesh.obj"
            if mo.exists():
                m = load_mesh(mo)
                meshes[f"{L['id']}/{k}"] = SB.glb_b64(m, tmp)
                e["mesh"] = f"{L['id']}/{k}"
            entry["iters"].append(e)
            print(L["id"], k, "ok", flush=True)
        data["loops"].append(entry)
    shutil.rmtree(tmp, ignore_errors=True)

    # 图片路径：img_small 返回相对 demo/ 的 "assets/..."，页面在 demo/ 下，直接可用
    (OUT / "demo_data.js").write_text("window.P6DEMO = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    (OUT / "loop_meshes.js").write_text("window.P6LOOPMESH = " + json.dumps(meshes) + ";\n", encoding="utf-8")
    tot = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"demo/assets 合计 {tot / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
