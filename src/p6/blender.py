"""从普通 Python 调用无头 Blender。"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from .paths import BLENDER_SCRIPTS, blender_exe


def _run(script: str, extra: list[str], timeout: int) -> subprocess.CompletedProcess:
    cmd = [blender_exe(), "-b", "--factory-startup", "--python-exit-code", "1",
           "-P", str(BLENDER_SCRIPTS / script), "--"] + extra
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")


def build_mesh(script_path: Path, out_obj: Path, timeout: int = 180) -> dict:
    """执行建模脚本 → OBJ。返回 meta（含 ok / error / tris）。"""
    meta_p = out_obj.with_suffix(".meta.json")
    if meta_p.exists():
        meta_p.unlink()
    t0 = time.time()
    try:
        cp = _run("build_mesh.py", ["--script", str(script_path), "--out", str(out_obj), "--meta", str(meta_p)], timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "stage": "timeout", "error": f"Blender 超时（>{timeout}s）"}
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {"ok": False}
    meta["wall_seconds"] = round(time.time() - t0, 2)
    if not meta.get("ok") and "error" not in meta:
        meta["error"] = (cp.stderr or cp.stdout)[-3000:]
        meta["stage"] = "blender"
    return meta


def render_ortho3(mesh_obj: Path, outdir: Path, res: int = 512, timeout: int = 300) -> dict[str, Path]:
    """渲染 front/side/top 灰模（RGBA），再合成白底 RGB 版 <view>.png。"""
    outdir.mkdir(parents=True, exist_ok=True)
    cp = _run("render_views.py", ["--mesh", str(mesh_obj), "--outdir", str(outdir), "--mode", "ortho3",
                                  "--res", str(res)], timeout)
    if cp.returncode != 0:
        raise RuntimeError(f"Blender 渲染失败:\n{cp.stderr[-2000:]}\n{cp.stdout[-2000:]}")
    out = {}
    for v in ("front", "side", "top"):
        rgba = cv2.imread(str(outdir / f"{v}_rgba.png"), cv2.IMREAD_UNCHANGED)
        a = rgba[:, :, 3:4].astype(np.float32) / 255.0
        rgb = rgba[:, :, :3].astype(np.float32) * a + 255.0 * (1 - a)
        p = outdir / f"{v}.png"
        cv2.imwrite(str(p), rgb.round().astype(np.uint8))
        out[v] = p
    return out


def render_turntable(mesh_obj: Path, outdir: Path, frames: int = 36, elev: float = 20.0, res: int = 512,
                     texture: str = "noise", engine: str = "auto", timeout: int = 1800) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    cp = _run("render_views.py", ["--mesh", str(mesh_obj), "--outdir", str(outdir), "--mode", "turntable",
                                  "--frames", str(frames), "--elev", str(elev), "--res", str(res),
                                  "--texture", texture, "--engine", engine], timeout)
    if cp.returncode != 0:
        raise RuntimeError(f"Blender 转台渲染失败:\n{cp.stderr[-2000:]}\n{cp.stdout[-2000:]}")
    return json.loads((outdir / "render_turntable.json").read_text())


def blender_version() -> str:
    cp = subprocess.run([blender_exe(), "--version"], capture_output=True, text=True, timeout=60)
    return cp.stdout.splitlines()[0] if cp.stdout else "?"


def decimate(inp: Path, out_obj: Path, faces: int = 200000, timeout: int = 1800) -> dict:
    cp = _run("decimate.py", ["--in", str(inp), "--out", str(out_obj), "--faces", str(faces)], timeout)
    if cp.returncode != 0 or not out_obj.exists():
        raise RuntimeError("Blender 简化失败: " + cp.stderr[-2000:])
    return json.loads(Path(str(out_obj) + ".json").read_text())
