"""项目路径与外部工具位置（集中在一处，方便在服务器上改）。"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
BLENDER_SCRIPTS = ROOT / "blender_scripts"
BENCH = ROOT / "bench"
BENCH_DATA = BENCH / "data"          # gitignore：生成的真值网格与三视图
RUNS = ROOT / "runs"                 # gitignore：每次运行的留痕目录
TOOLS = ROOT / "tools"               # gitignore：便携版工具
MODELS = ROOT / "models"             # gitignore：模型权重


def blender_exe() -> str:
    """优先环境变量 P6_BLENDER，其次 tools/ 里的便携版，最后 PATH。"""
    env = os.environ.get("P6_BLENDER")
    if env:
        return env
    cands = sorted((TOOLS / "blender").glob("blender-*/blender.exe")) + sorted(
        (TOOLS / "blender").glob("blender-*/blender")
    )
    if cands:
        return str(cands[-1])
    w = shutil.which("blender")
    if w:
        return w
    raise FileNotFoundError("找不到 Blender：设置 P6_BLENDER 或把便携版解压到 tools/blender/")
