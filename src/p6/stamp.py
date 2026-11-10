"""运行留痕：每个 run 目录必须落 _env_stamp.json / 输入 / 输出 / result.json。"""
from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .paths import ROOT


def _cmd(args: list[str]) -> str | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=20, cwd=ROOT).stdout.strip()
    except Exception:
        return None


def env_stamp(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    st: dict[str, Any] = {
        "time": _dt.datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "argv": sys.argv,
        "git_commit": _cmd(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_cmd(["git", "status", "--porcelain"])),
    }
    pk = {}
    for name in ["numpy", "torch", "trimesh", "scipy", "skimage", "cv2", "PIL"]:
        try:
            mod = __import__(name)
            pk[name] = getattr(mod, "__version__", "?")
        except Exception:
            pass
    st["packages"] = pk
    try:
        import torch  # noqa

        if torch.cuda.is_available():
            st["cuda"] = {
                "device": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "torch_cuda": torch.version.cuda,
            }
    except Exception:
        pass
    smi = _cmd(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,driver_version", "--format=csv,noheader"])
    if smi:
        st["nvidia_smi"] = smi
    if extra:
        st.update(extra)
    return st


def new_run_dir(route: str, tag: str = "") -> Path:
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    d = ROOT / "runs" / route / (f"{ts}_{tag}" if tag else ts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_default), encoding="utf-8")


def _default(o: Any):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    if isinstance(o, Path):
        return str(o)
    return str(o)


def stamp_run(run_dir: Path, extra: dict[str, Any] | None = None) -> None:
    write_json(run_dir / "_env_stamp.json", env_stamp(extra))
