"""ComfyUI 便携版：启动/停止 + API 模式提交工作流 + 取回产物。

  server = ComfyServer(reserve_gb=None)   # 自动按 nvidia-smi 剩余显存决定预留多少给别人
  server.start(); ...; server.stop()
  c = ComfyClient(server.url)
  name = c.upload(Path("front.png"))
  outs = c.run(api_workflow_dict, out_dir)   # 返回下载到本地的文件列表

API 工作流（comfy_workflows/*.api.json）里的占位符 "{{key}}" 由 fill() 替换。
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import requests

from ..paths import MODELS, ROOT, TOOLS

PORTABLE = TOOLS / "comfyui" / "ComfyUI_windows_portable"


def gpu_free_mb() -> tuple[int, int]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()[0]
    used, total = (int(x) for x in out.split(","))
    return total - used, total


class ComfyServer:
    def __init__(self, port: int = 8196, reserve_gb: float | None = None, budget_gb: float = 12.0,
                 extra_args: list[str] | None = None):
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.proc: subprocess.Popen | None = None
        free, total = gpu_free_mb()
        if reserve_gb is None:
            # 本项目最多用 budget_gb；其余显存留给其他进程/代理
            reserve_gb = max(2.0, total / 1024 - budget_gb)
        self.reserve_gb = round(reserve_gb, 1)
        self.gpu_free_at_start_mb = free
        self.extra_args = extra_args or []

    def _yaml(self) -> Path:
        d = ROOT / "runs" / "comfy"
        d.mkdir(parents=True, exist_ok=True)
        p = d / "extra_model_paths.abs.yaml"
        subs = ["diffusion_models", "checkpoints", "text_encoders", "vae", "loras", "clip_vision", "background_removal"]
        lines = ["p6:", f"  base_path: {MODELS.as_posix()}"] + [f"  {s}: {s}" for s in subs] + ["  clip: text_encoders"]
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return p

    def start(self, timeout: int = 300) -> None:
        if self.alive():
            return
        py = PORTABLE / "python_embeded" / "python.exe"
        main = PORTABLE / "ComfyUI" / "main.py"
        (ROOT / "runs" / "comfy").mkdir(parents=True, exist_ok=True)
        log = open(ROOT / "runs" / "comfy" / "server.log", "a", encoding="utf-8")
        args = [str(py), "-s", str(main), "--port", str(self.port), "--listen", "127.0.0.1",
                "--extra-model-paths-config", str(self._yaml()), "--reserve-vram", str(self.reserve_gb),
                "--disable-auto-launch"] + self.extra_args
        log.write(f"\n=== start {time.ctime()} {' '.join(args)}\n")
        log.flush()
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        self.proc = subprocess.Popen(args, cwd=str(PORTABLE), stdout=log, stderr=subprocess.STDOUT, env=env)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.alive():
                return
            if self.proc.poll() is not None:
                raise RuntimeError("ComfyUI 进程退出，见 runs/comfy/server.log")
            time.sleep(2)
        raise TimeoutError("ComfyUI 启动超时")

    def alive(self) -> bool:
        try:
            return requests.get(self.url + "/system_stats", timeout=2).ok
        except Exception:
            return False

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class ComfyClient:
    def __init__(self, url: str = "http://127.0.0.1:8196"):
        self.url = url
        self.cid = uuid.uuid4().hex

    def system_stats(self) -> dict:
        try:
            return requests.get(self.url + "/system_stats", timeout=10).json()
        except Exception as e:  # noqa
            return {"error": repr(e)}

    def object_info(self, node: str | None = None) -> dict:
        return requests.get(self.url + "/object_info" + (f"/{node}" if node else ""), timeout=60).json()

    def upload(self, path: Path, name: str | None = None) -> str:
        with open(path, "rb") as f:
            r = requests.post(self.url + "/upload/image", files={"image": (name or path.name, f, "image/png")},
                              data={"overwrite": "true"}, timeout=60)
        r.raise_for_status()
        j = r.json()
        return (j.get("subfolder") + "/" if j.get("subfolder") else "") + j["name"]

    def queue(self, wf: dict) -> str:
        r = requests.post(self.url + "/prompt", json={"prompt": wf, "client_id": self.cid}, timeout=60)
        if not r.ok:
            raise RuntimeError(f"提交失败 {r.status_code}: {r.text[:3000]}")
        return r.json()["prompt_id"]

    def wait(self, pid: str, timeout: int = 3600, poll: float = 2.0) -> dict:
        t0 = time.time()
        while time.time() - t0 < timeout:
            h = requests.get(self.url + f"/history/{pid}", timeout=30).json()
            if pid in h:
                st = h[pid].get("status", {})
                if st.get("completed") or st.get("status_str") in ("success", "error"):
                    return h[pid]
            time.sleep(poll)
        raise TimeoutError(pid)

    def fetch_outputs(self, hist: dict, out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        files = []
        for node_id, o in hist.get("outputs", {}).items():
            for key, items in o.items():
                if not isinstance(items, list):
                    continue
                for it in items:
                    if not isinstance(it, dict) or "filename" not in it:
                        continue
                    r = requests.get(self.url + "/view", params={"filename": it["filename"],
                                                                 "subfolder": it.get("subfolder", ""),
                                                                 "type": it.get("type", "output")}, timeout=120)
                    p = out_dir / f"{node_id}_{it['filename'].replace('/', '_')}"
                    p.write_bytes(r.content)
                    files.append(p)
        return files

    def run(self, wf: dict, out_dir: Path, timeout: int = 3600) -> tuple[list[Path], dict]:
        t0 = time.time()
        pid = self.queue(wf)
        hist = self.wait(pid, timeout)
        files = self.fetch_outputs(hist, out_dir)
        status = hist.get("status", {})
        status["seconds"] = round(time.time() - t0, 1)
        return files, status


def load_workflow(name: str) -> dict:
    return json.loads((ROOT / "comfy_workflows" / name).read_text(encoding="utf-8"))


def fill(wf: dict, **values) -> dict:
    """把工作流里值为 "{{key}}" 的字段替换成 values[key]（保留类型）。"""
    wf = copy.deepcopy(wf)
    for node in wf.values():
        if not isinstance(node, dict) or "inputs" not in node:
            continue
        for k, v in node["inputs"].items():
            if isinstance(v, str) and v.startswith("{{") and v.endswith("}}"):
                key = v[2:-2]
                if key not in values:
                    raise KeyError(f"缺少占位符取值: {key}")
                node["inputs"][k] = values[key]
    # 删除以 _ 开头的注释字段
    return {k: v for k, v in wf.items() if not k.startswith("_")}


if __name__ == "__main__":
    # 前台常驻：python -m p6.comfy.client [budget_gb]
    import sys

    s = ComfyServer(budget_gb=float(sys.argv[1]) if len(sys.argv) > 1 else 12.0)
    print(f"reserve-vram={s.reserve_gb} GB, free at start={s.gpu_free_at_start_mb} MB", flush=True)
    s.start()
    print("ComfyUI ready at", s.url, flush=True)
    try:
        s.proc.wait()
    except KeyboardInterrupt:
        s.stop()
