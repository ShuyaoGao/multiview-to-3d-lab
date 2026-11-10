"""MiniMax（海螺）视频生成客户端：正视图 → 360° 转台视频，供路线 B 重建。

接口（2026-10-07 依据 platform.minimax.io 官方文档核实，详见 docs/LANDSCAPE.md §3.1）：
  POST {base}/v2/video_generation
       {"model": "MiniMax-H3" | "MiniMax-H3-Max",
        "content": [{"type": "text", "text": ...},
                    {"type": "image_url", "image_url": {"url": <公网URL | mm_file://id | base64>}, "role": "first_frame"}],
        "resolution": "480P" | "768P" | "2K", "duration": 4..15, "ratio": "1:1"}
       → {"task_id": "..."}
  GET  {base}/v2/query/video_generation/{task_id}  → status=succeeded 时 content.url 为限时下载地址
  鉴权：Authorization: Bearer <MINIMAX_API_KEY>

环境变量：MINIMAX_API_KEY（必填才会真正调用）、MINIMAX_BASE_URL（默认 https://api.minimax.io；
国内账号用 platform.minimaxi.com 对应域名，未逐一核实）、MINIMAX_VIDEO_MODEL（默认 MiniMax-H3）。
没有 key 时 generate() 走 stub：只把请求体写到磁盘并返回 None，不发任何网络请求。

  python -m p6.video.minimax --image front.png --out runs/video/x [--duration 8]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path

TURNTABLE_PROMPT = (
    "Turntable product shot: the object from the reference image sits alone on a plain white background and "
    "slowly rotates a full 360 degrees around its vertical axis at constant speed. Fixed camera, fixed focal length, "
    "eye level slightly above the object, even soft studio lighting, no shadows on the background, no zoom, "
    "no camera shake, no other objects, the object's shape and proportions never change."
)


class MiniMaxVideo:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        self.key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        self.base = (base_url or os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io")).rstrip("/")
        self.model = model or os.environ.get("MINIMAX_VIDEO_MODEL", "MiniMax-H3")

    @property
    def is_stub(self) -> bool:
        return not self.key

    def _headers(self):
        return {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}

    def build_request(self, image: Path, prompt: str = TURNTABLE_PROMPT, duration: int = 8,
                      resolution: str = "768P", ratio: str = "1:1") -> dict:
        b64 = base64.standard_b64encode(Path(image).read_bytes()).decode()
        return {
            "model": self.model,
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}, "role": "first_frame"},
            ],
            "resolution": resolution,
            "duration": int(duration),
            "ratio": ratio,
            "extra": {"prompt_expansion_mode": "disabled"},
        }

    def generate(self, image: Path, out_dir: Path, poll: float = 10.0, timeout: float = 1800, **kw) -> Path | None:
        out_dir.mkdir(parents=True, exist_ok=True)
        req = self.build_request(image, **kw)
        redacted = json.loads(json.dumps(req))
        redacted["content"][1]["image_url"]["url"] = f"<base64 of {Path(image).name}>"
        (out_dir / "minimax_request.json").write_text(json.dumps(redacted, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.is_stub:
            (out_dir / "STUB.txt").write_text(
                "未设置 MINIMAX_API_KEY：只生成了请求体，没有调用 API。\n"
                "设置后重跑：set MINIMAX_API_KEY=...（PowerShell: $env:MINIMAX_API_KEY='...'）\n", encoding="utf-8")
            return None
        import requests

        r = requests.post(f"{self.base}/v2/video_generation", headers=self._headers(), json=req, timeout=120)
        r.raise_for_status()
        task = r.json()
        (out_dir / "minimax_task.json").write_text(json.dumps(task, indent=2), encoding="utf-8")
        tid = task["task_id"]
        t0 = time.time()
        while time.time() - t0 < timeout:
            q = requests.get(f"{self.base}/v2/query/video_generation/{tid}", headers=self._headers(), timeout=60).json()
            st = str(q.get("status", "")).lower()
            if st == "succeeded":
                url = q["content"]["url"]
                mp4 = out_dir / "minimax_turntable.mp4"
                mp4.write_bytes(requests.get(url, timeout=600).content)
                (out_dir / "minimax_query.json").write_text(json.dumps(q, indent=2), encoding="utf-8")
                return mp4
            if st in ("failed", "fail", "error"):
                raise RuntimeError(f"MiniMax 任务失败: {q}")
            time.sleep(poll)
        raise TimeoutError(tid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=int, default=8)
    ap.add_argument("--resolution", default="768P")
    a = ap.parse_args()
    c = MiniMaxVideo()
    p = c.generate(Path(a.image), Path(a.out), duration=a.duration, resolution=a.resolution)
    print("stub（未调用 API，请设置 MINIMAX_API_KEY）" if p is None else f"视频: {p}")


if __name__ == "__main__":
    main()
