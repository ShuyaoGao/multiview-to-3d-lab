"""模型权重/工具下载清单（可复现）。全部落到 gitignore 的 models/ 和 tools/。

  python -m p6.fetch --list
  python -m p6.fetch qwen21 hunyuan_mv pixal3d_mv trellis2 birefnet vggt
  python -m p6.fetch comfyui blender colmap

models/ 按 ComfyUI 的目录约定组织，ComfyUI 通过 extra_model_paths.yaml 直接读取这里。
所有仓库名/文件名/大小于 2026-10-07 通过 HF API 核实。许可证见 docs/LANDSCAPE.md。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from .paths import MODELS, TOOLS

# name -> list of (hf_repo, file_in_repo, comfy_subdir, size_GB, license)
HF = {
    "qwen21": [
        ("Comfy-Org/Qwen-Image-2.1", "diffusion_models/qwen_image_2.1_int8_convrot.safetensors", "diffusion_models", 7.257, "Qwen Research License（非商用）"),
        ("Comfy-Org/Qwen-Image-2.1", "text_encoders/qwen3vl_8b_w4a8.safetensors", "text_encoders", 6.312, "同上（Qwen3-VL 本体 Apache-2.0）"),
        ("Comfy-Org/Qwen-Image-2.1", "vae/qwen_image_2.1_vae_bf16.safetensors", "vae", 0.676, "同上"),
        ("alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs", "models/Qwen-Image-2.1-Fun-Acc-4Step.safetensors", "loras", 0.346, "见模型卡"),
    ],
    "hunyuan_mv": [
        ("Comfy-Org/hunyuan3D_2.0_repackaged", "split_files/hunyuan3d-dit-v2-mv_fp16.safetensors", "checkpoints", 4.928, "Tencent Hunyuan Community License（不含欧盟/英国/韩国）"),
    ],
    "pixal3d_mv": [
        ("Comfy-Org/Pixal3D", "diffusion_models/pixal3d_multiview_int8_convrot.safetensors", "diffusion_models", 5.585, "MIT（TencentARC Pixal3D）"),
        ("Comfy-Org/Pixal3D", "clip_vision/dino_v3_L_naf_fp32.safetensors", "clip_vision", 1.215, "DINOv3 License（Meta）"),
        ("Comfy-Org/Pixal3D", "vae/trellis_2_shape_vae_bf16.safetensors", "vae", 1.096, "MIT"),
        ("Comfy-Org/Pixal3D", "vae/trellis_2_texture_vae_bf16.safetensors", "vae", 0.948, "MIT"),
    ],
    "trellis2": [
        ("Comfy-Org/TRELLIS.2", "diffusion_models/trellis_2_int8_convrot.safetensors", "diffusion_models", 5.253, "MIT"),
        ("Comfy-Org/TRELLIS.2", "clip_vision/dino_v3_vit_l.safetensors", "clip_vision", 1.213, "DINOv3 License（Meta）"),
        ("Comfy-Org/TRELLIS.2", "vae/trellis_2_shape_vae_bf16.safetensors", "vae", 1.096, "MIT"),
        ("Comfy-Org/TRELLIS.2", "vae/trellis_2_texture_vae_bf16.safetensors", "vae", 0.948, "MIT"),
    ],
    "birefnet": [
        ("Comfy-Org/BiRefNet", "background_removal/birefnet.safetensors", "background_removal", 0.444, "MIT（BiRefNet）"),
    ],
    "vggt": [
        ("facebook/VGGT-1B", "model.safetensors", "vggt", 5.026, "CC-BY-NC-4.0（非商用；商用版 VGGT-1B-Commercial 需申请）"),
    ],
}

TOOL_URLS = {
    "comfyui": ("https://github.com/Comfy-Org/ComfyUI/releases/download/v0.39.0/ComfyUI_windows_portable_nvidia.7z", 2003629308),
    "blender": ("https://download.blender.org/release/Blender5.2/blender-5.2.2-windows-x64.zip", 404453484),
    "colmap": ("https://github.com/colmap/colmap/releases/download/4.2.1/colmap-x64-windows-cuda.zip", 414666626),
    "7zr": ("https://www.7-zip.org/a/7zr.exe", None),
}


def hf_meta(repo: str, f: str) -> dict:
    """HF API 给出的精确字节数与 LFS sha256（下载后据此校验）。"""
    import requests

    d = f.rsplit("/", 1)[0] if "/" in f else ""
    r = requests.get(f"https://huggingface.co/api/models/{repo}/tree/main/{d}", timeout=60)
    r.raise_for_status()
    for it in r.json():
        if it.get("path") == f:
            return {"size": it.get("size"), "sha256": (it.get("lfs") or {}).get("oid")}
    raise FileNotFoundError(f"{repo}/{f}")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_hf(name: str, verify: bool = True) -> list[dict]:
    out = []
    for repo, f, sub, gb, lic in HF[name]:
        dest = MODELS / sub / Path(f).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        meta = hf_meta(repo, f)
        for attempt in range(5):
            have = dest.stat().st_size if dest.exists() else 0
            if have == meta["size"]:
                break
            if have > meta["size"]:  # 并发写坏的文件
                dest.unlink()
            url = f"https://huggingface.co/{repo}/resolve/main/{f}"
            print(f"下载 {repo}/{f} ({gb} GB) → {dest}（第 {attempt + 1} 次）", flush=True)
            subprocess.run(["curl", "-L", "-C", "-", "--retry", "5", "-sS", "-o", str(dest), url])
        rec = {"repo": repo, "file": f, "dest": str(dest), "bytes": dest.stat().st_size,
               "expected_bytes": meta["size"], "license": lic}
        if verify and meta.get("sha256"):
            got = sha256_file(dest)
            rec["sha256"] = got
            rec["sha256_ok"] = got == meta["sha256"]
            if not rec["sha256_ok"]:
                print(f"!! sha256 不匹配：{dest.name}，删除后请重下", flush=True)
                dest.unlink()
        print(f"OK {dest.name} {rec['bytes']} B sha256_ok={rec.get('sha256_ok')}", flush=True)
        out.append(rec)
    return out


def fetch_tool(name: str) -> dict:
    url, size = TOOL_URLS[name]
    dest = TOOLS / "dl" / url.rsplit("/", 1)[1]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not (dest.exists() and (size is None or dest.stat().st_size == size)):
        subprocess.run(["curl", "-L", "-C", "-", "--retry", "5", "-sS", "-o", str(dest), url], check=True)
    h = hashlib.sha256()
    with open(dest, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return {"url": url, "dest": str(dest), "bytes": dest.stat().st_size, "sha256": h.hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list or not a.names:
        tot = 0
        for k, v in HF.items():
            s = sum(x[3] for x in v)
            tot += s
            print(f"{k:12s} {s:6.2f} GB  " + ", ".join(Path(x[1]).name for x in v))
        print(f"合计（HF，不去重）≈ {tot:.1f} GB")
        return
    log_p = MODELS / "_downloads.json"
    log = json.loads(log_p.read_text(encoding="utf-8")) if log_p.exists() else {}
    for n in a.names:
        log[n] = fetch_hf(n) if n in HF else fetch_tool(n)
        log_p.parent.mkdir(parents=True, exist_ok=True)
        log_p.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
