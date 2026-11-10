"""ComfyUI API 格式工作流构造器（节点名与参数取自 ComfyUI v0.39.0 官方模板 + /object_info）。

对应官方模板：
  qwen21_t2i      ← image_qwen_image_2_1_t2i（去掉了 9B 提示词改写 LLM，省 9.5GB）
  qwen21_edit     ← image_qwen_image_2_1_image_edit
  hunyuan3d_mv    ← 3d_hunyuan3d_multiview_to_model
  pixal3d_mv      ← 3d_pixal3d_multi_views（只保留几何阶段，跳过纹理烘焙）
  trellis2_image  ← 3d_pixal3d_trellis2_image_to_model（TRELLIS.2 分支，几何阶段）
python -m p6.comfy.workflows 会把带占位符的版本导出到 comfy_workflows/*.api.json 供 GUI/他人复用。
"""
from __future__ import annotations

import json

from ..paths import ROOT

QWEN_UNET = "qwen_image_2.1_int8_convrot.safetensors"
QWEN_TE = "qwen3vl_8b_w4a8.safetensors"
QWEN_VAE = "qwen_image_2.1_vae_bf16.safetensors"
QWEN_ACC_LORA = "Qwen-Image-2.1-Fun-Acc-4Step.safetensors"


def _qwen_common(lora_strength: float) -> dict:
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": QWEN_UNET, "weight_dtype": "default"}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN_TE, "type": "qwen_image"}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_VAE}},
    }
    model_src = ["1", 0]
    if lora_strength:
        g["3"] = {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": ["1", 0], "lora_name": QWEN_ACC_LORA, "strength_model": lora_strength}}
        model_src = ["3", 0]
    g["2"] = {"class_type": "QwenImage21Cache", "inputs": {"model": model_src, "device": "auto", "dtype": "default"}}
    return g


def qwen21_t2i(prompt, negative="", width=1536, height=640, seed=0, steps=4, cfg=1.0, lora_strength=1.0,
               prefix="p6_qwen21_t2i") -> dict:
    g = _qwen_common(lora_strength)
    g.update({
        "5": {"class_type": "TextEncodeQwenImage21",
              "inputs": {"clip": ["4", 0], "prompt": prompt, "negative_prompt": negative, "resolution": 1024}},
        "7": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "8": {"class_type": "KSampler", "inputs": {"model": ["2", 0], "positive": ["5", 0], "negative": ["5", 1],
                                                   "latent_image": ["7", 0], "seed": seed, "steps": steps, "cfg": cfg,
                                                   "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["6", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix}},
    })
    return g


def qwen21_edit(image, prompt, negative="", width=1536, height=640, seed=0, steps=4, cfg=1.0, lora_strength=1.0,
                prefix="p6_qwen21_edit", extra_images: list[str] | None = None) -> dict:
    g = _qwen_common(lora_strength)
    enc = {"clip": ["4", 0], "prompt": prompt, "negative_prompt": negative, "resolution": 1024, "vae": ["6", 0],
           "images.image_1": ["11", 0]}
    g["11"] = {"class_type": "LoadImage", "inputs": {"image": image}}
    for k, im in enumerate(extra_images or [], start=2):
        g[f"1{k}"] = {"class_type": "LoadImage", "inputs": {"image": im}}
        enc[f"images.image_{k}"] = [f"1{k}", 0]
    g.update({
        "5": {"class_type": "TextEncodeQwenImage21", "inputs": enc},
        "7": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "8": {"class_type": "KSampler", "inputs": {"model": ["2", 0], "positive": ["5", 0], "negative": ["5", 1],
                                                   "latent_image": ["7", 0], "seed": seed, "steps": steps, "cfg": cfg,
                                                   "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["6", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix}},
    })
    return g


def hunyuan3d_mv(views: dict[str, str], seed=0, steps=20, cfg=7.5, octree=256, prefix="p6_hy3d_mv") -> dict:
    """views: {'front'|'left'|'back'|'right': 已上传的文件名}，缺的视角不连（节点允许可选）。"""
    g = {
        "54": {"class_type": "ImageOnlyCheckpointLoader", "inputs": {"ckpt_name": "hunyuan3d-dit-v2-mv_fp16.safetensors"}},
        "70": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["54", 0], "shift": 1.0}},
        "66": {"class_type": "EmptyLatentHunyuan3Dv2", "inputs": {"resolution": 3072, "batch_size": 1}},
    }
    cond = {}
    for i, (name, fn) in enumerate(views.items()):
        g[f"l{i}"] = {"class_type": "LoadImage", "inputs": {"image": fn}}
        g[f"e{i}"] = {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["54", 1], "image": [f"l{i}", 0], "crop": "none"}}
        cond[name] = [f"e{i}", 0]
    g["65"] = {"class_type": "Hunyuan3Dv2ConditioningMultiView", "inputs": cond}
    g["3"] = {"class_type": "KSampler", "inputs": {"model": ["70", 0], "positive": ["65", 0], "negative": ["65", 1],
                                                   "latent_image": ["66", 0], "seed": seed, "steps": steps, "cfg": cfg,
                                                   "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}}
    g["61"] = {"class_type": "VAEDecodeHunyuan3D", "inputs": {"samples": ["3", 0], "vae": ["54", 2], "num_chunks": 8000,
                                                              "octree_resolution": octree}}
    g["82"] = {"class_type": "VoxelToMesh", "inputs": {"voxel": ["61", 0], "algorithm": "surface net", "threshold": 0.6}}
    g["83"] = {"class_type": "SaveGLB", "inputs": {"mesh": ["82", 0], "filename_prefix": prefix}}
    return g


def _trellis_geometry(g: dict, model_node: str, pos, neg, seed=42, upsample_res=1536, prefix="p6_trellis") -> dict:
    g.update({
        "117": {"class_type": "VAELoader", "inputs": {"vae_name": "trellis_2_shape_vae_bf16.safetensors"}},
        "199": {"class_type": "CFGOverride", "inputs": {"model": [model_node, 0], "cfg": 1.0, "start_percent": 0.667, "end_percent": 1.0}},
        "125": {"class_type": "RescaleCFG", "inputs": {"model": ["199", 0], "multiplier": 0.7}},
        "108": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["125", 0], "shift": 5.0}},
        "87": {"class_type": "EmptyTrellis2LatentStructure", "inputs": {"batch_size": 1}},
        "3": {"class_type": "KSampler", "inputs": {"model": ["108", 0], "positive": pos, "negative": neg, "latent_image": ["87", 0],
                                                   "seed": seed + 14, "steps": 12, "cfg": 7.5, "sampler_name": "euler",
                                                   "scheduler": "normal", "denoise": 1.0}},
        "119": {"class_type": "VaeDecodeStructureTrellis2", "inputs": {"samples": ["3", 0], "vae": ["117", 0], "resolution": "32"}},
        "91": {"class_type": "Trellis2ShapeStage", "inputs": {"positive": pos, "negative": neg, "voxel": ["119", 0]}},
        "279": {"class_type": "CFGOverride", "inputs": {"model": [model_node, 0], "cfg": 1.0, "start_percent": 0.769, "end_percent": 1.0}},
        "126": {"class_type": "RescaleCFG", "inputs": {"model": ["279", 0], "multiplier": 0.5}},
        "18": {"class_type": "KSampler", "inputs": {"model": ["126", 0], "positive": ["91", 0], "negative": ["91", 1],
                                                    "latent_image": ["91", 2], "seed": seed, "steps": 20, "cfg": 7.5,
                                                    "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
        "94": {"class_type": "Trellis2UpsampleStage", "inputs": {"positive": ["91", 0], "negative": ["91", 1], "shape_latent": ["18", 0],
                                                                 "vae": ["117", 0], "target_resolution": upsample_res}},
        "23": {"class_type": "KSampler", "inputs": {"model": ["126", 0], "positive": ["94", 0], "negative": ["94", 1],
                                                    "latent_image": ["94", 2], "seed": seed, "steps": 12, "cfg": 7.5,
                                                    "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "92": {"class_type": "VaeDecodeShapeTrellis", "inputs": {"samples": ["23", 0], "vae": ["117", 0]}},
        "83": {"class_type": "SaveGLB", "inputs": {"mesh": ["92", 0], "filename_prefix": prefix}},
    })
    return g


def _bg_crop(g: dict, src: str, idx: str) -> list:
    """去背景 + 按掩码裁剪成 1024² 黑底（与官方模板一致）。"""
    g.setdefault("193", {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": "birefnet.safetensors"}})
    g[f"rb{idx}"] = {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["193", 0], "image": [src, 0]}}
    g[f"cm{idx}"] = {"class_type": "ImageCropToMask", "inputs": {"images": [src, 0], "masks": [f"rb{idx}", 0], "width": 1024,
                                                                "height": 1024, "pad_factor": 1.1, "grow_mask": 0,
                                                                "background": "#000000"}}
    return [f"cm{idx}", 0]


def pixal3d_mv(views: dict[str, str], fov=20.0, seed=42, upsample_res=1536, prefix="p6_pixal3d_mv",
               bg_crop: bool = True) -> dict:
    g = {
        "319": {"class_type": "UNETLoader", "inputs": {"unet_name": "pixal3d_multiview_int8_convrot.safetensors", "weight_dtype": "default"}},
        "15": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "dino_v3_L_naf_fp32.safetensors"}},
    }
    cond = {"clip_vision_model": ["15", 0], "fov": fov}
    for i, (name, fn) in enumerate(views.items()):
        g[f"l{i}"] = {"class_type": "LoadImage", "inputs": {"image": fn}}
        cond[name] = _bg_crop(g, f"l{i}", str(i)) if bg_crop else [f"l{i}", 0]
    g["324"] = {"class_type": "Pixal3DMultiViewConditioning", "inputs": cond}
    return _trellis_geometry(g, "319", ["324", 0], ["324", 1], seed, upsample_res, prefix)


def trellis2_image(image: str, seed=42, upsample_res=1536, clip_vision="dino_v3_vit_l.safetensors",
                   prefix="p6_trellis2", bg_crop: bool = True) -> dict:
    g = {
        "40": {"class_type": "UNETLoader", "inputs": {"unet_name": "trellis_2_int8_convrot.safetensors", "weight_dtype": "default"}},
        "15": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": clip_vision}},
        "l0": {"class_type": "LoadImage", "inputs": {"image": image}},
    }
    img = _bg_crop(g, "l0", "0") if bg_crop else ["l0", 0]
    g["299"] = {"class_type": "Trellis2Conditioning", "inputs": {"clip_vision_model": ["15", 0], "image": img}}
    return _trellis_geometry(g, "40", ["299", 0], ["299", 1], seed, upsample_res, prefix)


def export_templates() -> None:
    d = ROOT / "comfy_workflows"
    d.mkdir(exist_ok=True)
    tpl = {
        "qwen21_t2i.api.json": qwen21_t2i("{{prompt}}", "{{negative}}", 1536, 640, 0),
        "qwen21_edit.api.json": qwen21_edit("{{image}}", "{{prompt}}", "{{negative}}", 1536, 640, 0),
        "hunyuan3d_mv.api.json": hunyuan3d_mv({"front": "{{front}}", "left": "{{left}}", "back": "{{back}}", "right": "{{right}}"}),
        "pixal3d_mv.api.json": pixal3d_mv({"front": "{{front}}", "left": "{{left}}", "back": "{{back}}", "right": "{{right}}"}),
        "trellis2_image.api.json": trellis2_image("{{image}}"),
    }
    for name, g in tpl.items():
        (d / name).write_text(json.dumps(g, ensure_ascii=False, indent=1), encoding="utf-8")
        print("写出", d / name)


if __name__ == "__main__":
    export_templates()
