# 三视图 → 3D 模型：技术版图调研（核查日 2026-10-07）

> 本文所有模型名、版本、仓库、许可证、文件大小与数字均在 2026-10-07 通过官方 GitHub / Hugging Face API 与模型卡 / 官方文档 / arXiv 核对。
> 没能从一手来源核实的内容一律标 **「未核实」** 或 **「仅第三方来源」**。文件大小取 HF API 返回的字节数，按十进制 GB 换算（1 GB = 10^9 B）。
> 核查方式说明：HF 文件清单用 `https://huggingface.co/api/models/<repo>?blobs=true`；GitHub 发布资产用 GitHub REST API；arXiv 编号用 arXiv export API 逐个核对标题。
> 核查后期检索受限，所以 Kling、Wan 2.5 及以后版本、Qwen-Image-2.0 这几项只拿到第三方来源，已在文中逐条标明。

## 0. 先说结论

1. **最初设想里的「image21」就是 Qwen-Image-2.1**（阿里 Qwen，2026-09 开源，HF 仓库 `Qwen/Qwen-Image-2.1`）。它把文生图和编辑合在一个模型里，视觉生成部分 7B，ComfyUI 已原生支持。**许可证是 Qwen Research License，仅限非商业用途**，这一点和之前 Apache-2.0 的 Qwen-Image / Qwen-Image-Edit-2509/2511/2512 不同。
2. **「MiniMax H3」确有其模型**：海螺（Hailuo）的新一代全模态视频模型 MiniMax H3，2026-07-31 发布。权重已放到 HF（`MiniMaxAI/MiniMax-H3`，采用 MiniMax H3 Community License），API 模型名为 `MiniMax-H3` 和 `MiniMax-H3-Max`。
3. **「GPT6 ultra」不存在**。核查日能查到的 GPT-6 只有 Astra、Sol、Luna 三个版本，均为 2026-09 发布（来源：Wikipedia 和 CNBC，没有 OpenAI 官网一手核实）。
4. **对本项目最关键的发现：ComfyUI 已经原生支持「多视图 → 3D」**，不用自己编译 CUDA 扩展：
   - **Hunyuan3D-2mv**：原生节点 `Hunyuan3Dv2ConditioningMultiView`，输入槽为 front/left/back/right，只出几何。
   - **Pixal3D Multi-View**：官方模板 `3d_pixal3d_multi_views`，能直接吃「一张拼好的转面图（turnaround sheet）」或四张分开的视图，输出带纹理的模型，MIT 许可。
   - **TRELLIS.2**：官方模板 `3d_pixal3d_trellis2_image_to_model`，单图输入，MIT 许可。

   这三条都绕开了 TRELLIS.2 / Hunyuan3D 官方代码在 Windows 和 Blackwell（sm_120）上的编译问题，是开发用笔记本 上最省事的路线。
5. 三视图里的**俯视图（top）**，目前的开源多视图 → 3D 模型都不接收：Hunyuan3D-2mv 和 Pixal3D-MV 只认 front/left/back/right。俯视图只能用在「视觉外壳雕刻」（路线 D）或「LLM 写 Blender/CAD 代码」（路线 C）的校验环节。

---

## 1. 三视图 / 角色转面图生成

### 1.1 Qwen-Image 家族现状（截至 2026-10-07）

| 模型 | HF 仓库 | 创建日期（HF） | 许可证 | 要点 |
|---|---|---|---|---|
| Qwen-Image | `Qwen/Qwen-Image` | 2025-08-02 | Apache-2.0 | 20B MMDiT，文生图 |
| Qwen-Image-Edit | `Qwen/Qwen-Image-Edit` | 2025-08-17 | Apache-2.0 | 编辑 |
| Qwen-Image-Edit-2509 | `Qwen/Qwen-Image-Edit-2509` | 2025-09-22 | Apache-2.0 | 多图编辑 |
| Qwen-Image-Edit-2511 | `Qwen/Qwen-Image-Edit-2511` | 2025-12-17 | Apache-2.0 | 官方卡写明：角色一致性更好、内置部分社区 LoRA、工业设计生成更强、几何推理更强，**基础模型就能直接生成新视角**（"generating new viewpoints can now be done directly with the base model"） |
| Qwen-Image-2512 | `Qwen/Qwen-Image-2512` | 2025-12-30 | Apache-2.0 | 文生图更新版 |
| Qwen-Image-Layered | `Qwen/Qwen-Image-Layered` | 2025-12-17 | — | RGBA 分层 |
| **Qwen-Image-2.1** | `Qwen/Qwen-Image-2.1` | 2026-09-14 | **Qwen Research License（非商用）**，许可证日期 2026-09-20 | 生成和编辑合一，视觉生成部分 7B（32 层 Single-Stream DiT），文本编码器为 Qwen3-VL-8B；原生 RGBA 透明图；**编辑时最多 10 张参考图**；原生 2K（1:1 为 2048×2048） |
| Qwen-Image-2.1-PE-T2I / PE-I2I | `Qwen/Qwen-Image-2.1-PE-*` | 2026-09-20 | 同上 | 提示词增强模型（Qwen3.5-9B） |
| Qwen-Image-2.0 | 无开放权重 | — | — | **仅第三方来源**：2026-02-10 发布，只提供 API（DashScope），没有开放权重 |
| Qwen Image 3.0 Pro | 无开放权重 | — | — | ComfyUI 官方模板里有 API 节点模板 `api_qwen3_t2i` / `api_qwen3_image_edit`，属于闭源 API |

Qwen-Image-2.1 许可证原文第 2 条：*"FOR NON-COMMERCIAL PURPOSES ONLY… If you wish to use the Materials commercially, you shall request a license from us at model-business@notice.qwencloud.com"*。如果成果要商用，应改用 Apache-2.0 的 Qwen-Image-Edit-2511。

### 1.2 ComfyUI 原生支持（节点、文件、大小）

- **ComfyUI 最新版**：v0.39.0（2026-10-05）。该版本的 release notes 里有 "Fix crash when selecting int8 or int4 cache in qwen image 2.1"，说明 2.1 是原生支持的。仓库已迁到 `Comfy-Org/ComfyUI`，旧地址 `comfyanonymous/ComfyUI` 会重定向过去。
- **原生节点**（`comfy_extras/nodes_qwen.py`）：`TextEncodeQwenImageEdit`、`TextEncodeQwenImageEditPlus`（2509/2511 多图编辑）、`TextEncodeQwenImage21`、`QwenImage21Cache`、`EmptyQwenImageLayeredLatentImage`。
  - 通用加载器：Load Diffusion Model、Load CLIP、Load VAE、LoraLoaderModelOnly。
  - 文生图：EmptySD3LatentImage + KSampler（见官方教程）。
- Qwen-Image 最早是哪个 ComfyUI 版本开始原生支持的：**未核实**（官方教程只写了"请更新到最新/Nightly"），2025-08 前后就已有支持。

**A. Qwen-Image-2.1（最新，非商用）**：`Comfy-Org/Qwen-Image-2.1`

| 用途 | 文件（放到 `ComfyUI/models/…`） | 大小 |
|---|---|---|
| 扩散模型 | `diffusion_models/qwen_image_2.1_bf16.safetensors` | 14.23 GB |
| 扩散模型（int8） | `diffusion_models/qwen_image_2.1_int8_convrot.safetensors` | **7.26 GB** |
| 文本编码器 | `text_encoders/qwen3vl_8b_bf16.safetensors` | 17.53 GB |
| 文本编码器 | `text_encoders/qwen3vl_8b_int8_convrot.safetensors` | 9.35 GB |
| 文本编码器（最小） | `text_encoders/qwen3vl_8b_w4a8.safetensors` | **6.31 GB** |
| VAE | `vae/qwen_image_2.1_vae_bf16.safetensors` | 0.68 GB |
| ControlNet（可选） | `model_patches/qwen_image_2.1_fun_controlnet_union_int8_convrot.safetensors` | 3.78 GB |
| 提示词增强（可选） | `text_encoders/qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors` / `..._pe_i2i...` | 各 9.47 GB |

- 官方模板：`image_qwen_image_2_1_t2i`、`image_qwen_image_2_1_image_edit`、`image_qwen_image_2_1_background_removal`。
- 加速 LoRA：`alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs` → `models/Qwen-Image-2.1-Fun-Acc-4Step.safetensors`（0.35 GB，HF 许可字段为 other）。截至核查日，lightx2v 还没有发布 2.1 的 Lightning 版本（HF 上搜不到）。
- GGUF：`unsloth/Qwen-Image-2.1-GGUF`，例如 Q4_K_M 4.20 GB、Q5_K_M 5.39 GB、Q8_0 7.64 GB。模型卡推荐的运行方式是 stable-diffusion.cpp / Unsloth Desktop，配套的文本编码器是 `unsloth/Qwen3-VL-8B-Instruct-GGUF` 的 `Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf`（卡上写 5.15 GB）。**city96/ComfyUI-GGUF 能否加载 Qwen-Image-2.1 的 GGUF：未核实**（该插件最后一次提交是 2026-01-12，早于 2.1 发布）。在 ComfyUI 里建议直接用官方的 int8_convrot 文件。

**B. Qwen-Image / 2512（Apache-2.0）**：`Comfy-Org/Qwen-Image_ComfyUI`

| 文件 | 大小 |
|---|---|
| `split_files/diffusion_models/qwen_image_bf16.safetensors` | 40.86 GB |
| `split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors` | 20.43 GB |
| `split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors` | 20.43 GB |
| `split_files/diffusion_models/qwen_image_nvfp4.safetensors` | 19.77 GB |
| `split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors` | **9.39 GB** |
| `split_files/text_encoders/qwen_2.5_vl_7b_nvfp4.safetensors` | 6.11 GB |
| `split_files/text_encoders/qwen_2.5_vl_7b.safetensors` | 16.58 GB |
| `split_files/vae/qwen_image_vae.safetensors` | **0.25 GB** |

GGUF：`city96/Qwen-Image-gguf`（Apache-2.0），可选 Q2_K 7.06 / Q3_K_M 9.68 / Q4_K_S 12.14 / Q4_K_M 13.07 / Q5_K_M 14.94 / Q8_0 21.76 GB；2512 版另有 `unsloth/Qwen-Image-2512-GGUF`。

**C. Qwen-Image-Edit 2509 / 2511（Apache-2.0）**：`Comfy-Org/Qwen-Image-Edit_ComfyUI`

| 文件 | 大小 |
|---|---|
| `split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors` | 20.53 GB |
| `split_files/diffusion_models/qwen_image_edit_2511_int8_convrot.safetensors` | 20.50 GB |
| `split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors` | 20.43 GB |
| `split_files/loras/Qwen-Edit-2509-Multiple-angles.safetensors` | 0.24 GB |

文本编码器和 VAE 与 B 共用（qwen_2.5_vl_7b + qwen_image_vae）。
GGUF：`unsloth/Qwen-Image-Edit-2511-GGUF`（Apache-2.0），可选 Q2_K 7.47 / Q3_K_M 9.92 / Q4_K_S 12.41 / Q4_K_M 13.25 / Q5_K_M 15.03 / Q8_0 21.76 GB；`QuantStack/Qwen-Image-Edit-2509-GGUF` 的 Q4_K_M 为 13.07 GB。

**Lightning 少步 LoRA（lightx2v，Apache-2.0）**
- `lightx2v/Qwen-Image-Lightning`：`Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors`、`Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors`（各 0.85 GB）；子目录 `Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors`（0.85 GB）。
- `lightx2v/Qwen-Image-Edit-2511-Lightning`：`Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` / `...-8steps-V1.0-bf16.safetensors`（各 0.85 GB）。
- `lightx2v/Qwen-Image-2512-Lightning`：存在，文件清单未逐一列出。
- 官方教程说明：distill 版模型和 Lightning LoRA "do not seem to be compatible for simultaneous use"（两者似乎不能同时用）。

**显存参考（官方教程实测）**：RTX 4090D 24GB 上跑 fp8 版 Qwen-Image，显存占用约 86%，首张约 94 s；加 8 步 Lightning 后约 55 s。开发用笔记本 只有约 12GB 可用显存，**单靠显卡放不下 20B 的 fp8 模型**，只能用 GGUF Q3/Q4 加内存卸载，或者改用 7B 的 Qwen-Image-2.1 int8（7.26 GB）。笔记本上的实际速度：**未核实**。

### 1.3 多视角 / 转面图专用 LoRA（均已核实存在）

| 仓库 | 基座 | 许可 | 用法（来自模型卡） |
|---|---|---|---|
| `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA`，文件 `qwen-image-edit-2511-multiple-angles-lora.safetensors`（0.30 GB） | Edit-2511 | Apache-2.0 | 96 个相机位（4 种仰角 × 8 个方位角 × 3 种距离），训练数据是 3000+ 组高斯泼溅渲染对。提示词格式 `<sks> [azimuth] [elevation] [distance]`，例如 `<sks> front view eye-level shot medium shot`、`<sks> right side view ...`、`<sks> back view ...` |
| `dx8152/Qwen-Edit-2509-Multiple-angles`（`镜头转换.safetensors`，0.24 GB；Comfy-Org 另有重打包版） | Edit-2509 | Apache-2.0 | 无触发词，直接写中文指令，如「将镜头向左旋转45度」「将镜头转为俯视」；卡上建议配合 Lightning LoRA 使用 |
| `tarn59/character_turnaround_sheet_qwen_edit_2511`（0.30 GB） | Edit-2511 | Apache-2.0 | 触发词 `Character turnaround sheet`；推荐输入 600×1080、输出 2600×1080 或 3000×1080。作者自述**出图人物偏窄，横向拉宽约 20% 效果更好**（对比例一致性有直接影响） |
| `tarn59/character_turnaround_sheet_qwen_image_edit_2509` | Edit-2509 | — | 同一作者的 2509 版本 |

### 1.4 怎样得到「正交、比例一致、视角一致」的三视图

- **推荐流程**：先画/生成一张正面定稿图，再用 Edit-2511 加 Multiple-Angles LoRA，逐张生成 `front / right side / back / left side` 四个视角（俯视用 `high-angle` 只能近似，**不是真正的正交俯视**）。也可以用 turnaround LoRA 一次出整张转面图，然后横向拉宽约 20%，再切成单张。
- **提示词模板**（经验写法，**没有做过基准验证**）：
  `character turnaround sheet, orthographic projection, no perspective, front view | left side view | back view | right side view, same height, feet on the same baseline, neutral A-pose, plain white background, even flat lighting, full body, centered`
  中文写法：「正交投影、无透视、同一比例、脚底对齐同一水平线、A字姿势、纯白背景、均匀平光」。
- **一致性自检**（建议脚本化）：把四张图抠图后比较外轮廓的高度和宽度。正面宽度应等于背面宽度，左视宽度应等于右视宽度，四张高度应一致。轮廓误差超过阈值就重新生成。这一步也给路线 D 的雕刻提供了现成的输入。
- **专用多视图扩散模型**（开源，均已核实）：

| 模型 | arXiv | 代码许可 | 权重许可 | 备注 |
|---|---|---|---|---|
| MV-Adapter | 2412.03632 | Apache-2.0 | Apache-2.0（`huanngzh/mv-adapter`） | 基于 SDXL，768 分辨率；另有 SD2.1 版本（<10G 显存）；发布了正交视角数据集 Objaverse-Ortho10View |
| Era3D | 2405.11616 | AGPL-3.0 | Apache-2.0（`pengHTYX/MacLab-Era3D-512-6view`） | 6 视角，512 分辨率 |
| Zero123++ | 2310.15110 | Apache-2.0 | **CC-BY-NC 4.0** | 6 视角 |
| SV3D | 2403.12008 | — | `sv3d-nc-community`（非商用社区许可，HF 需申请） | 基于视频扩散做环绕视角 |
| Hunyuan3D 多视图 | — | — | — | Hunyuan3D-2mv 是「多视图 → 形状」模型，**不生成多视图图像**，见 §2 |

### 来源（§1）
- https://huggingface.co/Qwen/Qwen-Image-2.1 （模型卡、LICENSE）
- https://huggingface.co/Comfy-Org/Qwen-Image-2.1 ， https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI ， https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI
- https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF ， https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF ， https://huggingface.co/city96/Qwen-Image-gguf ， https://huggingface.co/QuantStack/Qwen-Image-Edit-2509-GGUF
- https://huggingface.co/lightx2v/Qwen-Image-Lightning ， https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning ， https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs
- https://huggingface.co/Qwen/Qwen-Image-Edit-2511
- https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA ， https://huggingface.co/dx8152/Qwen-Edit-2509-Multiple-angles ， https://huggingface.co/tarn59/character_turnaround_sheet_qwen_edit_2511
- https://docs.comfy.org/tutorials/image/qwen/qwen-image
- https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.39.0 ， https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_qwen.py
- https://github.com/Comfy-Org/workflow_templates （templates/index.json）
- https://github.com/huanngzh/MV-Adapter ， https://github.com/pengHTYX/Era3D ， https://github.com/SUDO-AI-3D/zero123plus ， https://huggingface.co/stabilityai/sv3d
- Qwen-Image-2.0（仅第三方来源）：https://gigazine.net/gsc_news/en/20260212-qwen-image-2/ ， https://wavespeed.ai/blog/posts/blog-what-is-qwen-image-2-0-features-benchmarks/

---

## 2. 原生 图像 / 多视图 → 3D

### 2.1 对比总表

| 模型 | 输入 | 输出 | 参数 / 权重大小 | 显存（官方数字） | 许可证 | Windows | 备注 |
|---|---|---|---|---|---|---|---|
| **Hunyuan3D-2mv** | **多视图：front / left / back / right**（任选，至少一张；**没有 top**） | 白模几何 | 1.1B；`model.fp16.safetensors` 4.93 GB（标准 / fast / turbo 三种） | 仅几何约 6GB（2.0 系列）；ComfyUI 文档写几何 6GB、2mini 5GB | Tencent Hunyuan 3D 2.0 Community License（见 §2.3） | ComfyUI 原生可用；官方仓库只做几何时**不需编译** | 源码 `preprocessors.py`：`view2idx = {'front':0,'left':1,'back':2,'right':3}`，注释写 "view order: front, front clockwise 90, back, front clockwise 270" |
| Hunyuan3D-2mini | 单图 | 几何 | 0.6B；3.82 GB | ComfyUI 文档：5GB | 同上 | 同上 | 有 turbo / fast 变体 |
| Hunyuan3D-2 / 2.0 | 单图（+ Paint 纹理） | 几何 + 纹理 | DiT 1.1B，Paint 1.3B | 几何 6GB，几何加纹理 16GB | 同上 | 纹理需要编译 `custom_rasterizer`、`differentiable_renderer` | 社区有 Windows 整合包 |
| Hunyuan3D-2.1 | 单图 | 几何 + PBR | 形状 3.0B（README 表）/ 3.3B（报道）；`hunyuan3d-dit-v2-1/model.fp16.ckpt` 7.37 GB | 几何 10GB，纹理 21GB，合计 29GB | Tencent Hunyuan 社区许可（HF 许可字段为 other；2.1 版条款本轮未逐条核对，按 2.0 的地域限制处理） | 官方在 Python 3.10 + torch 2.5.1+cu124 下测试；ComfyUI 原生有模板 `3d_hunyuan3d-v2.1` | 训练代码已开源 |
| Hunyuan3D-Omni | 图像 + 点云 / 体素 / 包围盒 / 骨架等控制 | 几何 | — | 未核实 | Tencent Hunyuan Community | 未核实 | 仓库描述："Unified Framework for Controllable Generation of 3D Assets" |
| Hunyuan3D 2.5 / 3.0 / 3.1 | 3.1 支持最多 8 张输入视图（腾讯官方 X 账号） | — | — | — | **闭源，只有 API** | — | 2.5 有技术报告（arXiv 2506.16504）；版本时间线来自 Tencent Cloud techpedia |
| Hunyuan3D-Buffalo 1.0 | 生成、理解、编辑一体 | — | — | — | — | — | 截至核查日只有论文（arXiv 2608.02711）和项目页，**没有权重** |
| **TRELLIS.2-4B** | 单图 | 带 PBR 纹理的网格（O-Voxel） | 4B；HF 权重分为多个 1.3B DiT 和 VAE | README 要求 ≥24GB，在 A100/H100 上验证过；H100 上 1024³ 约 17 s | 代码和权重 MIT；**依赖 `facebook/dinov3-vitl16-pretrain-lvd1689m`（DINOv3 许可）和 `briaai/RMBG-2.0`（许可证本轮未核实）** | 官方**只在 Linux 测试过**；需编译 flash-attn、nvdiffrast、nvdiffrec、cumesh、o-voxel、flexgemm；默认 torch 2.6.0 + cu124 | **ComfyUI 原生**：`Comfy-Org/TRELLIS.2` 的 int8 版 5.25 GB |
| TRELLIS（v1） | 单图；多图条件是 tuning-free 的实验功能 | 网格 / 3DGS / RF | 最大 2B | ≥16GB | MIT | 仅 Linux 测试，Windows 参见 issue #3 | — |
| **Pixal3D**（TencentARC） | 单图（像素对齐）+ **Multi-View 变体（front/back/left/right，可直接输入拼好的转面图）** | 带纹理网格 | `pixal3d_bf16` 11.0 GB / `int8_convrot` 5.59 GB（单图和多视图各一套） | 有官方 `--low_vram` 模式；具体显存数**未核实** | MIT（DINOv3 编码器另有其许可） | **ComfyUI 原生模板**：`3d_pixal3d_multi_views`、`3d_pixal3d_trellis2_image_to_model` | arXiv 2605.10922；复用 TRELLIS.2 的形状和纹理 VAE |
| Step1X-3D | 单图 | 几何 + 纹理 | 几何 1.3B | 几何加纹理 27GB（带 label 版 29GB），50 步 152 s | Apache-2.0 | 未核实 | 公开了 80 万条高质量资产的 uid |
| TripoSG | 单图（另有 scribble 变体） | 几何 | 1.5B | ≥8GB | MIT | 未核实 | 依赖 RMBG-1.4 |
| TripoSplat | 单图 | 3D 高斯 | DiT fp16 0.74 GB | 未核实 | MIT | ComfyUI 原生模板 | arXiv 2605.16355（只在 HF 卡上看到引用，本轮没有单独核对） |
| SPAR3D | 单图 | 网格 | — | 默认 10.5GB，低显存模式约 7GB | Stability AI Community License | Windows 为**实验性**支持，需 VS2022 | — |
| Hi3DGen（仓库已更名为 `Stable-X/Stable3DGen`） | 单图，经法线做桥接 | 几何 | — | 未核实 | MIT（README 说明去掉了 kaolin、nvdiffrast、flexicubes 依赖，以便商用） | 依赖较少 | arXiv 2503.22236 |
| Direct3D-S2 | 单图 | 几何（1024³ 稀疏 SDF） | — | 512 分辨率 ≥10GB，1024 分辨率约 24GB | MIT | 社区有 Windows 方案（issue #11、#12） | README 称 8 张 GPU 就能训练 1024³ |
| Sparc3D | — | — | — | — | 仓库无 LICENSE | — | **只有论文（arXiv 2505.14521）、项目页和 HF Demo，没看到代码和权重** |
| UniLat3D | 单图，单阶段 | 3DGS / 网格 | — | — | 未核实 | — | arXiv 2509.25079；HF 有 `UniLat3D/UniLat3D`，可用性未核实 |
| SAM 3D Objects（Meta） | 单图 | 3D | ss_generator 6.69 GB + slat_generator 4.91 GB | 未核实 | HF 许可字段为 other，需手动申请 | 未核实 | arXiv 2511.16624 |

### 2.2 ComfyUI 原生 3D 工作流（无需编译，推荐在开发用笔记本 上使用）

- **Hunyuan3D-2mv**（模板 `3d_hunyuan3d_multiview_to_model`，另有 `_turbo` 版）
  - 节点链：`ImageOnlyCheckpointLoader` → `CLIPVisionEncode` ×N → `Hunyuan3Dv2ConditioningMultiView`（可选输入 front / left / back / right）→ `EmptyLatentHunyuan3Dv2` → `ModelSamplingAuraFlow` → `KSampler` → `VAEDecodeHunyuan3D` → `VoxelToMesh` → `SaveGLB`。
  - 模型：`https://huggingface.co/Comfy-Org/hunyuan3D_2.0_repackaged/resolve/main/split_files/hunyuan3d-dit-v2-mv_fp16.safetensors`（4.93 GB；turbo 版 4.93 GB）。
  - **放置位置有出入**：模板和官方文档写的是 `models/checkpoints/`，Comfy-Org 仓库 README 写的是 `models/diffusion_models/`。按模板来，放 `checkpoints`。
  - ComfyUI 文档说明：只有正面一张图也能跑；ComfyUI 原生**不做纹理**。
- **Pixal3D Multi-View**（模板 `3d_pixal3d_multi_views`）。模板描述："Upload your own separate view images, or use one mixed multi-view turnaround sheet. The workflow prepares front, back, left and right views, then reconstructs a textured 3D model"。模型 URL：
  - `Comfy-Org/Pixal3D/.../diffusion_models/pixal3d_multiview_int8_convrot.safetensors`（5.59 GB）
  - `clip_vision/dino_v3_L_naf_fp32.safetensors`（1.22 GB）
  - `vae/trellis_2_shape_vae_bf16.safetensors`（1.10 GB）
  - `vae/trellis_2_texture_vae_bf16.safetensors`（0.95 GB）
  - `Comfy-Org/BiRefNet/.../background_removal/birefnet.safetensors`（0.44 GB）
  - 模板自带后处理节点：重新网格化、减面、UV 展开、贴图烘焙（`DecimateMesh`、`BakeTextureFromVoxel`、`BakeNormalMapFromMesh` 等）。
- **TRELLIS.2 / Pixal3D 单图**（模板 `3d_pixal3d_trellis2_image_to_model`）：在上面基础上加 `Comfy-Org/TRELLIS.2/.../trellis_2_int8_convrot.safetensors`（5.25 GB）和 `Comfy-Org/MoGe/.../moge_2_vitl_normal_fp16.safetensors`（0.66 GB）。
- 这几条原生路线**在约 12GB 显存的笔记本上能否跑通、要多久，都未实测**。
- 社区替代方案：`visualbruno/ComfyUI-Trellis2`（MIT，提供 Windows 预编译 wheel，目录为 Torch270 / Torch280 / Torch2100；README 写明 2026-02-27 加入了 "Python 3.13, Torch 2.10.0, CUDA 13.1" 的 wheel）；`PozzettiAndrea/ComfyUI-TRELLIS2`；`kijai/ComfyUI-Hunyuan3DWrapper`；`MrForExample/ComfyUI-3D-Pack`。

### 2.3 许可证注意：Tencent Hunyuan 3D 社区许可

LICENSE 原文："THIS LICENSE AGREEMENT DOES NOT APPLY IN THE EUROPEAN UNION, UNITED KINGDOM AND SOUTH KOREA"，Territory 定义为全球范围但不含欧盟、英国、韩国；条款还写明不得在授权地域以外使用模型、**其输出或结果**。月活超过 100 万的需要单独申请。中国境内的科研使用不受地域条款限制，但论文或产物如果要在欧盟、英国、韩国分发，需要注意这一点。Comfy-Org 重打包仓库也写了 "NOT use in EU, UK, or South Korea"。

### 2.4 Windows 与 Blackwell（sm_120）兼容性

- RTX 5090 Laptop 属于 Blackwell 架构（sm_120），需要 CUDA 12.8 及以上编译的 PyTorch。Hunyuan3D-2.1 官方测试环境是 torch 2.5.1+cu124，TRELLIS.2 默认 torch 2.6.0+cu124。**按官方默认环境安装，大概率在 5090 上跑不了**（cu124 构建不含 sm_120 内核；这是推断，没有在本机实测）。
- `YanWenKun/Hunyuan3D-2-WinPortable`（GPL-3.0，支持 2.0 和 2.1）：README 兼容表把 **Blackwell RTX 5090 对应到 CUDA 12.9 版整合包**；几何生成最低约 3GB 显存（2.1 开最大优化），纹理约 6GB（2.0），系统内存 ≥24GB。
- ComfyUI 便携版自带 **Python 3.13 和 PyTorch cu130**，README 写 "supports 20 series and above"，所以 RTX 50 系能用。原生 3D 节点不需要编译扩展，**在 Windows + 5090 上是阻力最小的路线**。
- Hunyuan3D-2 只做几何：`pip install -r requirements.txt && pip install -e .` 即可，包名 `hy3dgen`，`setup.py` 里版本为 2.0.2。依赖包括 diffusers、transformers≥4.48、trimesh、pymeshlab、xatlas、pygltflib 等。**只有纹理部分**才需要编译 `hy3dgen/texgen/custom_rasterizer` 和 `differentiable_renderer`。`hy3dgen` 是否发布到 PyPI：**未核实**，建议从仓库源码安装。

### 来源（§2）
- https://github.com/Tencent-Hunyuan/Hunyuan3D-2 （README、`examples/shape_gen_multiview.py`、`hy3dgen/shapegen/preprocessors.py`、`setup.py`、LICENSE）
- https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 ； https://huggingface.co/tencent/Hunyuan3D-2mv ， https://huggingface.co/tencent/Hunyuan3D-2mini ， https://huggingface.co/tencent/Hunyuan3D-2.1
- https://huggingface.co/Comfy-Org/hunyuan3D_2.0_repackaged ， https://huggingface.co/Comfy-Org/hunyuan3D_2.1_repackaged ， https://docs.comfy.org/tutorials/3d/hunyuan3D-2
- https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_hunyuan3d.py
- https://github.com/Tencent-Hunyuan/Hunyuan3D-Omni ， https://github.com/Tencent-Hunyuan/Hunyuan3D-Buffalo1.0 （arXiv 2608.02711）
- Hunyuan3D 3.x 只有 API：https://www.tencentcloud.com/techpedia/148273?lang=en ， https://x.com/TencentHunyuan/status/2016449283428659599
- https://github.com/microsoft/TRELLIS.2 ， https://huggingface.co/microsoft/TRELLIS.2-4B （pipeline.json），arXiv 2512.14692 ； https://github.com/microsoft/TRELLIS （arXiv 2412.01506）
- https://huggingface.co/TencentARC/Pixal3D ， https://huggingface.co/Comfy-Org/Pixal3D ， https://huggingface.co/Comfy-Org/TRELLIS.2 （arXiv 2605.10922）
- https://github.com/stepfun-ai/Step1X-3D （arXiv 2505.07747）； https://github.com/VAST-AI-Research/TripoSG （arXiv 2502.06608）； https://huggingface.co/VAST-AI/TripoSplat
- https://github.com/Stability-AI/stable-point-aware-3d （arXiv 2501.04689）； https://github.com/Stable-X/Stable3DGen （arXiv 2503.22236）
- https://github.com/DreamTechAI/Direct3D-S2 （arXiv 2505.17412）； https://github.com/lizhihao6/Sparc3D （arXiv 2505.14521）； arXiv 2509.25079（UniLat3D）； https://huggingface.co/facebook/sam-3d-objects （arXiv 2511.16624）
- https://github.com/YanWenKun/Hunyuan3D-2-WinPortable ， https://github.com/visualbruno/ComfyUI-Trellis2

---

## 3. 视频路线（环绕视频 → 重建）

### 3.1 海螺 / MiniMax

- **MiniMax H3（海螺 3.0）**：2026-07-31 发布，全模态输入（文本、图像、视频、音频），输出带 32kHz 立体声的视频，时长 4–15 s，24fps，默认短边 768，用 H3-Regenerate-2K 可以出 2K。
  - 模型结构（HF 卡）：H3-Omni-Transformer 为 33B 单流 dense Transformer，其中约 13B 在 AdaLN 分支，纯推理部署时可以不加载；编码器用 Qwen3-VL-32B 第 50 层的隐状态。
  - 变体：H3-Base-FL2VA（首尾帧模式，0–2 张图）、H3-Base-Ref2VA（全参考模式，≤9 张图、≤3 段视频、≤3 段音频）。
  - 开放权重：`MiniMaxAI/MiniMax-H3`，**MiniMax H3 Community License**；HF 卡上有一个 "Application form (only for USA/EU/UK/South Korea)"，这几个地区需要另外申请。H3-Context-IR 是托管服务，不开源。
  - ComfyUI 已原生支持，有官方模板 `video_minimax_h3_i2v` 等。`Comfy-Org/MiniMax-H3` 里最小的扩散模型是 `minimax_h3_fl2va_pruned_w6a8.safetensors`（15.98 GB），另有 4 步 turbo LoRA（1.96 GB）。**笔记本上本地跑不现实**；在服务器 2×3090 上能否运行：**未核实**。
- **API**（官方文档 platform.minimax.io）

| 项目 | 内容 |
|---|---|
| Base URL | `https://api.minimax.io`（国内为 `https://api.minimaxi.com` 一侧，见 platform.minimaxi.com，**具体域名未逐一核实**） |
| 鉴权 | `Authorization: Bearer <API_KEY>` |
| H3 创建任务 | `POST /v2/video_generation`；必填 `model`（`MiniMax-H3` / `MiniMax-H3-Max`）、`content`（多模态数组，必须含文本）、`resolution`（`480P` / `768P` / `2K`，取决于模型）、`duration`（4–15）；可选 `ratio`（默认 `adaptive`）、`extra`、`callback_url` |
| H3 查询任务 | `GET /v2/query/video_generation/{task_id}`；成功时 `status` 为 `succeeded`，`content.url` 是**有时效的**下载地址；只能查询最近 7 天的任务 |
| 旧版（海螺 02 / 2.3）创建 | `POST /v1/video_generation`，模型可选 `MiniMax-Hailuo-2.3`、`MiniMax-Hailuo-02`、`T2V-01-Director`、`T2V-01`；分辨率 720P / 768P / 1080P；时长 6 s，海螺模型在 768P 下可选 10 s |
| 旧版查询 | `GET /v1/query/video_generation?task_id=...`，状态有 Preparing / Queueing / Processing / Success / Fail，成功后返回 `file_id` |
| 旧版下载 | `GET /v1/files/retrieve?file_id=...`，返回 `file.download_url` |
| 价格（按量付费页） | MiniMax-H3：768P $0.08/s，2K $0.13/s；MiniMax-H3-Max：480P $0.05/s，768P $0.08/s；Hailuo-2.3：768P 6s $0.28、10s $0.56、1080P 6s $0.49；Hailuo-2.3-Fast：768P 6s $0.19、10s $0.32、1080P 6s $0.33；Hailuo-02：与 2.3 相同，另有 512P 6s $0.10、10s $0.15 |

### 3.2 其他视频模型

- **Wan 2.2（开放权重，Apache-2.0）**：`Wan-AI/Wan2.2-TI2V-5B`（720P@24fps，官方称单张 4090 级消费卡就能跑）、`Wan2.2-I2V-A14B`、`Wan2.2-T2V-A14B`、`Wan2.2-Animate-14B`、`Wan2.2-S2V-14B`，以及 2026-07 发布的 `Wan2.2-Animate-2-14B`。ComfyUI 有原生模板，包括 `video_wan2_2_14B_fun_camera`（相机运动控制，**可以用来做环绕运镜**）和 `video_wan2.1_fun_camera_v1.1_*`。
- **Wan 2.5 / 2.6 / 2.7 / 3.0**：ComfyUI 模板里有 `api_wan2_6_i2v`、`api_wan2_7_i2v` 等 API 节点，说明这些版本是闭源 API。「最后一个开放权重的版本是 2.2，3.0 于 2026-08 以 API 形式内测」这一说法**仅第三方来源**（howaiworks.ai、wan27.org）。
- **Kling（可灵）**：**仅第三方来源**。Kling 3.0 于 2026-02-05 发布，Kling 3.0 Turbo / Omni 于 2026-06-17 发布。官方 API 文档页需要 JS 渲染，本轮没有抓到，端点和模型名**未核实**。
- **生成环绕视频**的思路：把正面定稿图作为首帧，提示词写 "camera orbits 360° around the static object, turntable, constant distance, neutral lighting"，或用 Wan 2.2 Fun Camera 的相机轨迹控制。视频模型**不保证几何严格一致**（物体可能变形、细节漂移），这是路线 B 的根本风险。这里是经验判断，**没有量化验证**。

### 3.3 由视频重建

| 工具 | 最新版本 / 状态 | 许可 | Windows | 备注 |
|---|---|---|---|---|
| **COLMAP** | **4.2.1（2026-09-29）** | 新 BSD | **有官方 CUDA 版二进制** `colmap-x64-windows-cuda.zip`（414,666,626 B ≈ 415 MB）；nocuda 版 127,731,595 B | 4.x 已内置 GLOMAP，作为 "global" mapper；4.2.0 加入 LoMa 学习特征（ONNX） |
| GLOMAP | 1.2.0（2025-10-31），**已停止维护** | — | 有独立 Windows 包 | README 写明已"fully migrated to COLMAP" |
| nerfstudio | 最后一次推送 2025-07-29 | Apache-2.0 | — | 活跃度下降 |
| gsplat | 活跃（2026-09 有推送） | Apache-2.0 | — | 3DGS 光栅化库 |
| 3DGS（graphdeco-inria） | — | Inria 研究许可（非商用） | — | — |
| 2DGS | 活跃（2026-08） | 研究用途许可 | — | 几何更准，适合提取网格 |
| SuGaR | 最后推送 2024-09 | 研究用途许可（同 Inria 类） | — | 高斯 → 网格 |
| **VGGT** | VGGT-1B；后继 **VGGT-Omega**（2026-05-18 发布） | 代码已改为允许商用（军事用途除外）；**`facebook/VGGT-1B` 权重为 CC-BY-NC-4.0**；**`facebook/VGGT-1B-Commercial` 可商用，需填表申请（HF gated）**；VGGT-Omega 同样需申请 | 纯 PyTorch，**不需要编译** | 权重 5.03 GB；可导出 COLMAP 格式，直接接 gsplat；2026-05-15 修复了显存冗余，同样显存下能处理多 2–3 倍帧数 |
| DUSt3R / MASt3R | — | **CC BY-NC-SA 4.0** | — | 不需要相机位姿 |
| π³（Pi3） | — | GitHub 仓库为 BSD-3-Clause；HF `yyfz233/Pi3` 卡上写 bsd-2-clause；**`yyfz233/Pi3X` 为 CC-BY-NC-4.0** | — | arXiv 2507.13347 |
| MapAnything | — | 代码 Apache-2.0；**`facebook/map-anything` 为 CC-BY-NC-4.0，`facebook/map-anything-apache` 为 Apache-2.0** | — | arXiv 2509.13414，度量尺度的前馈重建 |
| HunyuanWorld-Mirror（WorldMirror） | — | — | — | ICML 2026，见腾讯混元仓库，细节未核实 |

推荐管线：环绕视频抽帧 → 抠图（BiRefNet）→ VGGT 或 COLMAP global mapper 估计位姿 → gsplat / 2DGS 训练 → TSDF 或 2DGS 提取网格 → Blender 清理。

### 来源（§3）
- https://huggingface.co/MiniMaxAI/MiniMax-H3 ， https://www.minimax.io/blog/minimax-h3 ， https://huggingface.co/Comfy-Org/MiniMax-H3
- https://platform.minimax.io/docs/api-reference/video-generation-v2-create ， https://platform.minimax.io/docs/api-reference/video-generation-v2-query ， https://platform.minimax.io/docs/api-reference/video-generation-t2v ， https://platform.minimax.io/docs/api-reference/video-generation-query ， https://platform.minimax.io/docs/api-reference/file-management-retrieve ， https://platform.minimax.io/docs/guides/pricing-paygo
- https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B ， https://huggingface.co/Wan-AI
- 仅第三方来源：https://howaiworks.ai/blog/alibaba-wan-open-weights-stopped-at-2-2 ， https://pasqualepillitteri.it/en/news/2354/kling-ai-3-0-complete-guide-kuaishou-video-generator ， https://www.atlascloud.ai/blog/guides/kling-3.0-turbo-kling-omni
- https://github.com/colmap/colmap/releases/tag/4.2.1 （CHANGELOG.rst）， https://github.com/colmap/glomap
- https://github.com/facebookresearch/vggt ， https://huggingface.co/facebook/VGGT-1B ， https://huggingface.co/facebook/VGGT-1B-Commercial ， https://huggingface.co/facebook/VGGT-Omega
- https://github.com/naver/dust3r ， https://github.com/naver/mast3r ， https://github.com/yyfz/Pi3 ， https://github.com/facebookresearch/map-anything
- https://github.com/nerfstudio-project/nerfstudio ， https://github.com/nerfstudio-project/gsplat ， https://github.com/graphdeco-inria/gaussian-splatting ， https://github.com/hbb1/2d-gaussian-splatting ， https://github.com/Anttwo/SuGaR

---

## 4. LLM + Blender（写代码建模）

### 4.1 工具：MCP for Blender（原名 BlenderMCP）

- 仓库已从 `ahujasid/blender-mcp` 更名为 **`ahujasid/mcp-for-blender`**（MIT，约 3 万星，2026-10-06 仍有推送）。PyPI 包名改为 `mcp-for-blender`，旧的 `uvx blender-mcp` 仍然能用。
- 接入 Claude Code：`claude mcp add blender uvx mcp-for-blender`；Windows 一键安装命令：`powershell -ExecutionPolicy ByPass -c "irm https://www.mcp-for-blender.com/install.ps1 | iex"`。
- 工作原理：Blender 插件 `addon.py` 在 Blender 内开一个 socket 服务（默认端口 `BLENDER_PORT=9876`），接收并执行 Python 命令。默认**可以执行任意 Python**，设置 `BLENDER_MCP_SAFE_MODE=1` 后会先检查脚本，拦截文件读写、联网、启动其他进程等危险操作。
- 资产来源：Poly Haven、Sketchfab、Poly Pizza，以及通过 Tripo / Hyper3D Rodin / Hunyuan3D 生成 AI 模型（这部分是付费高级功能）。

### 4.2 论文（arXiv 编号均已用 arXiv API 核对标题）

| 名称 | arXiv | 一句话 |
|---|---|---|
| 3D-GPT | 2310.12945 | 用 LLM 驱动程序化 3D 建模（Infinigen 类程序生成） |
| 3D-PreMise | 2401.06437 | 评测 LLM 能否生成带锐利特征、可参数控制的 3D 形状（代码形式） |
| SceneCraft（Hu 等） | 2403.01248 | "An LLM Agent for Synthesizing 3D Scene as Blender Code"。注意另有同名的 2410.09049（布局引导的场景生成），两者不是一回事 |
| BlenderAlchemy | 2404.17672 | 用 VLM 编辑 Blender 图形（材质 / 程序），渲染后由视觉反馈迭代 |
| BlenderGym | 2504.01786 | 基础模型做图形编辑任务的基准 |
| BlenderLLM | 2412.14203 | 在 Qwen2.5-Coder-7B-Instruct 上微调，输出 bpy 脚本；数据集 BlendNet 共 8k 条（2k 人工核验 + 6k GPT-4o 核验），基准 CADBench 700 条；每轮 SFT 约 21 分钟（4×A800 80GB） |
| LL3M | 2508.08228 | Large Language 3D Modelers：多智能体写 Blender 代码建模 |
| MeshCoder | 2508.14879 | 由点云生成结构化网格代码（Blender 脚本） |
| CAD-Recode | 2412.14042 | 点云 → CadQuery 代码；基座 Qwen2-1.5B，训练数据为 100 万条程序化生成的 CadQuery 代码，**1 张 H100 训练约 12 小时**（10 万步，batch 18）；在 DeepCAD、Fusion360、CC3D 上评测 |
| Text2CAD（Khan 等） | 2409.17106 | 从初学者到专家级的文本描述生成顺序 CAD 模型。另有同名的 2411.06206（经工程图生成 CAD） |
| CADCodeVerify | 2410.05340 | 论文标题为 "Generating CAD Code with Vision-Language Models for 3D Designs"，用 VLM 验证并修复 CAD 代码（**渲染 → 比对 → 修复**的典型闭环） |
| CAD-Coder（Doris 等） | 2505.14646 | 开源 VLM（LLaVA-1.5 13B）把图像转成 CadQuery 代码；数据集 GenCAD-Code 共 163,671 条脚本，每条配 5 张渲染图；4×H100 两阶段共约 10.2 小时 |
| CAD-Coder（另一篇） | 2505.19713 | 文本 → CAD，带 CoT 和几何奖励 |
| Img2CAD | 2408.01437 / 2410.03417 | 两篇同名论文，都是单图 → CAD |
| CADCrafter | 2504.04753 | 无约束图像 → CAD |
| Drawing2CAD | 2508.18733 | 矢量工程图 → CAD 序列，**和「三视图 → CAD」最接近** |

### 4.3 渲染—比对—修复闭环（建议实现）

1. LLM（或 VLM）读入三视图，输出 bpy 或 CadQuery 代码。
2. 用无界面的 Blender（`blender -b -P render_ortho.py`）把生成的模型按**同一套正交相机**渲染出 front / side / top 三张剪影和法线图。
3. 计算各视图的剪影 IoU、Chamfer（2D 轮廓）和尺寸比例误差。
4. 把差异图和数值反馈给 LLM，让它修改代码；达到阈值或最大轮数后停止。

这个闭环和 CADCodeVerify、BlenderAlchemy 的思路一致。俯视图在这里能直接用作校验信号。

### 4.4 「GPT6 ultra」核查

- 截至 2026-10-07，GPT-6 系列有 **GPT-6 Astra**（2026-09-03 有限预览、09-04 公开，API id `gpt-6-astra`，1M 上下文）和 **GPT-6 Sol、GPT-6 Luna**（2026-09-22）。**没有叫 "Ultra" 的版本**。来源为 Wikipedia 和 CNBC，**没有在 OpenAI 官网一手核实**。
- 本次调研所在的会话用的是 Claude Opus 5.5；ComfyUI v0.39.0 的 release notes 提到 Anthropic 合作节点新增了 "Sonnet 5.5"。哪个 LLM 或代码代理在 Blender/CAD 代码生成上最强，**没有可核实的统一基准结论**，建议用 BlenderGym / CADBench 自己测。

### 来源（§4）
- https://github.com/ahujasid/mcp-for-blender
- arXiv：2310.12945、2401.06437、2403.01248、2404.17672、2504.01786、2412.14203、2508.08228、2508.14879、2412.14042、2409.17106、2410.05340、2505.14646、2505.19713、2408.01437、2410.03417、2504.04753、2508.18733（均在 https://arxiv.org/abs/<id> ）
- GPT-6：https://en.wikipedia.org/wiki/GPT-6_Astra ， https://www.cnbc.com/2026/09/03/open-ai-astra-gpt-6-cyber.html

---

## 5. 「训练成模型」

### 5.1 数据集

| 数据集 | 规模 | 许可 | 来源 |
|---|---|---|---|
| Objaverse 1.0 | 80 万+ 个对象 | 整体 ODC-By v1.0；单个对象：CC-BY 4.0 721K、CC-BY-NC 4.0 25K、CC-BY-NC-SA 4.0 52K、CC-BY-SA 4.0 16K | HF `allenai/objaverse` |
| Objaverse-XL | 1000 万+ | 整体 ODC-By，单个对象许可各不相同 | HF `allenai/objaverse-xl` |
| ShapeNet | 已索引 300 万+，其中 22 万已分类到 3,135 个类别 | HF `ShapeNet/ShapeNetCore` 为 gated（需手动审批），**条款全文未核实** | arXiv 1512.03012 |
| ABC | 100 万个 CAD 模型（参数化曲线和曲面） | **未核实** | arXiv 1812.06216 |
| Fusion 360 Gallery | Reconstruction 子集 8,625 条（草图 + 拉伸）序列；Assembly 子集 8,251 个装配体 / 154,468 个零件；Segmentation 子集 35,680 个零件 | **仅限非商业研究**（LICENSE 第 1 条） | GitHub AutodeskAILab |
| DeepCAD | 178,238 个模型及其 CAD 构造序列（解析自 ABC / Onshape 公开文档） | **未核实** | arXiv 2105.09492 |
| Toys4K | 4,000 个对象、105 个类别（常见说法） | **未核实**（本轮搜索配额用完） | — |
| GSO（Google Scanned Objects） | 1000+ 件扫描的家居物品 | Creative Commons（论文摘要原话；具体为 CC-BY 4.0 **未核实**） | arXiv 2204.11918 |
| Thingi10K | 10,000 个 3D 打印模型，涉及 10 种开源许可 | 按模型分别授权 | arXiv 1605.04797 |
| Step1X-3D-obj-data | 80 万条高质量资产 uid | — | HF `stepfun-ai/Step1X-3D-obj-data` |
| Objaverse-Ortho10View | 正交 10 视角渲染 | — | HF `huanngzh/Objaverse-Ortho10View`（MV-Adapter 发布） |
| GenCAD-Code | 163,671 条 CadQuery 脚本 × 5 张渲染图 | — | CAD-Coder 论文 |

### 5.2 训练一个「图像 → 3D」大模型要多少算力（论文原文）

| 模型 | 报告的训练算力 | 数据 |
|---|---|---|
| LRM（arXiv 2311.04400） | 128 张 A100 40G，batch 1024，30 个 epoch，**约 3 天** | Objaverse 730,648 个 + MVImgNet 220,219 段视频 |
| Instant3D（arXiv 2311.06214） | 多视图扩散：32 张 A100 训练 **3 小时**（1 万条精选数据）；重建器：128 张 A100，**7 天** | Objaverse 约 75 万 |
| TRELLIS（arXiv 2412.01506） | XL（2B）：64 张 A100 40G，40 万步，batch 256 | 约 50 万资产（Objaverse-XL、ABO、3D-FUTURE、HSSD），每个资产渲染 150 张图 |
| TRELLIS.2（arXiv 2512.14692） | SC-VAE：16 张 H100；生成模型：32 张 H100，batch 256 | 约 80 万资产（加入了 TexVerse） |
| Direct3D-S2 | README 称 1024³ 分辨率**只需 8 张 GPU** | 公开数据 |
| Hunyuan3D 2.0 / 2.1 | **未在本轮核实到 GPU 小时数** | — |

结论：**从零训练一个通用的图像 → 3D 基座模型，需要几十到上百张 A100/H100 跑数天，2×3090 做不到。**

### 5.3 2×3090（48GB，NVLink）能做的事

1. **微调生成三视图的 LoRA**（可行性最高）：在 Qwen-Image-Edit-2511（Apache-2.0，商用友好）上，用 Objaverse 的 CC-BY 子集按正交相机渲染出（正面图 → 标准三视图）的配对数据，训练一个「正交三视图」LoRA。社区已有 multi-angle 和 turnaround LoRA，说明这条路走得通（§1.3）。20B 基座在 24GB 卡上做 LoRA 训练需要量化或卸载；用什么训练框架、显存是否够：**未核实**。
2. **训练小型「三视图 → 代码」模型**（与本项目最契合）：照 CAD-Recode 的做法（1.5B LLM、100 万条程序化 CadQuery 数据、1 张 H100 约 12 小时）和 CAD-Coder 的做法（13B VLM、16 万条数据、4×H100 约 10 小时）。数据自己造：程序化生成 CadQuery 或 bpy 代码 → 渲染正交三视图 → 得到（三视图图像, 代码）配对 → 对小型 VLM 做 LoRA 或 SFT。按本机全局约定，**打分或推理类任务至少用 7B 模型，0.6B/1.7B 只能用来跑通流程**。训练用 `~/lab/.venv`，评测推理用 `infer-py`。
3. **微调现成的 3D 生成模型**：Hunyuan3D-2.1、TRELLIS.2、Step1X-3D 都开源了训练代码，可以拿少量领域数据（例如某类硬表面零件）做微调。在 2×3090 上的可行性和显存需求：**未核实**。原作的预训练规模见 §5.2。
4. **不训练也能拿到成果**：把「Qwen 生成三视图 → Pixal3D-MV / Hunyuan3D-2mv → Blender 清理」这条管线做成评测基准，比较不同模型的剪影 IoU 和比例误差，本身就能写成论文。

### 来源（§5）
- https://huggingface.co/datasets/allenai/objaverse ， https://huggingface.co/datasets/allenai/objaverse-xl
- https://github.com/AutodeskAILab/Fusion360GalleryDataset （README、LICENSE.md）， https://github.com/ChrisWu1997/DeepCAD
- arXiv：1512.03012、1812.06216、2105.09492、2204.11918、1605.04797、2010.02392、2311.04400、2311.06214、2412.01506、2512.14692、2412.14042、2505.14646
- https://github.com/DreamTechAI/Direct3D-S2 ， https://github.com/huanngzh/MV-Adapter ， https://github.com/stepfun-ai/Step1X-3D

---

## 6. 路线对比表

机器说明：开发用笔记本 为 RTX 5090 Laptop，24GB 共享显存，实际可用约 12GB，Windows，Blackwell sm_120；服务器为 2×RTX 3090（48GB，NVLink），Ubuntu 24.04，Ampere sm_86。

| 路线 | 精度 | 可控性 | 硬表面 | 有机体 / 角色 | 成本 | 本机（5090 Laptop，约 12GB，Windows） | 服务器（2×3090，Linux） |
|---|---|---|---|---|---|---|---|
| **A. 原生图像 / 多视图 → 3D**（Hunyuan3D-2mv / Pixal3D-MV / TRELLIS.2） | 中高：外形像，但尺寸不精确，细节靠模型脑补 | 中：只能控制输入视图，没有参数 | 中（倒角、平面容易被"软化"） | **高**（主场） | 低（开源，本地跑） | **能**：ComfyUI 原生节点，int8 模型 5–5.6 GB；实际速度和显存余量**未实测** | **能**：官方代码可在 Linux 上编译，单卡 24GB 满足 TRELLIS.2 的 ≥24GB 要求（余量很紧） |
| **B. 视频 → 重建**（H3 / Wan / Kling 生成环绕视频 → VGGT / COLMAP → 3DGS / 2DGS → 网格） | 低到中：受视频生成的几何一致性限制 | 低 | 低到中 | 中 | API 费用（H3 768P 15 s 约 $1.2）+ 重建算力 | 部分可行：COLMAP CUDA 版和 VGGT 能跑；视频生成走 API；H3 本地跑不动 | **能**：gsplat / 2DGS 编译和训练没问题；Wan2.2-5B 本地生成可行（官方称 4090 级显卡可跑） |
| **C. LLM + Blender / CAD 代码**（MCP for Blender，渲染—比对—修复） | **尺寸精确**（参数化） | **最高**（代码可改） | **最高** | 低（角色、布料难写成代码） | LLM API 费用 | **能**（Blender 5.2 LTS 便携版 + MCP，几乎不占 GPU） | 能（无界面 Blender 渲染；本地 VLM 微调或评测用 `infer-py`） |
| **D. 视觉外壳雕刻**（三视图剪影求交，体素） | 低：凹陷处无法恢复 | 中 | 中（适合拉伸、方块类） | 低 | 几乎为零 | **能**（numpy 即可） | 能 |
| **E. 自己训练模型**（三视图 LoRA / 三视图 → 代码的小模型 / 微调 3D 生成器） | 取决于数据 | 高（可以针对领域） | 高（如果训练代码模型） | 中（如果训练三视图 LoRA） | 高（数据制作 + 数十 GPU 小时） | 不建议（显存不够，合盖就中断） | **能**：LoRA / SFT 的 1.5B–8B VLM 或代码模型；从零训练 3D 基座**不行**（§5.2） |

**建议组合**：用 C 做硬表面，用 A 做角色和有机体，用 D 做基线和校验。三视图统一用 Qwen-Image-Edit-2511 生成；如果只做研究，也可以用 Qwen-Image-2.1。E 作为第二阶段的论文贡献。

---

## 7. 安装相关事实（已核实）

| 软件 | 版本 | 下载 | 大小 |
|---|---|---|---|
| Blender（**5.2 是当前 LTS**，blender.org/download/lts 页面列出 "Blender 5.2 LTS"） | 5.2.2（2026-09-15） | https://download.blender.org/release/Blender5.2/blender-5.2.2-windows-x64.zip | 404,453,484 B（约 404 MB） |
| Blender 上一个 LTS | 4.5.9（2026-04-21） | https://download.blender.org/release/Blender4.5/blender-4.5.9-windows-x64.zip | 399,051,129 B |
| ComfyUI 便携版（NVIDIA） | v0.39.0（2026-10-05）；README 写明自带 Python 3.13 + PyTorch cu130，支持 20 系及以上（含 RTX 50）；**具体 torch 小版本号未核实** | https://github.com/Comfy-Org/ComfyUI/releases/download/v0.39.0/ComfyUI_windows_portable_nvidia.7z | 2,003,629,308 B（约 2.0 GB） |
| ComfyUI 便携版 cu126 | 同上（Python 3.12 + cu126，**仅供 10 系及更老显卡**，20 系以上不要用） | `.../ComfyUI_windows_portable_nvidia_cu126.7z` | 1,895,335,518 B |
| COLMAP | 4.2.1（2026-09-29） | https://github.com/colmap/colmap/releases/download/4.2.1/colmap-x64-windows-cuda.zip | 414,666,626 B |

### 来源（§6–7）
- https://download.blender.org/release/Blender5.2/ ， https://www.blender.org/download/lts/
- https://github.com/Comfy-Org/ComfyUI （README "Windows Portable" 一节）， https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.39.0
- https://github.com/colmap/colmap/releases/tag/4.2.1

---

## 8. 本项目实测补充（2026-10-08，开发用笔记本）

以下是调研之后在本机实际安装、运行得到的事实，用来更正或补全上文标“未实测/未核实”的条目。数字见 REPORT.md。

- **ComfyUI v0.39.0 便携版**：实测内置 Python 3.13.14 + **torch 2.14.0+cu130**，RTX 5090 Laptop（sm_120）CUDA 可用。
  Windows「智能应用控制」拦截了其中 `kornia_rs` 的 DLL，导致 `nodes_latent.py`、`nodes_post_processing.py` 两个可选节点模块加载失败；
  本项目用到的 Qwen-Image-2.1、Hunyuan3D、TRELLIS.2 / Pixal3D 节点都不受影响。
- **Qwen-Image-2.1 int8 + Qwen3-VL-8B w4a8 文本编码器**（合计约 14.2 GB 文件）在约 12 GB 显存预算下可跑（ComfyUI 动态显存加载）：
  1536×640、25 步约 23–29 s/张；加 Fun-Acc-4Step LoRA、4 步约 5 s/张，但画面明显发灰，有 2/10 张淡到抠不出前景。
- **Pixal3D-MV（int8）**：节点说明要求“所有视图同一尺度、物体最长边约占画面 1/1.1、黑底”。官方模板对每个视图分别抠图裁剪，
  会破坏视图间的尺度关系；本项目改为自己做同尺度预处理后几何明显变好（REPORT.md 的消融表）。上采样 1024 时每个物体 40–130 s 生成，
  输出网格 160 万–1200 万面，必须简化（trimesh 的 fast_simplification 被智能应用控制拦截，改在 Blender 里简化）。
- **Hunyuan3D-2mv（fp16，ComfyUI 原生）**：显存约 5–6 GB，每个物体约 50 s。
- **VGGT-1B**：24 帧 518² 输入，峰值显存 9.9 GB，约 15–25 s；纯 PyTorch，无需编译。
- **未在本机安装**：COLMAP（VGGT 已覆盖位姿估计；COLMAP Windows 包未下载，是否被智能应用控制拦截未知）、gsplat/2DGS（需要 MSVC + nvcc 编译，
  本机没有且不允许装系统级软件）、MiniMax H3 本地权重（≥16 GB，放不下）。
