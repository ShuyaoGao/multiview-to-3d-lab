# 第三方组件、数据与模型输出

本仓库**不包含任何模型权重**。下面列出仓库里随附的第三方文件，以及演示里展示的第三方模型输出各自的许可证。
本仓库自己的代码用 MIT（`LICENSE`），文档和教学素材用 CC BY 4.0（`LICENSE-docs`）；下表中的条目**不受**这两份许可证覆盖，以各自的许可证为准。

## 1. 随附的第三方代码

| 文件 | 来源 | 许可证 |
|---|---|---|
| `showcase/assets/vendor/three.min.js` | three.js r147 | MIT（版权 three.js authors，原文见 `showcase/assets/vendor/LICENSE`） |
| `showcase/assets/vendor/GLTFLoader.js`、`OrbitControls.js` | three.js r147 的 `examples/js` | MIT（同上） |

页面只用系统字体，没有随附字体文件。

## 2. 评测基准里的形体（全部是合成数据）

| 形体 | 来源 | 许可证 |
|---|---|---|
| bracket、flange_shaft、gear、mug、table、chair、house、figure | 本项目程序化生成（`bench/shapes/*.py`） | 随本仓库（代码 MIT） |
| 6 个盲测零件 `blind_01`–`blind_06` | `bench/build.py --blind 6` 用固定种子随机生成 | 随本仓库 |
| spot（奶牛）、bob | Keenan Crane, *3D Model Repository*（https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/） | CC0 1.0 |

所有输入三视图都是本项目用 Blender 渲染的灰模图，**不是照片，也不含任何受版权保护的角色形象**。

## 3. 演示里展示的第三方模型输出

| 路线 | 模型 | 模型许可证 | 本仓库里的输出 |
|---|---|---|---|
| A | Pixal3D-MV（TencentARC，经 ComfyUI 原生节点） | MIT；图像编码器 DINOv3 另受 Meta DINOv3 License 约束 | `showcase/assets/meshes.js` 里 `*/A_pixal` 网格（已简化） |
| A | TRELLIS.2（Microsoft，经 ComfyUI 原生节点） | MIT；依赖 DINOv3（Meta DINOv3 License） | `*/A_trellis` 网格（已简化） |
| A | **Hunyuan3D-2mv**（Tencent） | Tencent Hunyuan 3D 2.0 Community License。该许可**不适用于欧盟、英国和韩国**，并限制在许可地域以外使用模型的输出 | **不随附**：网格和渲染图都没有放进本仓库，演示里对应的视窗只显示评测指标 |
| B | VGGT-1B（Meta） | 权重 CC BY-NC 4.0（非商用） | `*/B` 网格：VGGT 估计的深度经 TSDF 融合得到。**仅限非商业用途** |
| G | Qwen-Image-2.1（阿里 Qwen） | Qwen Research License，**仅限非商业用途** | `showcase/assets/gen/*/gen_*.png`、`showcase/assets/img/t2i_*.jpg`、`docs/img/qwen_*_sheets.jpg`。**仅限非商业用途** |
| G | BiRefNet（经 ComfyUI 重打包） | MIT | 只用于给生成图去背景，没有单独的输出文件 |
| C | Claude（Anthropic），由 Claude Code 代理通过文件交换扮演闭环里的 LLM | — | `demo/assets/demo_data.js` 里回放的建模脚本和思路文字 |

工具：Blender（GPL-3.0）和 ComfyUI（GPL-3.0）只作为外部程序调用，本仓库不包含它们的代码。
`comfy_workflows/*.api.json` 由 ComfyUI 官方模板改写成 API 格式，模型文件名指向各自上游仓库，权重需要自己按 `src/p6/fetch.py` 的清单下载并遵守各自的许可证。

商用前请逐项核对上游许可证；调研记录见 `docs/LANDSCAPE.md`（2026-10 核查）。
