# 现场演示脚本（约 12 分钟）

> 设备：开发用笔记本。演示前 10 分钟做完“准备”一节。所有命令在项目根目录的 PowerShell 里执行，先 `$env:PYTHONPATH = "src"`。

## 准备（演示前）

```powershell
nvidia-smi                                            # 看剩余显存，≥ 10 GB 再开 ComfyUI
.venv\Scripts\python -m p6.comfy.client 12            # 另开一个窗口常驻 ComfyUI（端口 8196），等出现 "ComfyUI ready"
start showcase\index.html                             # 浏览器打开展示页（离线可用）
```

备用：网络不好或显存被占满时，直接用展示页和 `runs/` 里已有的结果讲，不现场生成。

## 1. 开场：问题是什么（1 分钟）
- 展示页顶部：最初的三个设想（ComfyUI + Qwen 三视图；MiniMax 旋转视频再扫描；最强 LLM + Blender）。
- 一句话：我们做了一把有真值的尺子（16 个物体），把每条路线量了一遍。

## 2. 基准（1 分钟）
- 打开 `docs/img/bench_overview.png` 或展示页“图 2”上方的输入三视图。
- 强调“盲测零件”：随机生成、参数没给写脚本的人看过。

## 3. 现场跑一条最快的路线：三视图轮廓雕刻（1 分钟）

```powershell
.venv\Scripts\python -m p6.routes.visual_hull --cases bracket,mug
```
- 讲：轮廓 IoU 0.98 以上，但杯子内腔被填实、支架三角筋被拉成整块——“三视图对得上 ≠ 三维对”。

## 4. LLM + Blender 闭环（3 分钟）
- 展示页“图 5”：blind_05 两轮的差异图（红=缺、蓝=多）。
- 现场跑规则 mock 版本，让观众看到闭环每一轮在 Blender 里自动建模、渲染、比对：

```powershell
.venv\Scripts\python -m p6.routes.llm_blender --provider mock --cases blind_05 --iters 3
```
- 讲清楚：表里的 “Claude 代理” 是 Claude Code 代理手动扮演 LLM（文件交换），不是 API 自动调用；有 key 后命令是
  `--provider anthropic`（README）。
- 打开 `runs/C_llm_blender/<最新>/blind_05/iter_03/script.py` 给大家看生成的 bpy 代码（mock 的是挤出求交，Claude 的是可读的“底板 + 筋 + 孔”，见 `runs/C_llm_blender/20261007-233505_file_claude-in-loop-b2/blind_05/iter_02/script.py`）。

## 5. ComfyUI：Qwen-Image-2.1 画三视图（2 分钟，需要 ComfyUI 在线）

```powershell
.venv\Scripts\python -m p6.gen_views edit --cases spot --seeds 3 --lora 0 --steps 25
```
- 约 30 秒出图。打开生成的三联图和 `views/` 下切分规范化后的三张，再看终端打印的侧/顶视图 IoU。
- 讲：正视图像，侧视图、顶视图经常不对（展示页“图 4”有 4 个例子：方杯、薄板支架）。

## 6. 原生 3D 模型（2 分钟，需要 ComfyUI 在线）

```powershell
.venv\Scripts\python -m p6.routes.native3d --method hunyuan_mv --cases spot
```
- 约 1 分钟。同时在展示页“图 2”切到 spot，旋转对比真值、Pixal3D、Hunyuan3D、视频重建、Claude 代理拼的基本体。
- 讲：有机体这条最好；切到 chair / bracket 看它在薄件、硬表面上的失败。

## 7. 视频路线（1 分钟）
- 展示 `docs/img/video_gear_rings.png`：一圈 vs 两圈。MiniMax 客户端没有 key，只演示 stub：

```powershell
.venv\Scripts\python -m p6.video.minimax --image bench\data\spot\views\front.png --out runs\video\demo
type runs\video\demo\minimax_request.json
```

## 8. 收尾：推荐与待决（1 分钟）
- 展示页“路线对比”表和“落地前要考虑的事”。

## 常见问题
- ComfyUI 启动失败：看 `runs/comfy/server.log`；`kornia_rs` 的导入告警是智能应用控制拦截，可忽略。
- 显存不够：先关 ComfyUI 再跑 VGGT（峰值 9.9 GB），或改用展示页里的已有结果。
- 展示页 3D 不显示：浏览器禁用了 WebGL；改看 `docs/img/qual_*.png`。
