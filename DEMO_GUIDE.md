# 一键演示指南

> 适用：课堂或汇报现场，5–10 分钟讲清“AI 三视图 → 3D 建模”的五条路线。
> 演示页只回放 `runs/` 里已有的结果，**不启动 ComfyUI、不加载任何模型、不联网**。
> 在线版：https://shuyaogao.github.io/multiview-to-3d-lab/ （和本地双击打开的是同一份页面）。

## 1. 双击哪个文件

在线打开 https://shuyaogao.github.io/multiview-to-3d-lab/ 。离线时双击仓库根目录的 `index.html` 或 `demo/index.html`；Windows 上也可以双击 `启动演示.bat`。

- 打开后**自动开始播放**，共 6 章 17 步，1× 速度约 5 分 15 秒。
- 按 **F11** 全屏。投影仪分辨率 1600×900 及以上效果最好。
- 页面跟随系统深色/浅色主题。

| 操作 | 按键 / 按钮 |
|---|---|
| 暂停 / 继续 | 空格，或顶栏 ❚❚ |
| 上一步 / 下一步 | ← / →（翻页笔的上下翻页键也可以） |
| 跳到第 N 章 | 数字键 1–6，或点顶栏章节名 |
| 自动讲解开关 | 顶栏“自动讲解”。关掉后每步停住，按 → 才走（适合边讲边停） |
| 用系统语音朗读字幕 | 顶栏“朗读”（需要 Windows 装有中文语音；默认关闭） |
| 速度 | 顶栏 0.75× / 1× / 1.5× / 2× |
| 转 3D 视角 | 在任一 3D 视窗里拖动，松手 2.5 秒后恢复自动旋转 |
| 从某一步打开 | 地址后加 `?step=8`；加 `&autoplay=0` 则不自动翻页 |

## 2. 会出现什么画面（17 步）

| 章 | 步 | 画面 | 时长 |
|---|---|---|---|
| 1 开场 | 1 | 五张路线卡片依次弹出，每张标明它是传统算法、深度学习还是大语言模型 | 16 s |
| | 2 | 基准总览图 + 四个指标（Chamfer、F-score、体积 IoU、轮廓 IoU）的直白解释 | 15 s |
| 2 三视图不唯一 | 3 | 杯子的三视图，抛出问题：看得出杯子是空心的吗？ | 9 s |
| | 4 | **浏览器里现场做轮廓雕刻**：正视图拉伸（浅蓝）→ 侧视图切一刀（蓝）→ 顶视图切一刀，体素自下而上逐层出现（橙）。左边对应的那张视图高亮 | 20 s |
| | 5 | 现场体素结果、真值、离线 256³ 结果三窗并排，杯子内腔被填实 | 14 s |
| 3 各路线对比 | 6–9 | 支架、奶牛 Spot、盲测零件 05、椅子：左边输入三视图，右边 6–8 个 3D 窗依次“长出来”（剖切面自下而上扫过），长完显示 CD / F / 体积 IoU | 各 20–22 s |
| 4 LLM 闭环回放 | 10–11 | 盲测零件 05 的第 1、2 轮：脚本逐字打出（第 2 轮改动的行有底色），Blender 渲染图和差异图出现，轮廓 IoU 条和停止阈值线，右侧网格生长、真值线框作参照 | 16 + 15 s |
| | 12 | 规则 mock 的 3 轮（包围盒 → 正视图挤出 → 三视图求交），自动切换 | 18 s |
| | 13 | Claude 代理用椭球和胶囊拼奶牛的 3 轮，自动切换 | 18 s |
| 5 Qwen 三视图 | 14–15 | 支架、杯子、奶牛、房子：真值三视图与 Qwen-Image-2.1 补画的三视图上下对照，每张标 IoU（红 < 0.5，橙 < 0.8） | 16 + 15 s |
| | 16 | 只给一句话生成的四张三视图和尺寸一致性误差 | 13 s |
| 6 结论 | 17 | 盲测 6 件的 F-score 与体积 IoU 条形图、路线对比表、推荐、局限 | 24 s |

屏幕底部是字幕，就是这一步要讲的话。字幕里的数字在页面加载时从 `showcase/assets/data.js` 和 `demo/assets/demo_data.js` 读出，这两个文件又来自各次运行的 `result.json`，没有手填的数字。

截图：`demo/screenshots/01_carving.png`（第 4 步）、`03_llm_loop.png`（第 11 步）。

## 3. 5–10 分钟讲法

**5 分钟版**：开着“自动讲解”从头放到尾，照字幕讲，不停。

**10 分钟版**：关掉“自动讲解”，每步讲完按 →。建议在这几处多停一会儿：

1. **第 3 步（约 1 分钟）**：让观众先猜杯子是不是空心的，再按 → 看雕刻。这是全场的钩子。
2. **第 4–5 步（约 1.5 分钟）**：讲清轮廓雕刻是纯几何算法，不需要训练；它是三张轮廓能“解释”的最大形体。杯子体积 IoU 只有 0.329，因为从三个方向都看不到内腔。结论：**三视图不唯一**。
3. **第 8 步（约 1 分钟）**：盲测零件上，Claude 代理闭环 CD 0.0122，比所有生成模型都准。可以拖动视窗让大家看 Pixal3D 的碎面（Hunyuan3D-2mv 的网格因许可证地域限制未公开，视窗里只显示指标）。
4. **第 10–12 步（约 2 分钟）**：讲闭环为什么有效。最关键的一句：规则 mock 的轮廓 IoU 和 Claude 代理差不多，体积 IoU 却低很多（0.807 vs 0.963），差距在“读懂零件结构”。
   **必须说明**：表里的“Claude 代理”是 Claude Code 代理通过文件交换手动扮演 LLM，不是 API 自动调用；盲测生成器也是同一代理写的，数字是上限。
5. **第 14–15 步（约 1 分钟）**：Qwen 补画的侧视图、顶视图经常结构错误（支架侧视 IoU 0.186），所以 AI 链路的瓶颈在第一步。
6. **第 17 步（约 1 分钟）**：按物体类型给推荐：硬表面用 LLM 写代码，有机体用 Pixal3D / Hunyuan3D，有实物拍视频走 VGGT。

## 4. 出问题怎么办

| 现象 | 处理 |
|---|---|
| 3D 窗全是空的，左下角红条提示 `meshes.js 不存在` | `showcase/assets/meshes.js` 缺失或损坏。重建（需要本地 `runs/` 里的结果）：`$env:PYTHONPATH="src"; .venv\Scripts\python showcase\build_assets.py`（要用 Blender 简化网格，几分钟） |
| 闭环页没有图、红条提示素材没加载 | 重建：`$env:PYTHONPATH="src"; .venv\Scripts\python demo\build_demo_assets.py`（约 1 分钟，只读 runs/） |
| 3D 窗是空的但没有红条，或提示 WebGL 不可用 | 浏览器禁用了硬件加速：Edge 设置 → 系统 → 打开“使用图形加速”。图片和字幕不受影响 |
| 画面卡顿 | 机器上有别的程序占 GPU。调到 0.75× 或关掉自动讲解手动翻页 |
| 字体挤、表格被截 | 窗口太小。按 F11 全屏，或把浏览器缩放调到 90% |
| “朗读”没声音 | 系统没装中文语音包（设置 → 时间和语言 → 语音）。不影响演示 |
| 想回到开头 | 按数字键 1 |

## 5. 想现场跑某条路线

在项目根目录打开 PowerShell，先执行 `$env:PYTHONPATH = "src"`。下面两条**不用 GPU、不用联网、不加载模型**，适合现场跑：

```powershell
# 路线 D 轮廓雕刻：每件计算约 6 秒；整条命令含 Blender 渲染评测，今天实测两件共约 1 分 44 秒（机器上同时有别的任务）
.venv\Scripts\python -m p6.routes.visual_hull --cases bracket,mug

# 路线 C 闭环（规则 mock，不是 LLM）：每轮在无头 Blender 里建模、渲染、比对，3 轮约 40 秒
.venv\Scripts\python -m p6.routes.llm_blender --provider mock --cases blind_05 --iters 3
```

终端会逐件打印 CD、F-score、体积 IoU，结果写到 `runs/D_visual_hull/<时间戳>/`、`runs/C_llm_blender/<时间戳>/`。
轮廓雕刻的终端里可能夹着一段 `UnicodeDecodeError: 'gbk'`（读取子进程输出时的编码问题），不影响结果。

下面几条要 GPU 或 API key，**现场不建议跑**（显存、时间都不可控），需要时见 `showcase/DEMO.md` 和 `README.md`：

| 路线 | 命令 | 需要 |
|---|---|---|
| C 真 LLM 闭环 | `.venv\Scripts\python -m p6.routes.llm_blender --provider anthropic --cases blind_05` | `ANTHROPIC_API_KEY`（本项目没有） |
| A 原生 3D | `.venv\Scripts\python -m p6.routes.native3d --method hunyuan_mv --cases spot` | 先常驻 ComfyUI（约 12 GB 显存），约 1 分钟/件 |
| G Qwen 画三视图 | `.venv\Scripts\python -m p6.gen_views edit --cases spot --seeds 3 --lora 0 --steps 25` | 同上，约 30 秒/张 |
| B 视频重建 | `.venv\Scripts\python -m p6.routes.video_recon --cases gear --frames 24 --elevs 20,-20 --render-frames 48` | VGGT 峰值约 9.9 GB 显存 |

## 6. 素材从哪来

| 文件 | 来源 | 是否进 git |
|---|---|---|
| `demo/index.html` | 手写 | 是 |
| `demo/assets/demo_data.js`、`loop_meshes.js`、`loop/*.png` | `demo/build_demo_assets.py`：80×80 轮廓掩码（来自 `bench/data/*/mask_*.png`），4 段闭环记录（来自 `runs/C_llm_blender/`） | 是（约 1.4 MB） |
| `showcase/assets/data.js`、`vendor/` | `showcase/build_assets.py` | 是 |
| `showcase/assets/meshes.js`、`cases/`、`gen/` | `showcase/build_assets.py`（公开版不含 Hunyuan3D-2mv 的网格，见 THIRD_PARTY.md） | 是 |

