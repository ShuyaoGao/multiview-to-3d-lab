@echo off
chcp 65001 >nul
rem 一键打开离线演示页（只回放已有结果，不启动 ComfyUI、不加载模型）。说明见 DEMO_GUIDE.md
setlocal
set "ROOT=%~dp0"
if not exist "%ROOT%demo\index.html" (
  echo [错误] 找不到 demo\index.html
  pause
  exit /b 1
)
if not exist "%ROOT%showcase\assets\meshes.js" (
  echo [提示] 缺少 showcase\assets\meshes.js，3D 视窗会是空的。
  echo        重建：$env:PYTHONPATH="src"; .venv\Scripts\python showcase\build_assets.py
)
if not exist "%ROOT%demo\assets\demo_data.js" (
  echo [提示] 缺少 demo\assets\demo_data.js。重建：.venv\Scripts\python demo\build_demo_assets.py
)
echo 正在用默认浏览器打开演示页……按 F11 全屏，空格暂停，左右方向键切换。
start "" "%ROOT%demo\index.html"
endlocal
