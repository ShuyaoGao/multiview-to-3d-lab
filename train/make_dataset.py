"""“训练自己的模型”的数据管线原型：程序化随机零件 → (三视图, p6h 建模脚本) 配对 → SFT 用 jsonl。

思路同 CAD-Recode / BlenderLLM：代码本身就是标注，渲染出来的三视图就是输入，数据可以无限合成且零版权问题。
本脚本只生成小样本验证管线；规模化（10^5–10^6 条）应放到服务器上并行跑（见 docs/REPORT.md §5）。

  .venv/Scripts/python train/make_dataset.py --n 40 --out runs/dataset/demo
输出：
  <out>/<id>/{script.py, front.png, side.png, top.png, gt.obj}
  <out>/sft.jsonl   每行 {"id", "images": [front, side, top], "messages": [system, user, assistant(代码)]}
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))

from build import blind_script  # noqa: E402

from p6 import blender  # noqa: E402
from p6.align import normalize  # noqa: E402
from p6.llm import prompts  # noqa: E402
from p6.metrics import load_mesh  # noqa: E402
from p6.stamp import stamp_run  # noqa: E402


def lathe_script(seed: int) -> str:
    """随机回转体（轴、法兰、瓶子类），补充 blind_script 的箱体类分布。"""
    r = random.Random(seed)
    zs = sorted(r.uniform(-0.8, 0.8) for _ in range(r.randint(3, 6)))
    prof = [(0, -0.9)]
    for z in [-0.9] + zs + [0.9]:
        prof.append((round(r.uniform(0.1, 0.6), 3), round(z, 3)))
    prof.append((0, 0.9))
    # 让轮廓成阶梯：每个 z 处重复半径
    steps = [prof[0]]
    for (r0, z0), (r1, z1) in zip(prof[1:-2], prof[2:-1]):
        steps += [(r0, z0), (r0, z1)]
    steps += [prof[-2], prof[-1]]
    L = [f"part = revolve({steps}, axis='Z', segments=64)"]
    if r.random() < 0.6:
        n = r.choice([3, 4, 6, 8])
        rr = r.uniform(0.25, 0.45)
        L.append(f"holes = [cylinder(0.05, 3.0, ({rr:.3f} * math.cos(2 * math.pi * k / {n}), "
                 f"{rr:.3f} * math.sin(2 * math.pi * k / {n}), 0)) for k in range({n})]")
        L.append("difference(part, *holes)")
    return "# 随机回转体\n" + "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default=str(ROOT / "runs" / "dataset" / "demo"))
    ap.add_argument("--seed0", type=int, default=50000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp_run(out, {"what": "synthetic three-view→code dataset", "n": a.n})
    system = prompts.system_prompt()
    ok, t0 = 0, time.time()
    with open(out / "sft.jsonl", "w", encoding="utf-8") as fj:
        for i in range(a.n):
            sid = a.seed0 + i
            code = blind_script(sid) if i % 2 == 0 else lathe_script(sid)
            d = out / f"{sid}"
            d.mkdir(exist_ok=True)
            (d / "script.py").write_text(code, encoding="utf-8")
            meta = blender.build_mesh(d / "script.py", d / "raw.obj")
            if not meta.get("ok"):
                continue
            # 训练样本的约定：脚本本身就在归一化坐标里更好学；这里记录归一化变换供需要时使用
            m = load_mesh(d / "raw.obj")
            nm = normalize(m)
            nm.export(d / "gt.obj")
            imgs = blender.render_ortho3(d / "gt.obj", d / "views")
            lo, hi = m.bounds
            rec = {"id": str(sid), "images": [str(imgs[v].relative_to(out)) for v in ("front", "side", "top")],
                   "normalize": {"center": ((lo + hi) / 2).tolist(), "scale": float(1.6 / (hi - lo).max())},
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": "<image><image><image>\nFRONT, SIDE, TOP views. Write the Blender script."},
                                {"role": "assistant", "content": f"```python\n{code}```"}]}
            fj.write(json.dumps(rec, ensure_ascii=False) + "\n")
            (d / "raw.obj").unlink()
            ok += 1
    print(f"{ok}/{a.n} 条样本，{time.time() - t0:.0f}s → {out}")


if __name__ == "__main__":
    main()
