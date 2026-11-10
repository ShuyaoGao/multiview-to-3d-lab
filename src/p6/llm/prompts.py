"""路线 C 的提示词模板（给模型看的部分用英文，模型普遍对英文代码提示更稳）。"""
from __future__ import annotations

from pathlib import Path

from ..paths import BLENDER_SCRIPTS


def p6h_api_doc() -> str:
    src = (BLENDER_SCRIPTS / "p6h.py").read_text(encoding="utf-8")
    doc = src.split('"""')[1]
    return doc.strip()


SYSTEM = """You are an expert CAD / Blender modeller. You receive an orthographic three-view drawing
(front, side, top) of ONE object and must write a Blender Python script that builds that object as a solid.

COORDINATE CONVENTIONS (exact, all three views share one scale):
- World is Z-up. Every view image spans world coordinates [-1, 1] x [-1, 1]; images are 512x512 px,
  so 1 px = 0.0039 world units. The image centre is world 0.
- FRONT view: camera at -Y looking toward +Y. Image right = +X, image up = +Z.
- SIDE  view: camera at +X looking toward -X. Image right = +Y, image up = +Z.
- TOP   view: camera at +Z looking down.     Image right = +X, image up = +Y.
- Views are grey shaded with black outlines on white background; only geometry matters (no colour/material).

AVAILABLE HELPERS (already imported, plus `bpy` and `math`) -- module p6h:
{api}

RULES:
- Reply with a short plan (<= 8 lines) and then exactly ONE ```python code block containing the COMPLETE script.
- Do not import anything except math; do not read/write files; do not call clear() (scene is already empty).
- Everything left in the scene is unioned into one solid and exported automatically.
- Prefer exact primitives/booleans that match the drawing (holes, slots, fillets approximated by cylinders, etc.).
- Use the numeric measurements given to you; they come from the target images.
- When you receive feedback, rewrite the whole script (not a diff) to fix the reported discrepancies.
"""


def system_prompt() -> str:
    return SYSTEM.format(api=p6h_api_doc())


def first_message(measure: dict, extra_note: str = "") -> str:
    lines = ["Here is the target three-view drawing: images in the order FRONT, SIDE, TOP.",
             "Measured silhouette bounding boxes (world units, from the target images):"]
    for v, b in measure.items():
        lines.append(f"- {v}: {b}")
    if extra_note:
        lines.append(extra_note)
    lines.append("Write the Blender script that reproduces this object.")
    return "\n".join(lines)


def feedback_message(it: int, ious: dict, measure_t: dict, measure_p: dict, px: dict, err: str | None = None) -> str:
    if err:
        return (f"Iteration {it}: the script FAILED in Blender:\n```\n{err[-2500:]}\n```\n"
                "Fix the error and send the complete corrected script.")
    L = [f"Iteration {it} result. Silhouette IoU vs target: "
         + ", ".join(f"{v}={ious[v]:.3f}" for v in ("front", "side", "top")) + f" (mean {ious['mean']:.3f}).",
         "Attached images: for each view a DIFF image (dark grey = correct, RED = target has material that your model is MISSING, "
         "BLUE = your model has EXTRA material not in the target), followed by the shaded render of your current model (FRONT, SIDE, TOP).",
         "Bounding boxes target vs yours:"]
    for v in ("front", "side", "top"):
        L.append(f"- {v}: target {measure_t[v]} | yours {measure_p[v]} | missing px {px[v]['missing']}, extra px {px[v]['extra']}")
    L.append("Correct the discrepancies and send the complete revised script.")
    return "\n".join(L)


def extract_code(text: str) -> str | None:
    import re

    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    if not blocks:
        return None
    return max(blocks, key=len)


def write_prompt_snapshot(d: Path) -> None:
    (d / "system_prompt.md").write_text(system_prompt(), encoding="utf-8")
