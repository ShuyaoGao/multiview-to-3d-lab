"""拼图工具：把若干行图片拼成一张总览（带文字标签）。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _load(p, size):
    if isinstance(p, np.ndarray):
        img = p
    else:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            img = np.full((size, size, 3), 200, np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def grid(rows: list[tuple[str, list]], col_titles: list[str] | None = None, size: int = 160,
         label_w: int = 150) -> np.ndarray:
    ncol = max(len(r[1]) for r in rows)
    head = 24 if col_titles else 0
    H = head + size * len(rows)
    W = label_w + size * ncol
    out = np.full((H, W, 3), 255, np.uint8)
    if col_titles:
        for j, t in enumerate(col_titles):
            cv2.putText(out, t, (label_w + j * size + 4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    for i, (lab, imgs) in enumerate(rows):
        y = head + i * size
        for k, line in enumerate(lab.split("\n")):
            cv2.putText(out, line, (4, y + 20 + 16 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        for j, p in enumerate(imgs):
            out[y:y + size, label_w + j * size: label_w + (j + 1) * size] = _load(p, size)
            cv2.rectangle(out, (label_w + j * size, y), (label_w + (j + 1) * size - 1, y + size - 1), (220, 220, 220), 1)
    return out


def save(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
