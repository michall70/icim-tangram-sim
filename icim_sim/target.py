# -*- coding: utf-8 -*-
"""生成仿真用的"现场目标图案"图片（彩色七巧板示意图）。

比赛给的是**一张彩色图案图片**，程序要自己读它。仿真里也得有一张这样的图，
而且必须满足队友解析器的硬要求（`planning/target_image.py`）：

    · 七块**互相不同的纯色填充**（颜色距离 > 40 才算两种色）
    · 白底、无描边、无阴影、无渐变（描边会把一块切成多块）
    · 标准七巧板（面积必须正好是 L=S²/4、M/Q/P=S²/8、T=S²/16）

这里不"拍照片"，而是**直接按矢量画**：七块用标准七巧板坐标摆成经典正方形，
按模型量出来的实物尺度 S 缩放，白底填充。这样出来的图天然满足上面三条，
而且和模型里那七块板**严格同尺寸** —— 规划出来的落点就是模型该落的地方。

    python -m icim_sim.target                 # 默认画"正方形"，存 runs/target_sim.png
    python -m icim_sim.target --shape square --size 1000
    python -m icim_sim.target --s-mm 115.84   # 用队友的实测尺度而不是模型值
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .paths import RUNS

#: 经典七巧板正方形（边长 1，形心在原点）。从"标准正方形拼法"抄下来的精确坐标：
#:   L=1/4、M=Q=P=1/8、T=1/16，七块面积和 = 1 ⇒ 正好铺满 [-0.5, 0.5]²。
#: 坐标全部落在 0.25 的网格上，所以块与块之间没有缝。
SQUARE = [
    ("L", 1, [(-0.5, -0.5), (0.0, 0.0), (-0.5, 0.5)]),
    ("L", 2, [(-0.5, -0.5), (0.5, -0.5), (0.0, 0.0)]),
    ("M", 1, [(0.5, 0.0), (0.5, 0.5), (0.0, 0.5)]),
    ("T", 1, [(0.25, -0.25), (0.25, 0.25), (0.0, 0.0)]),
    ("T", 2, [(-0.25, 0.25), (0.0, 0.5), (-0.5, 0.5)]),
    ("P", 1, [(0.25, -0.25), (0.5, -0.5), (0.5, 0.0), (0.25, 0.25)]),
    ("Q", 1, [(0.0, 0.0), (0.25, 0.25), (0.0, 0.5), (-0.25, 0.25)]),
]

SHAPES = {"square": SQUARE}

#: 每块的颜色（RGB 0–1）。取的是模型里七块板自己的颜色，图片看起来就是
#: "这七块该拼成的样子"。只要求两两分得开（RGB 距离 > 40），与实物身份无关。
COLOURS = {
    ("L", 1): (0.85, 0.12, 0.12),    # 红
    ("L", 2): (0.15, 0.35, 0.90),    # 蓝
    ("M", 1): (0.95, 0.50, 0.10),    # 橙
    ("T", 1): (0.95, 0.85, 0.15),    # 黄
    ("T", 2): (0.60, 0.25, 0.80),    # 紫
    ("P", 1): (0.20, 0.70, 0.30),    # 绿
    ("Q", 1): (0.45, 0.28, 0.15),    # 咖啡
}

#: 归一化面积（S=1 时）：用来自检"这套坐标是不是标准七巧板"。
CANONICAL_AREA = {"L": 0.25, "M": 0.125, "T": 0.0625, "Q": 0.125, "P": 0.125}


def _area(points) -> float:
    p = np.asarray(points, float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def layout(shape: str = "square", s_mm: float = 116.364):
    """归一化图案 → 台面毫米多边形（形心在原点，单位 mm）。"""
    if shape not in SHAPES:
        raise SystemExit(f"没有这种图案 {shape!r}（有：{', '.join(SHAPES)}）")
    out = []
    for kind, index, poly in SHAPES[shape]:
        pts = np.asarray(poly, float) * float(s_mm)
        area = abs(_area(pts))
        want = CANONICAL_AREA[kind] * float(s_mm) ** 2
        if abs(area - want) > 1e-6 * max(1.0, want):
            raise SystemExit(f"{kind}{index} 面积 {area:.3f} 不是标准七巧板（应 {want:.3f}）")
        out.append({"kind": kind, "index": index,
                    "rgb": COLOURS[(kind, index)],
                    "polygon": [[float(x), float(y)] for x, y in pts],
                    "area": area})
    total = sum(item["area"] for item in out)
    if abs(total - float(s_mm) ** 2) > 1e-3:
        raise SystemExit(f"七块面积和 {total:.3f} ≠ S²={float(s_mm) ** 2:.3f}，图案不闭合")
    return out


def render(shape: str = "square", s_mm: float = 116.364, size: int = 1000,
           margin: float = 0.08):
    """画成图片（BGR uint8）。图案占画布 (1-2*margin) 的比例，白底。"""
    items = layout(shape, s_mm)
    span = float(s_mm)
    usable = size * (1.0 - 2.0 * margin)
    px_per_mm = usable / span
    canvas = np.full((size, size, 3), 255, np.uint8)
    cx = cy = size / 2.0
    for item in items:
        pts = np.asarray(item["polygon"], float) * px_per_mm
        pts[:, 0] += cx
        pts[:, 1] = cy - pts[:, 1]          # 图像 y 向下，台面 y 向上
        # RGB -> BGR：cv2 用 BGR 顺序，反一下即可
        bgr = tuple(int(round(255 * c)) for c in item["rgb"][::-1])
        cv2.fillPoly(canvas, [np.rint(pts).astype(np.int32)], bgr, lineType=cv2.LINE_8)
    return canvas, items, px_per_mm


def write(path, *, shape="square", s_mm=116.364, size=1000):
    """生成并保存图片 + 同名 `_layout.json`（毫米多边形，报告里对落点用）。"""
    img, items, px_per_mm = render(shape, s_mm, size)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    meta = {"shape": shape, "s_mm": float(s_mm), "size": int(size),
            "mm_per_px": 1.0 / px_per_mm,
            "pieces": items}
    meta_path = path.with_name(path.stem + "_layout.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    return path, meta_path, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description="生成仿真用的彩色目标图案图片")
    ap.add_argument("--shape", default="square", choices=sorted(SHAPES))
    ap.add_argument("--s-mm", type=float, default=None,
                    help="实物尺度；默认取模型算出来的值（七块面积和开根）")
    ap.add_argument("--size", type=int, default=1000, help="画布边长（px）")
    ap.add_argument("--out", default=str(RUNS / "target_sim.png"))
    args = ap.parse_args(argv)

    s_mm = args.s_mm
    if s_mm is None:
        from .calib import measure_calib

        s_mm = measure_calib().s_phys_mm
    path, meta_path, meta = write(args.out, shape=args.shape, s_mm=s_mm, size=args.size)
    print(f"[target] {args.shape}，S={s_mm:.3f} mm，{args.size}×{args.size} px"
          f"（{meta['mm_per_px']:.4f} mm/px）")
    for item in meta["pieces"]:
        print(f"  {item['kind']}{item['index']}  面积 {item['area']:8.1f} mm²")
    print(f"[target] 已存 {path}\n[target] 布局已存 {meta_path}")


if __name__ == "__main__":
    main()
