# -*- coding: utf-8 -*-
"""镜头畸变仿真：把理想的针孔图像变成"真实相机拍出来的原图"。

为什么需要这一步
----------------
`config.CAMERA_MATRIX` 是**去畸变后**的针孔内参，`DIST_COEFFS` 是镜头畸变系数，
队友的 `CameraMapper.undistort()` 会对每一帧先做去畸变。仿真里 MuJoCo 渲染出的
图像本身**已经是理想针孔**（没有畸变）；如果直接交给他们，那一句 `undistort`
就会对一个已经去畸变好的图再做一次去畸变，等于凭空加了一层反向畸变
（1280×720、k1=0.154 时画面边缘几十像素）。

所以仿真侧要反过来：**先按畸变模型把理想图"打歪"**，他们的 `undistort`
就能精确地把它还原成理想针孔图。这样像素坐标和 `CAMERA_MATRIX` 严格自洽，
剩下的误差只可能来自单应矩阵或布局——这才是我们要检验的东西。

数学约定
--------
OpenCV 的畸变模型：把**理想**归一化坐标 (x, y) 映射到**实际**成像点 (x_d, y_d)::

    r² = x² + y²
    radial   = 1 + k1·r² + k2·r⁴ + k3·r⁶ + ...
    x_d = x·radial + 2·p1·x·y + p2·(r² + 2x²)
    y_d = y·radial + p1·(r² + 2y²) + 2·p2·x·y

`cv2.undistort` 做的正是它的逆：`U(p) = raw(D(p))`。
所以要让 `undistort(raw) == ideal`，必须 `raw(p) = ideal(D⁻¹(p))`，
即对每个**原图**像素 p，去理想图里采样 `D⁻¹(p)`。

`D⁻¹` 不自己写迭代：`cv2.undistortPointsIter` 就是 OpenCV 自己的反解，
拿它生成采样表最省事，也保证两边用的是同一个模型。
"""
from __future__ import annotations

import cv2
import numpy as np


def _pixel_grid(width: int, height: int) -> np.ndarray:
    """(height·width, 1, 2) 的 float32 像素坐标，顺序与 remap 一致。"""
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)                      # 都是 (H, W)
    return np.stack([gx, gy], axis=-1).reshape(-1, 1, 2)


def distort_maps(camera_matrix, dist_coeffs, width, height,
                 iterations=40, eps=1e-12):
    """返回 remap 用的 (map_x, map_y)：原图像素 → 理想图像素。"""
    grid = _pixel_grid(width, height)
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, iterations, eps)
    ideal = cv2.undistortPointsIter(grid, camera_matrix, dist_coeffs,
                                    None, camera_matrix, criteria)
    ideal = ideal.reshape(height, width, 2)
    return (ideal[..., 0].astype(np.float32),
            ideal[..., 1].astype(np.float32))


def distort_image(img, camera_matrix, dist_coeffs, *, border=cv2.BORDER_REPLICATE):
    """理想针孔图 → 带镜头畸变的原图（与真实相机读数同构）。"""
    if camera_matrix is None or dist_coeffs is None:
        return img
    h, w = img.shape[:2]
    mx, my = distort_maps(camera_matrix, dist_coeffs, w, h)
    return cv2.remap(img, mx, my, cv2.INTER_CUBIC, borderMode=border)


def undistort_points(points, camera_matrix, dist_coeffs):
    """(N,2) 原图像素 → 理想图像素。"""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, None, camera_matrix)
    return out.reshape(-1, 2)


# --------------------------------------------------------------------- selftest
def selftest(camera_matrix, dist_coeffs, width=1280, height=720, verbose=True):
    """验证 `undistort(distort(I)) == I`：圆点重投影误差应远小于 1 像素。

    返回 (max_err_px, mean_err_px)。
    """
    px = np.array([[300, 200], [640, 360], [980, 200],
                   [300, 520], [980, 520], [500, 700]], dtype=np.float32)

    ideal = np.zeros((height, width, 3), np.uint8)
    for x, y in px:
        cv2.circle(ideal, (int(x), int(y)), 3, (255, 255, 255), -1, cv2.LINE_AA)

    raw = distort_image(ideal, camera_matrix, dist_coeffs)
    back = cv2.undistort(raw, camera_matrix, dist_coeffs)

    gray = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    n, _, stats, centres = cv2.connectedComponentsWithStats(mask, 8)
    found = [centres[i] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 3]

    errs = []
    for x, y in px:
        if not found:
            break
        c = min(found, key=lambda c: (c[0] - x) ** 2 + (c[1] - y) ** 2)
        errs.append(float(np.hypot(c[0] - x, c[1] - y)))
    errs = np.array(errs) if errs else np.array([np.nan])
    if verbose:
        print(f"[lens] 圆点往返误差：max {np.nanmax(errs):.3f} px，"
              f"mean {np.nanmean(errs):.3f} px（{len(errs)}/{len(px)} 个点）")
    return float(np.nanmax(errs)), float(np.nanmean(errs))


if __name__ == "__main__":
    from .devices import load_config

    cfg = load_config()
    selftest(cfg.CAMERA_MATRIX, cfg.DIST_COEFFS)
