"""黑底白剪影图 → 归一化轮廓多边形。

赛事公告明确：目标图案"以图片文件形式提供"，白色剪影示例是**黑底白**（Fig 2 顶部）。
本模块只负责"把那张图变成一条规整的多边形"，不做求解。

流程：
    读图 → 估计背景色 → 取与背景差异最大的连通域（面积最大的一块）
        → 外轮廓 → 简化共线点 → 吸附到 45° 整数倍 → 平移到质心、缩放到面积 1

为什么要吸附 45°：七巧板所有边只沿 45° 的整数倍方向。图片里的轮廓总带噪声
（抗锯齿、缩放插值），不吸附就直接求解，会解出一个"差了半度"的近似摆法。

依赖 cv2；求解器本身不依赖本模块（求解器只吃多边形）。
"""

import math
import os

import cv2 as cv
import numpy as np

from scenarios.tangram_puzzle.tangram import geometry as geo

# 前景判定：像素与背景色的最大通道差超过该值
FOREGROUND_DISTANCE = 60

# 轮廓简化：epsilon 取周长的比例
APPROX_RATIO = 0.004

# 45° 吸附允许的最大角度偏差；超过就认为这张图不是七巧板图形
SNAP_TOLERANCE_DEG = 10.0

# 多边形顶点数上限：七巧板最多 13 个凸图形，顶点数不会太多
MAX_VERTICES = 14


class Silhouette:
    """一张剪影图解析出来的结果（多边形已归一化到面积 1、逆时针、质心在原点）。"""

    def __init__(self, polygon, raw_polygon, background, notes, source):
        self.polygon = polygon          # (N,2) 归一化多边形，S=1
        self.raw_polygon = raw_polygon  # 吸附前的像素多边形
        self.background = background    # 估计出的背景色 (B,G,R)
        self.notes = notes              # 解析过程中的提示（非致命）
        self.source = source            # 来源路径或描述
        self.scale = 1.0                # 归一化倍数：原始像素 / 归一化单位

    def scaled(self, scale):
        """把归一化多边形放大到实际尺寸（单位由调用方决定）。"""
        return np.asarray(self.polygon, dtype=float) * float(scale)

    def summary(self):
        return "顶点 %d，面积 %.6f，原始尺寸系数异常=%s" % (
            len(self.polygon), geo.area(self.polygon), bool(self.notes))


def _estimate_background(img):
    """用图像四条边的中位数估计背景色。"""
    border = np.concatenate([
        img[0, :, :].reshape(-1, 3), img[-1, :, :].reshape(-1, 3),
        img[:, 0, :].reshape(-1, 3), img[:, -1, :].reshape(-1, 3)])
    return np.median(border, axis=0)


def _largest_foreground_blob(img):
    """取与背景差异最大、面积最大的连通域，返回 (mask, notes)。"""
    notes = []
    background = _estimate_background(img)
    distance = np.max(np.abs(img.astype(np.int16) - background.astype(np.int16)), axis=2)
    mask = (distance > FOREGROUND_DISTANCE).astype(np.uint8) * 255
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    count, labels, stats, _ = cv.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        raise ValueError("图里没有找到与背景不同的区域")
    # 第 0 个是背景；其余按面积取最大
    areas = stats[1:, cv.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    if count > 2:
        others = np.sort(areas)[::-1]
        if len(others) > 1 and others[1] > 0.3 * others[0]:
            notes.append("图里有多个面积相近的区域，只取最大的一个（第二大是它的 %.0f%%）"
                         % (100.0 * others[1] / others[0]))
    blob = (labels == best).astype(np.uint8) * 255
    # 填掉内部空洞（剪影里可能印了字或反光）
    contours, _ = cv.findContours(blob, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    filled = np.zeros_like(blob)
    cv.drawContours(filled, contours, -1, 255, cv.FILLED)
    return filled, background, notes


def extract_polygon(img):
    """从图像里提取归一化轮廓。返回 Silhouette。"""
    if img is None:
        raise ValueError("图像为空")
    mask, background, notes = _largest_foreground_blob(img)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("没有找到轮廓")
    contour = max(contours, key=cv.contourArea)
    if cv.contourArea(contour) <= 0:
        raise ValueError("轮廓面积为 0")

    # 面积反推尺度：轮廓面积 = S^2
    scale = math.sqrt(float(cv.contourArea(contour)))

    perimeter = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, APPROX_RATIO * perimeter, True)
    polygon = geo.ensure_ccw(approx.reshape(-1, 2).astype(float))
    if len(polygon) > MAX_VERTICES:
        notes.append("简化后仍有 %d 个顶点，可能不是七巧板图形" % len(polygon))

    polygon = geo.simplify_collinear(polygon, tolerance=0.01 * scale)
    # 先归一到 S=1，再按"四档合法边长"合并被拆开的边——这一步是能否求解的关键
    polygon, factor0 = geo.normalize_polygon(polygon, target_area=1.0)
    polygon, merged = geo.snap_edge_lengths(polygon, tolerance=0.06)
    if merged:
        notes.append("合并了 %d 个多余顶点（边被拆段）" % merged)
    snapped, snapped_edges = geo.snap_edges_to_45(polygon, SNAP_TOLERANCE_DEG)
    if snapped is None:
        raise ValueError("轮廓的边不是 45° 的整数倍，这张图不是七巧板图形")
    if snapped_edges:
        notes.append("%d 条边被吸附到 45° 整数倍（原图有噪声）" % snapped_edges)

    polygon = geo.ensure_ccw(snapped)
    # 归一化：平移到质心、面积缩到 1
    normalized, factor = geo.normalize_polygon(polygon, target_area=1.0)

    result = Silhouette(normalized, polygon, background, notes, "image")
    result.scale = scale
    result.snapped_edges = snapped_edges
    return result


def load(path):
    """读一张图片并解析成轮廓。"""
    if not os.path.isfile(path):
        raise FileNotFoundError("找不到剪影图: %s" % path)
    img = cv.imread(path)
    if img is None:
        raise ValueError("无法读取图像（格式不支持？）: %s" % path)
    result = extract_polygon(img)
    result.source = path
    return result


def render(img, silhouette, solution=None, alpha=0.35):
    """把轮廓与解画在图上，便于人眼核对。"""
    canvas = img.copy() if img.ndim == 3 else cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    height, width = canvas.shape[:2]
    scale = silhouette.scale
    center = np.array([width / 2.0, height / 2.0])
    outline = silhouette.scaled(scale) + center
    cv.polylines(canvas, [np.rint(outline).astype(np.int32)], True, (0, 255, 255), 2)

    if solution is not None:
        palette = [(60, 60, 255), (60, 140, 255), (60, 255, 255), (60, 255, 60),
                   (255, 180, 60), (255, 100, 200), (140, 90, 60)]
        for index, item in enumerate(solution.placements):
            points = np.asarray(item["polygon"], dtype=float) * scale + center
            color = palette[index % len(palette)]
            overlay = canvas.copy()
            cv.fillPoly(overlay, [np.rint(points).astype(np.int32)], color)
            canvas = cv.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)
            cv.polylines(canvas, [np.rint(points).astype(np.int32)], True, color, 1)
            centroid = points.mean(axis=0)
            cv.putText(canvas, "%s%d" % (item["kind"], item["index"]),
                       (int(centroid[0]) - 14, int(centroid[1]) + 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
    return canvas
