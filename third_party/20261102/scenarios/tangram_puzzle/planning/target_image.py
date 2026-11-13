"""目标图（七块分色的示意图）→ 七块台面位姿。

## 为什么需要这一层

赛场的目标图**不是相机拍的**：它是赛方给的图片文件，像素坐标系与单应矩阵无关。
而现有 `puzzle_planner.plan_puzzle()` 的模板通道假定"模板 = 相机在拍照位拍的一帧"
（模板块与原料块套同一个单应）。直接拿目标图当模板，落点会整体错位。

本模块把目标图**只当作形状信息**使用：

    目标图 → 检测七块 → 归一化（正方形边长 S=1）→ 按 S_phys 缩放
            → 锚定到成品区中心 → 反算回相机像素 → 交给 plan_puzzle

也就是说：目标图决定"拼成什么形状"，实物尺度 `S_phys` 决定"多大"，
`template_roi` 的中心决定"放在哪"。三者解耦。

## 类别怎么认（不依赖颜色）

目标图里七块的颜色未必与实物一致，所以**不用颜色做身份**：

- 三角形 5 块按面积 4:2:1 分档 → 前二 L、第三 M、后二 T；
- 四边形 2 块按内角判正方形 Q / 平行四边形 P。

判据直接从 `planning/assign.py` 复用，因此目标图与实物走同一套分类，不会出现
"目标图判 Q、实物判 P"的口径漂移。

## 与"照片模板"的关系

本模块不替换旧路径：`TEMPLATE_PATH` 仍可指向实拍照片。它只是多提供一种
"目标图 → 模板块"的构造方式，两者产出同一种对象（`TangramPiece` 列表），
因此 `plan_puzzle` 之后的一切（几何门槛、IK 预检、执行）完全复用。
"""

import math

import cv2 as cv
import numpy as np

from scenarios.tangram_puzzle.planning import assign, pattern

# 归一化（S=1）下每类块的面积，来自件库定义：L=S²/4、M/Q/P=S²/8、T=S²/16
CANONICAL_AREA = {"L": 0.25, "M": 0.125, "T": 0.0625, "Q": 0.125, "P": 0.125}
EXPECTED_COUNT = {"L": 2, "M": 1, "T": 2, "Q": 1, "P": 1}

# 面积容差：目标图是矢量/位图绘制，边角有像素误差；超过这个比例说明"不是标准七巧板"
AREA_TOLERANCE = 0.08
# 同类两块之间的面积差（两个大三角必须等大）
PAIR_TOLERANCE = 0.06
# 彩色像素判据：排除白底（S 低）与黑线（V 低）
SAT_MIN = 60
VAL_MIN = 40
# 小碎块过滤：小于最大块这个比例的连通域忽略（防图例、标注文字）
SMALL_BLOB_RATIO = 0.15


class TargetImageError(RuntimeError):
    """目标图无法解析为标准七巧板：块数不对、形状不标准或尺寸不自洽。"""


def _colored_mask(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, np.array([0, SAT_MIN, VAL_MIN], np.uint8),
                      np.array([180, 255, 255], np.uint8))
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    return mask


def _order_ccw(points):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles)]


def _polygon_of(contour):
    """轮廓 → 3/4 个顶点的凸多边形；返回 None 表示形状不合法。"""
    hull = cv.convexHull(contour)
    perimeter = cv.arcLength(hull, True)
    for epsilon_ratio in (0.01, 0.02, 0.03, 0.05):
        approx = cv.approxPolyDP(hull, epsilon_ratio * perimeter, True)
        if len(approx) in (3, 4):
            return _order_ccw(approx.reshape(-1, 2))
    return None


def _dominant_colors(image, mask, limit=12):
    """从彩色像素里挑出若干互不相同的填充色（按像素数从多到少）。

    目标图是七块**分色**示意图，所以"一个颜色 = 一块"；先按颜色分开，
    相邻不同色的块就不会粘成一个连通域。颜色按 RGB 距离去重，容差 40。
    """
    # 注意：mask 是 0/255 的 uint8，必须用布尔索引（mask > 0），
    # 直接写 image[mask] 会被 numpy 当成"按值索引第 0 轴"，分配出上百 MB 的数组。
    pixels = image[mask > 0]
    if len(pixels) == 0:
        return []
    quantized = (pixels // 8) * 8
    colors, counts = np.unique(quantized.reshape(-1, 3), axis=0, return_counts=True)
    order = np.argsort(-counts)
    accepted = []
    minimum = max(30, int(0.02 * len(pixels)))
    for index in order:
        color = colors[index].astype(float)
        if counts[index] < minimum:
            continue
        if all(np.linalg.norm(color - other) > 40 for other in accepted):
            accepted.append(color)
        if len(accepted) >= limit:
            break
    return accepted


def detect_pieces(image, expect=7):
    """目标图 → 七块多边形（像素坐标）。

    按**填充色**分块（目标图是七块分色的示意图），因此相邻的块不会粘在一起，
    白底、黑线、灰色标注也不会混进来。
    """
    if image is None:
        raise TargetImageError("目标图无法读取")
    colored = _colored_mask(image)
    colors = _dominant_colors(image, colored)
    if not colors:
        raise TargetImageError("目标图里没有检测到任何彩色块")

    pieces = []
    for color in colors:
        delta = np.abs(image.astype(np.int16) - color.astype(np.int16)).sum(axis=2)
        blob = np.logical_and(colored, delta <= 90).astype(np.uint8)
        contours, _ = cv.findContours(blob, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        contours = [c for c in contours if cv.contourArea(c) > 0]
        if not contours:
            continue
        contour = max(contours, key=cv.contourArea)
        polygon = _polygon_of(contour)
        if polygon is None:
            continue
        pieces.append(polygon)

    if len(pieces) != expect:
        raise TargetImageError(
            "目标图里解析出 %d 块，应为 %d 块（七个填充色是否都不同？相邻同色块会粘在一起）"
            % (len(pieces), expect))
    return pieces


def classify(polygons):
    """多边形 → 类别标签（L/M/T/Q/P），复用 assign 的形状判据，不用颜色。"""
    triangles, quads = [], []
    for polygon in polygons:
        (triangles if len(polygon) == 3 else quads).append(polygon)
    if len(triangles) != 5 or len(quads) != 2:
        raise TargetImageError("形状构成不对：三角形 %d 块、四边形 %d 块，应为 5 块与 2 块"
                               % (len(triangles), len(quads)))

    def area_of(points):
        x, y = np.asarray(points, float).T
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    triangles.sort(key=area_of, reverse=True)
    kinds = ["L", "L", "M", "T", "T"]
    labeled = list(zip(triangles, kinds))
    for quad in quads:
        kind = assign.quad_kind(quad)
        if kind is None:
            raise TargetImageError("目标图里的四边形顶点不足四个，无法判定 Q/P")
        labeled.append((quad, kind))
    return labeled


def normalize(labeled, tolerance=AREA_TOLERANCE):
    """像素多边形 → 归一化布局（总面积 1、形心在原点），并校验件型面积。"""
    polygons = [np.asarray(polygon, float) for polygon, _ in labeled]
    total = sum(abs(_signed_area(p)) for p in polygons)
    if total <= 0:
        raise TargetImageError("目标图面积为 0")
    scale = 1.0 / math.sqrt(total)

    weights = np.array([abs(_signed_area(p)) for p in polygons])
    centroid = (np.array([p.mean(axis=0) for p in polygons]) * weights[:, None]).sum(axis=0) / weights.sum()

    placements = []
    counters = {}
    for (polygon, kind), points in zip(labeled, polygons):
        normalized = (points - centroid) * scale
        area = abs(_signed_area(normalized))
        expected = CANONICAL_AREA[kind]
        if abs(area - expected) / expected > tolerance:
            raise TargetImageError(
                "%s 块面积 %.4f 与件库 %.4f 相差 %.1f%%，目标图可能不是标准七巧板"
                % (kind, area, expected, 100.0 * abs(area - expected) / expected))
        counters[kind] = counters.get(kind, 0) + 1
        placements.append({"kind": kind, "index": counters[kind],
                           "polygon": [[float(x), float(y)] for x, y in normalized],
                           "area": area})

    counts = {kind: sum(1 for p in placements if p["kind"] == kind) for kind in EXPECTED_COUNT}
    if counts != EXPECTED_COUNT:
        raise TargetImageError("类别数量不对：%s，应为 %s" % (counts, EXPECTED_COUNT))
    for kind in ("L", "T"):
        areas = [p["area"] for p in placements if p["kind"] == kind]
        if abs(areas[0] - areas[1]) / max(areas) > PAIR_TOLERANCE:
            raise TargetImageError("两块 %s 面积不相等（%.4f / %.4f），目标图不标准"
                                   % (kind, areas[0], areas[1]))
    return placements


def _signed_area(points):
    x, y = np.asarray(points, float).T
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


class _Solution:
    """最小解对象：只提供 pattern.solution_to_pieces 需要的 `.placements`。"""

    def __init__(self, placements):
        self.placements = placements


def placements_from_image(image):
    """内存中的目标图（numpy 数组）→ (placements, 摘要)。"""
    polygons = detect_pieces(image)
    labeled = classify(polygons)
    placements = normalize(labeled)
    summary = {
        "size": [int(image.shape[1]), int(image.shape[0])],
        "kinds": {kind: sum(1 for p in placements if p["kind"] == kind) for kind in EXPECTED_COUNT},
        "areas": {p["kind"] + str(p["index"]): round(p["area"], 6) for p in placements},
        "total_area": round(sum(p["area"] for p in placements), 6),
    }
    return placements, summary


def load_target(path):
    """读取目标图并返回 (placements, 摘要)。placements 为 S=1 归一化布局。"""
    image = cv.imread(path, cv.IMREAD_COLOR)
    if image is None:
        raise TargetImageError("无法读取目标图: %s" % path)
    placements, summary = placements_from_image(image)
    summary["image"] = path
    return placements, summary


def anchor_from_roi(roi, homography):
    """成品区矩形 → 图案锚点（台面 mm）。锚点取矩形中心经单应映射的位置。"""
    x, y, w, h = roi
    matrix = np.asarray(homography, float)
    point = np.array([x + w / 2.0, y + h / 2.0, 1.0])
    mapped = matrix @ point
    if abs(mapped[2]) < 1e-9:
        raise TargetImageError("成品区中心映射到台面时分母接近零，单应矩阵无效")
    return [float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])]


def build_template_pieces(placements, scale_mm, anchor_mm, yaw_deg, homography,
                          piece_ids=None):
    """归一化布局 → 模板块列表（顶点为相机像素，供 plan_puzzle 使用）。

    顶点顺序统一按**像素坐标的极角升序**重排，与检测器 `_order_points` 一致。
    不这样做时，同一条边在"模板"与"原料"两侧的循环方向可能相反，而
    `plan_puzzle._rigid_fit` 只允许旋转、不允许镜像 ⇒ 同一块会被判成疑似翻面，
    拟合残差约等于块半径（首次接线实测 26—55mm）。
    """
    pieces = pattern.solution_to_pieces(_Solution(placements), scale_mm, anchor_mm,
                                        yaw_deg, homography, piece_ids=piece_ids)
    for piece in pieces:
        points = np.asarray(piece.vertices, dtype=float)
        order = _angle_order(points)
        piece.vertices = [(float(points[i, 0]), float(points[i, 1])) for i in order]
        if getattr(piece, "target_mm", None) is not None:
            piece.target_mm = [list(map(float, piece.target_mm[i])) for i in order]
    return pieces


def _angle_order(points):
    """按绕形心的极角升序给出顶点索引（与检测器 `_order_points` 同一规则）。"""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return np.argsort(angles)


def quad_from_config(points):
    """``config.TARGET_AREA_QUAD_MM`` → np.ndarray(4,2)；未配置或非法时返回 None。

    四角按**顺时针或逆时针顺序**给出即可（不必首尾重复）。允许写成
    ``[(x1,y1), (x2,y2), (x3,y3), (x4,y4)]`` 或 4 个长度 2 的序列。
    """
    if not points:
        return None
    quad = np.asarray(points, dtype=float)
    if quad.shape != (4, 2):
        raise TargetImageError(
            "TARGET_AREA_QUAD_MM 必须是 4 个 (x, y) 角点，当前形状 %s" % (quad.shape,))
    if not np.isfinite(quad).all():
        raise TargetImageError("TARGET_AREA_QUAD_MM 含非有限值")
    return quad


def quad_center(quad):
    """四边形中心（四角平均；矩形/平行四边形都适用）。"""
    points = np.asarray(quad, dtype=float).reshape(-1, 2)
    center = points.mean(axis=0)
    return [float(center[0]), float(center[1])]


def quad_yaw_deg(quad):
    """四边形第一条边相对机器人 X 轴的夹角（度）。"""
    points = np.asarray(quad, dtype=float).reshape(-1, 2)
    edge = points[1] - points[0]
    return float(np.degrees(np.arctan2(edge[1], edge[0])))


def tray_quad_mm(roi, homography):
    """成品区矩形 → 台面四角（mm）。透视下它是梯形/斜四边形，用它判断"是否真的在区内"。"""
    x, y, w, h = roi
    corners = np.array([[x, y, 1.0], [x + w - 1, y, 1.0],
                        [x + w - 1, y + h - 1, 1.0], [x, y + h - 1, 1.0]])
    mapped = (np.asarray(homography, float) @ corners.T).T
    return mapped[:, :2] / mapped[:, 2, None]


def tray_yaw_deg(roi, homography):
    """成品区在台面上的朝向（度）。图案与它平行时最省地方。"""
    quad = tray_quad_mm(roi, homography)
    edge = quad[1] - quad[0]
    return float(np.degrees(np.arctan2(edge[1], edge[0])))


def vertices_inside_quad(pieces, quad):
    """所有顶点是否落在成品区四边形内；返回 (是否全部在内, 最差带符号距离 mm)。"""
    import cv2 as _cv
    polygon = np.asarray(quad, dtype=np.float32)
    points = np.vstack([np.asarray(p.target_mm, float) for p in pieces])
    distances = [_cv.pointPolygonTest(polygon, (float(a), float(b)), True) for a, b in points]
    worst = float(min(distances)) if distances else 0.0
    return worst >= 0, worst


def fits_area(pieces, area_mm):
    """图案是否放得进给定尺寸的成品区（mm）。

    比"像素包围盒是否在 template_roi 内"更直接：图案按形心对中，所以只要外形
    尺寸不超过可用区尺寸即可。template_roi 画小/画偏都不会误判。
    """
    points = np.vstack([np.asarray(p.target_mm, dtype=float) for p in pieces])
    span = points.max(axis=0) - points.min(axis=0)
    limit = np.asarray(area_mm, dtype=float)
    return bool(span[0] <= limit[0] and span[1] <= limit[1]), \
        [float(span[0]), float(span[1])]


def pixels_inside(pieces, roi, margin=0):
    """所有模板块的像素顶点是否都在成品区矩形内（含边距）。"""
    x, y, w, h = roi
    x0, y0, x1, y1 = x + margin, y + margin, x + w - 1 - margin, y + h - 1 - margin
    points = np.vstack([np.asarray(p.vertices, float) for p in pieces])
    outside = (points[:, 0].min() < x0 or points[:, 1].min() < y0
               or points[:, 0].max() > x1 or points[:, 1].max() > y1)
    bbox = [float(points[:, 0].min()), float(points[:, 1].min()),
            float(points[:, 0].max()), float(points[:, 1].max())]
    return (not outside), bbox


def describe(placements, summary=None):
    area = sum(p["area"] for p in placements)
    kinds = {kind: sum(1 for p in placements if p["kind"] == kind) for kind in EXPECTED_COUNT}
    text = "目标图七块：%s，面积和 %.6f（应为 1.0）" % (kinds, area)
    if summary and summary.get("size"):
        text += "，图 %dx%d" % tuple(summary["size"])
    return text
