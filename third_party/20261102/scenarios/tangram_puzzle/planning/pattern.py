"""图案 → 台面：把归一化的七块解（S=1）换算成台面上的目标位姿。

## 为什么需要这一层

求解器给出的是**归一化坐标**（正方形边长 S=1、质心在原点）。要落到机器人上还差三件事：

1. **绝对尺度**：用实物量出来的 `S_phys`（mm）乘上去。现场另发一副七巧板，这个数必须
   运行时测（`tools/measure_pieces.py`）；
2. **摆在哪**：图案的绝对位置由人决定——锚点 `anchor_mm`（建议取成品区托盘中心）；
3. **朝哪个方向**：目标图没有绝对朝向，`yaw_deg` 由调用方给。建议**保持目标图自身的
   朝向**，评委按图比对时最直观。

## 与现有规划器的衔接

`puzzle_planner.plan_puzzle()` 的输入是"模板图上的 `TangramPiece` 列表"（顶点为**像素**坐标，
内部再经单应映射到毫米）。所以这里不在毫米空间另写一套位姿计算，而是把解**反向映射回像素**，
直接复用已验证的 `plan_puzzle`——避免两份位姿逻辑长期漂移。

## 与"外轮廓优先"的关系

本文件只负责"每块应该落在哪"，不负责"缝对齐得严不严"。白剪影没有内部界线，评委只看外轮廓，
所以摆放策略是**外轮廓优先**：`anchor_mm` 与 `yaw_deg` 决定外轮廓的位置与朝向，内部缝隙让给
件尺寸公差（该副七巧板实测三条本应等长的边差 6.3%，零缝是不可能的）。
"""

import math

import numpy as np

from scenarios.tangram_puzzle.tangram_detector import TangramDetector, TangramPiece

# kind -> 探测器使用的形状名（必须与 config.DETECT_PARAMS.specs 的 shape 取值一致）
KIND_SHAPE = {"L": "triangle", "M": "triangle", "T": "triangle",
              "Q": "square", "P": "parallelogram"}


def _rotation(degrees):
    rad = math.radians(degrees)
    cos, sin = math.cos(rad), math.sin(rad)
    return np.array([[cos, -sin], [sin, cos]])


def transform_polygon(polygon, scale_mm, anchor_mm=(0.0, 0.0), yaw_deg=0.0):
    """归一化多边形 → 台面毫米坐标。

    顺序：缩放（面积单位 → mm）→ **翻转 y** → 绕原点旋转 → 平移到锚点。

    **为什么要翻转 y**：目标图是"一张图片"，像素坐标 y 向下；机器人台面坐标 y 向上，
    两者手性相反。不翻转等于把图案做成镜像——实测后果两个：
    ① 规划出的图案与电子图成镜像关系；
    ② 唯一有手性的件（平行四边形）永远被判"疑似翻面"（fit 很大、mirror 很小）。
    """
    points = np.asarray(polygon, dtype=float).reshape(-1, 2).copy()
    points[:, 1] = -points[:, 1]          # 图像 y 向下 → 台面 y 向上
    scaled = points * float(scale_mm)
    rotated = scaled @ _rotation(yaw_deg).T
    return rotated + np.asarray(anchor_mm, dtype=float)


def _angle_order(points):
    """按绕形心的极角升序给出顶点索引（与检测器 `_order_points` 同一规则，在像素坐标里做）。"""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return np.argsort(angles)


def solution_to_pieces(solution, scale_mm, anchor_mm, yaw_deg, homography,
                       piece_ids=None):
    """把求解结果转成 `puzzle_planner.plan_puzzle` 直接可用的模板块列表。

    solution      : lattice_solver.SolveResult（或含 placements 的同构对象）
    scale_mm      : 实物尺度 S_phys（mm）
    anchor_mm     : 图案形心的台面位置（mm），建议取成品区托盘中心
    yaw_deg       : 图案朝向（度）
    homography    : 像素 → 毫米的单应矩阵；用于把毫米位姿反算回像素
    piece_ids     : 可选 {slot 序号: piece_id}，由 assign 模块给出；缺省按顺序编号

    返回 TangramPiece 列表，**顶点为像素坐标**（因为 plan_puzzle 内部会再做一次单应映射）。
    毫米位姿同时挂在返回对象的 `target_mm` 属性上，便于输出与核对。
    """
    matrix = np.asarray(homography, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("单应矩阵无效，无法把图案映射回像素")
    inverse = np.linalg.inv(matrix)

    # 先算出全部块的毫米多边形（锚点先放原点），再按**外接框中心**平移到锚点。
    # 为什么不用形心：不对称图形（本例形心比外接框中心偏 15.5mm）按形心对齐会出现
    # "左边还有空间、右边却顶出托盘"——外接框中心对齐才是"放得下/放不下"的正确基准。
    raw_mm = [transform_polygon(item["polygon"], scale_mm, (0.0, 0.0), yaw_deg)
              for item in solution.placements]
    all_points = np.vstack(raw_mm)
    bbox_center = (all_points.min(axis=0) + all_points.max(axis=0)) / 2.0
    shift = np.asarray(anchor_mm, dtype=float).reshape(2) - bbox_center

    pieces = []
    for index, item in enumerate(solution.placements):
        polygon_mm = raw_mm[index] + shift
        homogeneous = np.column_stack((polygon_mm, np.ones(len(polygon_mm)))) @ inverse.T
        if np.any(np.abs(homogeneous[:, 2]) < 1e-9):
            raise ValueError("图案位姿反投影到像素时分母接近零")
        polygon_px = homogeneous[:, :2] / homogeneous[:, 2, None]

        center_mm = polygon_mm.mean(axis=0)
        # center / grasp_point 必须是**像素**：plan_puzzle 的 _piece_geometry 会把
        # grasp_point 再过一次单应映射。这里若填毫米值，会被当成像素再映射一次，
        # 拟合残差直接变成几百毫米（目标图通道首次接线时实测 466—574mm）。
        center_px = polygon_px.mean(axis=0)
        piece_id = (piece_ids or {}).get(index, index + 1)
        piece = TangramPiece(
            piece_id=piece_id,
            color_name=item["kind"] + str(item["index"]),
            shape=KIND_SHAPE[item["kind"]],
            center=(int(round(center_px[0])), int(round(center_px[1]))),
            grasp_point=(int(round(center_px[0])), int(round(center_px[1]))),
            vertices=[(float(x), float(y)) for x, y in polygon_px],
            angle=float(TangramDetector._polygon_angle(polygon_px)),
            contour=None,
        )
        # 便于调用方核对：这是该块在台面上的真实毫米位姿
        piece.target_mm = [[float(x), float(y)] for x, y in polygon_mm]
        piece.target_center_mm = [float(center_mm[0]), float(center_mm[1])]
        pieces.append(piece)
    return pieces


def pattern_bounds(solution, scale_mm, anchor_mm, yaw_deg):
    """图案在台面上的包围盒（mm），用于检查是否落在托盘/可达范围内。"""
    all_points = np.vstack([
        transform_polygon(item["polygon"], scale_mm, anchor_mm, yaw_deg)
        for item in solution.placements])
    return (float(all_points[:, 0].min()), float(all_points[:, 1].min()),
            float(all_points[:, 0].max()), float(all_points[:, 1].max()))
