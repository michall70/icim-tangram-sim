"""目标图通道（planning/target_image.py）的离线测试。

不读 data/、不写磁盘、不连相机与机器人；全部用内存里画的合成目标图。
"""

import math
import unittest

import cv2 as cv
import numpy as np

from scenarios.tangram_puzzle.planning import target_image as ti
from scenarios.tangram_puzzle.puzzle_planner import plan_puzzle

# 标准七巧板平方布局（S=1、形心在原点），由 lattice_solver 解正方形得到后固化，
# 只用作测试夹具：七块面积和 = 1，且互不重叠。
LAYOUT = [
    ("L", [[0.0, 0.0], [-0.5, 0.5], [-0.5, -0.5]]),
    ("L", [[0.0, 0.0], [-0.5, -0.5], [0.5, -0.5]]),
    ("T", [[-0.25, 0.25], [0.0, 0.5], [-0.5, 0.5]]),
    ("Q", [[0.0, 0.0], [0.25, 0.25], [0.0, 0.5], [-0.25, 0.25]]),
    ("T", [[0.0, 0.0], [0.25, -0.25], [0.25, 0.25]]),
    ("P", [[0.5, -0.5], [0.5, 0.0], [0.25, 0.25], [0.25, -0.25]]),
    ("M", [[0.5, 0.5], [0.0, 0.5], [0.5, 0.0]]),
]

COLORS = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 255, 255),
          (255, 0, 255), (255, 255, 0), (0, 140, 255)]

SCALE_PX = 220.0
SIZE = (640, 480)


def render(scale=SCALE_PX, rotation=0.0, colors=None, drop=None, size=SIZE):
    """把 LAYOUT 画成一张"七块分色"的目标图（白底、无抗锯齿）。"""
    colors = colors or COLORS
    width, height = size
    canvas = np.full((height, width, 3), 255, np.uint8)
    cos, sin = math.cos(math.radians(rotation)), math.sin(math.radians(rotation))
    rotation_matrix = np.array([[cos, -sin], [sin, cos]])
    for index, (_, polygon) in enumerate(LAYOUT):
        if drop is not None and index == drop:
            continue
        points = np.asarray(polygon, float) @ rotation_matrix.T * scale
        points = points + np.array([width / 2.0, height / 2.0])
        cv.fillPoly(canvas, [np.rint(points).astype(np.int32)], colors[index % len(colors)],
                    lineType=cv.LINE_8)
    return canvas


class AreaQuadTest(unittest.TestCase):
    """TARGET_AREA_QUAD_MM：四角配置的解析、中心与朝向。"""

    def test_quad_from_config_none_when_empty(self):
        self.assertIsNone(ti.quad_from_config(None))
        self.assertIsNone(ti.quad_from_config([]))

    def test_quad_from_config_rejects_wrong_shape(self):
        with self.assertRaises(ti.TargetImageError):
            ti.quad_from_config([(0, 0), (1, 1), (2, 2)])

    def test_center_and_yaw_of_rotated_square(self):
        import math
        center = np.array([42.0, -356.0])
        half = 95.0
        angle = math.radians(8.0)
        rot = np.array([[math.cos(angle), -math.sin(angle)],
                        [math.sin(angle), math.cos(angle)]])
        corners = [tuple(center + rot @ np.array(v))
                   for v in ((-half, -half), (half, -half), (half, half), (-half, half))]
        quad = ti.quad_from_config(corners)
        self.assertIsNotNone(quad)
        got_center = ti.quad_center(quad)
        self.assertAlmostEqual(got_center[0], 42.0, places=6)
        self.assertAlmostEqual(got_center[1], -356.0, places=6)
        self.assertAlmostEqual(ti.quad_yaw_deg(quad), 8.0, places=6)

    def test_pieces_inside_user_quad(self):
        """给一个 190×190 的倾斜四边形，七块应判定为在区内。"""
        center = np.array([42.0, -356.0])
        half = 95.0
        angle = math.radians(8.0)
        rot = np.array([[math.cos(angle), -math.sin(angle)],
                        [math.sin(angle), math.cos(angle)]])
        corners = [tuple(center + rot @ np.array(v))
                   for v in ((-half, -half), (half, -half), (half, half), (-half, half))]
        quad = ti.quad_from_config(corners)
        placements, _ = ti.load_target_image(render()) if hasattr(ti, "load_target_image") else \
            ti.placements_from_image(render())
        # 用纯缩放的假单应把毫米映射到像素（与生产链同一个接口）
        mapper_matrix = np.array([[2.0, 0.0, 640.0], [0.0, -2.0, 480.0], [0.0, 0.0, 1.0]])
        pieces = ti.build_template_pieces(placements, 100.0, ti.quad_center(quad),
                                          0.0, mapper_matrix)
        inside, worst = ti.vertices_inside_quad(pieces, quad)
        self.assertTrue(inside, "最小内距 %.1f mm" % worst)
        self.assertGreater(worst, 2.0)


class TargetImageTest(unittest.TestCase):

    def test_detect_and_normalize_square(self):
        placements, summary = ti.placements_from_image(render())
        kinds = {kind: sum(1 for p in placements if p["kind"] == kind) for kind in ti.EXPECTED_COUNT}
        self.assertEqual(kinds, ti.EXPECTED_COUNT)
        total = sum(p["area"] for p in placements)
        self.assertAlmostEqual(total, 1.0, places=2)

    def test_rotated_target_same_kinds(self):
        placements, _ = ti.placements_from_image(render(rotation=30.0))
        kinds = {kind: sum(1 for p in placements if p["kind"] == kind) for kind in ti.EXPECTED_COUNT}
        self.assertEqual(kinds, ti.EXPECTED_COUNT)

    def test_all_same_color_is_rejected(self):
        with self.assertRaises(ti.TargetImageError):
            ti.placements_from_image(render(colors=[(0, 0, 255)] * 7))

    def test_missing_piece_is_rejected(self):
        with self.assertRaises(ti.TargetImageError):
            ti.placements_from_image(render(drop=3))

    def test_non_tangram_shape_is_rejected(self):
        canvas = np.full((480, 640, 3), 255, np.uint8)
        cv.rectangle(canvas, (100, 100), (500, 400), (0, 0, 255), -1)          # 一个大矩形
        cv.rectangle(canvas, (520, 100), (600, 180), (0, 200, 0), -1)          # 一个小方块
        with self.assertRaises(ti.TargetImageError):
            ti.placements_from_image(canvas)

    def test_anchor_is_roi_center(self):
        matrix = np.array([[2.0, 0.0, 100.0], [0.0, 2.0, 50.0], [0.0, 0.0, 1.0]])
        anchor = ti.anchor_from_roi((100, 100, 200, 100), matrix)
        self.assertAlmostEqual(anchor[0], 2.0 * 200 + 100, places=6)
        self.assertAlmostEqual(anchor[1], 2.0 * 150 + 50, places=6)

    def test_built_pieces_fit_inside_roi(self):
        placements, _ = ti.placements_from_image(render())
        matrix = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
        roi = (100, 100, 300, 260)
        anchor = ti.anchor_from_roi(roi, matrix)
        pieces = ti.build_template_pieces(placements, 100.0, anchor, 0.0, matrix)
        inside, bbox = ti.pixels_inside(pieces, roi, margin=2)
        self.assertTrue(inside, "图案应完整落在成品区，实际包围盒=%s" % (bbox,))

    def test_plan_puzzle_accepts_target_pieces(self):
        placements, _ = ti.placements_from_image(render())
        matrix = np.eye(3)
        roi = (100, 100, 300, 260)
        anchor = ti.anchor_from_roi(roi, matrix)
        pieces = ti.build_template_pieces(placements, 100.0, anchor, 0.0, matrix)

        # 原料侧用"同一形状整体平移到原料区"的假块，只验证接口与几何闭环。
        # 注意 grasp_point 也要一起平移：plan_puzzle 用 grasp_point 换算质心，
        # 只平移 vertices 会人为造出上百毫米的拟合残差。
        raw = []
        for piece in pieces:
            shifted = [(float(x) - 120.0, float(y) + 60.0) for x, y in piece.vertices]
            grasp = (float(piece.grasp_point[0]) - 120.0, float(piece.grasp_point[1]) + 60.0)
            center = (float(piece.center[0]) - 120.0, float(piece.center[1]) + 60.0)
            raw.append(type(piece)(piece_id=piece.piece_id, color_name=piece.color_name,
                                   shape=piece.shape, center=center,
                                   grasp_point=grasp, vertices=shifted,
                                   angle=piece.angle, contour=None))
        plans = plan_puzzle(raw, pieces, Mapper(matrix), (180.0, 0.0, 0.0), 350.0, 351.0, 0.0)
        self.assertEqual(len(plans), 7)
        for plan in plans:
            self.assertLess(plan.fit_error_mm, 1e-6)
            self.assertAlmostEqual(plan.delta_angle, 0.0, places=6)


class Mapper:
    def __init__(self, matrix):
        self.homography = matrix


if __name__ == "__main__":
    unittest.main()
