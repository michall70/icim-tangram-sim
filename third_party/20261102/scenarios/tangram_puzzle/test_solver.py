"""求解器回归测试（不连接相机、不连接机器人）。

覆盖三层：
1. **件库几何**：七块面积和必须等于 1；边长只取四个档位；手性固定。
2. **几何判定**：点在多边形内、凸多边形重叠（贴边不算）、45° 吸附。
3. **整链求解**：单位正方形必须解出并通过自检；正五边形必须被拒绝。

注意：求解器当前对**部分**图形尚不收敛（详见 docs/已知条件.md 与备赛对话记录），
所以这里只断言单位正方形这一个已知可解的图形，不对其余图形做"必然解出"的断言——
把未验证的东西写成断言，会让测试变成装饰。
"""

import math
import unittest

import numpy as np

from scenarios.tangram_puzzle.tangram import geometry as geo
from scenarios.tangram_puzzle.config import HOMOGRAPHY
from scenarios.tangram_puzzle.planning import assign, pattern
from scenarios.tangram_puzzle.tangram_detector import TangramPiece
from scenarios.tangram_puzzle.tangram import lattice_solver, piece_set, solver, verify


def _square(scale=1.0):
    return geo.ensure_ccw(np.array([[0.0, 0.0], [scale, 0.0],
                                    [scale, scale], [0.0, scale]]))


class PieceGeometryTests(unittest.TestCase):
    """件库本身的几何不变量。"""

    def test_area_sum_is_one(self):
        forms = solver.piece_forms()
        self.assertEqual(len(forms), 7)
        total = sum(geo.area(form["poly"]) for form in forms)
        self.assertAlmostEqual(total, 1.0, places=12)

    def test_kind_counts(self):
        forms = solver.piece_forms()
        counts = {}
        for form in forms:
            counts[form["kind"]] = counts.get(form["kind"], 0) + 1
        self.assertEqual(counts, {"L": 2, "M": 1, "T": 2, "Q": 1, "P": 1})

    def test_edge_lengths_only_four_levels(self):
        """所有边长只能是 S·{1, 0.7071, 0.5, 0.3536} 这四个档位。"""
        allowed = [1.0, 1.0 / math.sqrt(2.0), 0.5, 1.0 / (2.0 * math.sqrt(2.0))]
        for form in solver.piece_forms():
            for edge in geo.edge_lengths(form["poly"]):
                nearest = min(allowed, key=lambda want: abs(want - edge))
                self.assertLess(abs(nearest - edge), 1e-9, "边长 %.6f 不在四个档位内" % edge)

    def test_piece_set_matches_solver(self):
        """piece_set 的标称系数与求解器实际多边形必须逐边一致。"""
        forms = {form["kind"]: form for form in solver.piece_forms()}
        expect_edges = {
            "L": [0.5 ** 0.5, 0.5 ** 0.5, 1.0],                      # 两条直角边 + 斜边
            "M": [0.5, 0.5, 1.0 / math.sqrt(2.0)],
            "T": [1.0 / (2.0 * math.sqrt(2.0))] * 2 + [0.5],
            "Q": [1.0 / (2.0 * math.sqrt(2.0))] * 4,
            "P": [1.0 / (2.0 * math.sqrt(2.0))] * 2 + [0.5] * 2,
        }
        for kind, form in forms.items():
            expected = sorted(expect_edges[kind])
            actual = sorted(geo.edge_lengths(form["poly"]))
            self.assertEqual(len(expected), len(actual), "%s 边数不符" % kind)
            for want, got in zip(expected, actual):
                self.assertAlmostEqual(want, got, places=9, msg="%s 边长不符" % kind)
        # 面积系数要按件数加权求和，才应当等于 1
        total = sum(piece_set.AREA_COEF[kind] * piece_set.COUNT[kind]
                    for kind in piece_set.COUNT)
        self.assertAlmostEqual(total, 1.0, places=12)


class GeometryTests(unittest.TestCase):
    """多边形判定的基本正确性。"""

    def test_area_and_centroid(self):
        square = _square(1.0)
        self.assertAlmostEqual(geo.area(square), 1.0, places=12)
        center = geo.centroid(square)
        self.assertAlmostEqual(float(center[0]), 0.5, places=12)
        self.assertAlmostEqual(float(center[1]), 0.5, places=12)

    def test_point_in_polygon(self):
        square = _square(1.0)
        self.assertTrue(geo.point_in_polygon((0.5, 0.5), square))
        self.assertFalse(geo.point_in_polygon((1.5, 0.5), square))
        self.assertTrue(geo.point_in_polygon((0.0, 0.0), square))   # 边界算在内

    def test_overlap_touching_is_not_overlap(self):
        left = geo.ensure_ccw(np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]))
        right = geo.ensure_ccw(np.array([[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]))
        self.assertFalse(geo.convex_overlap(left, right))
        shifted = geo.translate(right, (-0.1, 0.0))
        self.assertTrue(geo.convex_overlap(left, shifted))

    def test_snap_to_45(self):
        noisy = np.array([[0.0, 0.0], [1.0, 0.03], [1.02, 1.0], [0.0, 0.98]])
        snapped, count = geo.snap_edges_to_45(noisy)
        self.assertIsNotNone(snapped)
        self.assertGreaterEqual(count, 1)
        for edge in geo.edge_lengths(snapped):
            self.assertGreater(edge, 0.0)

    def test_snap_rejects_non_45(self):
        """一个边方向明显不是 45° 倍数的图形应当被拒绝。"""
        bad = np.array([[0.0, 0.0], [1.0, 0.0], [0.4, 0.9], [0.0, 1.0]])
        snapped, _ = geo.snap_edges_to_45(bad, tolerance_deg=5.0)
        self.assertIsNone(snapped)


class LatticeSolveTests(unittest.TestCase):
    """格点版求解器（进阶主用）：正方形与公告 Fig 2 的三角形都必须解出并通过自检。"""

    @classmethod
    def setUpClass(cls):
        cls.square = _square(1.0)
        cls.square_result = lattice_solver.solve(cls.square, time_limit=120.0)
        triangle = geo.ensure_ccw(np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]]))
        cls.triangle, _ = geo.normalize_polygon(triangle, target_area=1.0)
        cls.triangle_result = lattice_solver.solve(cls.triangle, time_limit=120.0)

    def test_square_solved(self):
        self.assertIsNotNone(self.square_result, "单位正方形必须解出")
        self.assertEqual(len(self.square_result.placements), 7)
        report = verify.verify(self.square_result.placements, self.square_result.target)
        self.assertTrue(report["ok"], verify.describe(report))

    def test_triangle_solved(self):
        self.assertIsNotNone(self.triangle_result, "公告 Fig 2 的三角形必须解出")
        self.assertEqual(len(self.triangle_result.placements), 7)
        report = verify.verify(self.triangle_result.placements, self.triangle_result.target)
        self.assertTrue(report["ok"], verify.describe(report))

    def test_triangle_pieces_valid(self):
        for item in self.triangle_result.placements:
            self.assertIn(item["kind"], ("L", "M", "T", "Q", "P"))
            self.assertIn(item["angle"], lattice_solver.ROTATIONS)


class SolveSquareTests(unittest.TestCase):
    """旧版求解器（顶点锚定）保留为基线：单位正方形仍须可解。"""

    @classmethod
    def setUpClass(cls):
        cls.target = _square(1.0)
        cls.result = solver.solve(cls.target, max_nodes=200000, time_limit=120.0)

    def test_solved(self):
        self.assertIsNotNone(self.result, "单位正方形是七块的标准拼法，必须解出")
        self.assertEqual(len(self.result.placements), 7)

    def test_pieces_are_valid(self):
        for item in self.result.placements:
            self.assertIn(item["kind"], ("L", "M", "T", "Q", "P"))
            self.assertIn(item["angle"], solver.ROTATIONS)

    def test_verification_passes(self):
        report = verify.verify(self.result.placements, self.target)
        self.assertTrue(report["ok"], verify.describe(report))
        self.assertAlmostEqual(report["area_sum"], 1.0, places=9)
        self.assertEqual(report["overlaps"], [])
        self.assertTrue(report["inside_all"])


class RejectTests(unittest.TestCase):
    """非七巧板图形必须被拒绝，而不是给一个近似解。"""

    def test_pentagon_rejected(self):
        angles = [2.0 * math.pi * k / 5.0 + 0.3 for k in range(5)]
        polygon = geo.ensure_ccw(np.array([[math.cos(a), math.sin(a)] for a in angles]))
        normalized, _ = geo.normalize_polygon(polygon, target_area=1.0)
        result = solver.solve(normalized, max_nodes=20000, time_limit=8.0)
        self.assertIsNone(result)

    def test_wrong_area_rejected_immediately(self):
        """面积不是 1 的轮廓应当立刻被拒绝，不做无谓搜索。"""
        big = _square(1.4)
        self.assertIsNone(solver.solve(big, time_limit=2.0))


class PatternTests(unittest.TestCase):
    """图案 → 台面：缩放/旋转/平移必须可逆，且与单应映射往返一致。"""

    @classmethod
    def setUpClass(cls):
        triangle = geo.ensure_ccw(np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]]))
        cls.target, _ = geo.normalize_polygon(triangle, target_area=1.0)
        cls.solution = lattice_solver.solve(cls.target, time_limit=120.0)
        cls.slots = pattern.solution_to_pieces(
            cls.solution, 116.0, (-40.0, -330.0), 0.0, HOMOGRAPHY)

    def test_transform_is_pure_similarity(self):
        # 旋转是绕**原点**做的（求解结果的质心就在原点），所以这里用居中的正方形
        square = _square(1.0)
        square = square - geo.centroid(square)
        moved = pattern.transform_polygon(square, 100.0, (10.0, -20.0), 90.0)
        # 面积按尺度的平方变化，旋转不改变面积
        self.assertAlmostEqual(geo.area(moved), 100.0 * 100.0, places=6)
        # 形心落在锚点上（正方形形心本来就在原点）
        center = geo.centroid(moved)
        self.assertAlmostEqual(float(center[0]), 10.0, places=6)
        self.assertAlmostEqual(float(center[1]), -20.0, places=6)

    def test_round_trip_through_homography(self):
        from scenarios.tangram_puzzle.puzzle_planner import _transform
        for piece in self.slots:
            back = _transform(piece.vertices, HOMOGRAPHY)
            error = float(np.abs(back - np.asarray(piece.target_mm)).max())
            self.assertLess(error, 1e-6, "往返误差 %.3g mm" % error)

    def test_pattern_bounds_reasonable(self):
        x0, y0, x1, y1 = pattern.pattern_bounds(self.solution, 116.0, (-40.0, -330.0), 0.0)
        self.assertLess(x0, x1)
        self.assertLess(y0, y1)
        # 图案整体尺度应当与 S_phys 同量级（不会是一个点或几百毫米）
        self.assertGreater(x1 - x0, 50.0)
        self.assertLess(x1 - x0, 400.0)


class AssignTests(unittest.TestCase):
    """解 ↔ 实物：分类、双射、以及两条必须拒绝的情形。"""

    @classmethod
    def setUpClass(cls):
        triangle = geo.ensure_ccw(np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]]))
        target, _ = geo.normalize_polygon(triangle, target_area=1.0)
        cls.solution = lattice_solver.solve(target, time_limit=120.0)
        cls.slots = pattern.solution_to_pieces(cls.solution, 116.0, (0.0, 0.0), 0.0, HOMOGRAPHY)
        cls.detected = []
        for index, piece in enumerate(cls.slots):
            # 用 pixel 顶点（而不是 target_mm）：真实检测器给出的就是像素多边形。
            # 2026-11-02 修了"图像 y 向下→台面 y 向上"的翻转后，target_mm 与 pixel
            # 之间多了一次镜像关系，这里若继续拿 target_mm 当像素用，夹具自身就矛盾了。
            cls.detected.append(TangramPiece(
                piece_id=10 + index, color_name="X%d" % index, shape=piece.shape,
                center=(0, 0), grasp_point=(20 * index, 10 * index),
                vertices=piece.vertices, angle=0.0, contour=None))

    def test_classify_counts(self):
        labeled = assign.classify_pieces(self.detected)
        counts = {}
        for _, kind in labeled:
            counts[kind] = counts.get(kind, 0) + 1
        self.assertEqual(counts, {"L": 2, "M": 1, "T": 2, "Q": 1, "P": 1})

    def test_assignment_is_bijection(self):
        mapping, report = assign.assign_ids(self.slots, self.detected)
        self.assertEqual(len(mapping), 7)
        self.assertEqual(len(set(mapping.values())), 7)
        self.assertIn("counts", report)

    def test_missing_piece_rejected(self):
        with self.assertRaises(assign.AssignError):
            assign.assign_ids(self.slots, self.detected[:-1])

    def test_quad_kind_uses_interior_angles(self):
        """方块掩模被高光咬掉一角时边长比会失真，仍应按内角判成 Q。"""
        skewed_square = [[0, 0], [100, 0], [100, 104], [3, 100]]   # 长短边比约 0.72
        parallelogram = [[0, 0], [100, 0], [140, 60], [40, 60]]    # 45/135°
        self.assertEqual(assign.quad_kind(skewed_square), "Q")
        self.assertEqual(assign.quad_kind(parallelogram), "P")

    def test_mirrored_parallelogram_rejected(self):
        broken = []
        for piece in self.detected:
            vertices = piece.vertices
            if piece.shape == "parallelogram":
                array = np.asarray(vertices, dtype=float).copy()
                array[:, 0] *= -1
                vertices = [[float(a), float(b)] for a, b in array]
            broken.append(TangramPiece(piece_id=piece.piece_id, color_name=piece.color_name,
                                       shape=piece.shape, center=(0, 0),
                                       grasp_point=piece.grasp_point, vertices=vertices,
                                       angle=0.0, contour=None))
        with self.assertRaises(assign.AssignError):
            assign.assign_ids(self.slots, broken)


if __name__ == "__main__":
    unittest.main(verbosity=2)
