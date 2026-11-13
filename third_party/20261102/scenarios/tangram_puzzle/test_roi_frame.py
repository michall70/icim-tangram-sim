"""tools/roi_frame.py 的离线测试：几何检查、归属判定、毫米换算、绘图与结论。

全部用合成图，不读 data/、不写磁盘、不碰相机与机器人，可在任何环境跑：

    python -B -m unittest scenarios.tangram_puzzle.test_roi_frame
"""

import unittest

import cv2 as cv
import numpy as np

from scenarios.tangram_puzzle.config import DETECT_PARAMS
from scenarios.tangram_puzzle.tools import roi_frame as rf

# 测试用 ROI：左原料、右成品，与现场布局一致，并留出画面顶部与两框之间的空白。
PARAMS = dict(DETECT_PARAMS)
PARAMS["raw_roi"] = (0, 100, 620, 620)
PARAMS["template_roi"] = (660, 200, 620, 520)

SCALE_H = np.array([[0.4, 0.0, 0.0],
                    [0.0, -0.4, 0.0],
                    [0.0, 0.0, 1.0]])


def blank():
    return np.zeros((720, 1280, 3), np.uint8)


def green_square(x, y, size=120):
    """在黑色背景上画一个能被 Green 阈值命中的方块。"""
    img = blank()
    cv.rectangle(img, (x, y), (x + size, y + size), (0, 220, 0), -1)
    return img


class CheckRoisTest(unittest.TestCase):
    def test_disjoint_boxes_pass(self):
        problems, notes = rf.check_rois((1280, 720), PARAMS)
        self.assertEqual(problems, [])
        self.assertTrue(any("不重叠" in note for note in notes))

    def test_overlap_reported(self):
        params = dict(PARAMS)
        params["template_roi"] = (600, 0, 680, 720)
        # 未配置四角（传统模式）时，重叠必须是"问题"
        original = rf._quad_configured
        rf._quad_configured = lambda: False
        try:
            problems, _ = rf.check_rois((1280, 720), params)
        finally:
            rf._quad_configured = original
        self.assertTrue(any("重叠" in item for item in problems))
        self.assertGreater(rf.overlap_area(params["raw_roi"], params["template_roi"]), 0)
        self.assertEqual(rf.gap_between(PARAMS["raw_roi"], PARAMS["template_roi"]), 40)

    def test_overlap_is_only_note_in_quad_mode(self):
        """配置了成品区四角时，template_roi 不参与，重叠不该判失败。"""
        params = dict(PARAMS)
        params["template_roi"] = (600, 0, 680, 720)
        original = rf._quad_configured
        rf._quad_configured = lambda: True
        try:
            problems, notes = rf.check_rois((1280, 720), params)
        finally:
            rf._quad_configured = original
        self.assertFalse(any("重叠" in item for item in problems))
        self.assertTrue(any("四角配置" in note for note in notes))

    def test_out_of_bounds_reported(self):
        params = dict(PARAMS)
        params["template_roi"] = (1000, 0, 400, 720)   # 右缘 1400 > 1280
        problems, _ = rf.check_rois((1280, 720), params)
        self.assertTrue(any("超出" in item for item in problems))

    def test_image_size_mismatch_reported(self):
        problems, _ = rf.check_rois((640, 480), PARAMS)
        self.assertTrue(any("image_size" in item for item in problems))


class RoiMmTest(unittest.TestCase):
    def test_size_and_center_follow_homography(self):
        width_mm, height_mm, center, yaw_deg = rf.roi_mm((100, 100, 200, 100), SCALE_H)
        self.assertAlmostEqual(width_mm, 80.0, places=6)
        self.assertAlmostEqual(height_mm, 40.0, places=6)
        self.assertAlmostEqual(center[0], 80.0, places=6)
        self.assertAlmostEqual(center[1], -60.0, places=6)
        # 轴对齐框在纯缩放的台面上应约为 0°，即"与机器人 X 轴平行"
        self.assertAlmostEqual(yaw_deg, 0.0, places=6)


class CoverageTest(unittest.TestCase):
    def test_blob_inside_raw_roi_is_fully_inside(self):
        rows = rf.measure_coverage(green_square(200, 200), PARAMS)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["region"], "raw")
        self.assertTrue(rows[0]["fully_inside"])

    def test_blob_in_the_gap_is_outside(self):
        rows = rf.measure_coverage(green_square(630, 300, size=10), PARAMS)
        # 10px 方块低于实体面积门槛，应被当成噪声忽略
        self.assertEqual(rows, [])

    def test_blob_above_raw_box_is_outside(self):
        rows = rf.measure_coverage(green_square(100, 20, size=80), PARAMS)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["region"], "outside")

    def test_blob_inside_template_box(self):
        rows = rf.measure_coverage(green_square(800, 300, size=120), PARAMS)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["region"], "template")
        self.assertTrue(rows[0]["fully_inside"])

    def test_blob_crossing_border_is_partial(self):
        rows = rf.measure_coverage(green_square(540, 300, size=140), PARAMS)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["region"], "raw")
        self.assertFalse(rows[0]["fully_inside"])

    def test_small_colour_noise_ignored(self):
        self.assertEqual(rf.measure_coverage(green_square(200, 200, size=30), PARAMS), [])


class AnnotateTest(unittest.TestCase):
    def test_boxes_and_blobs_are_drawn_without_touching_input(self):
        img = green_square(200, 200)
        before = img.copy()
        rows = rf.measure_coverage(img, PARAMS)
        vis = rf.annotate(img, PARAMS, rows, SCALE_H)
        self.assertEqual(vis.shape, img.shape)
        self.assertTrue(np.array_equal(img, before))
        self.assertFalse(np.array_equal(vis, img))
        # 原料框左边框应被涂成绿色
        self.assertTrue(np.array_equal(vis[360, 0][:3], np.array(rf.RAW_COLOR)))
        # 图例条落在最后 30 行
        self.assertTrue(np.all(vis[700, 5] == 0))

    def test_box_touching_top_edge_keeps_label_in_frame(self):
        params = dict(PARAMS)
        params["raw_roi"] = (0, 0, 620, 720)
        vis = rf.annotate(blank(), params, [], SCALE_H)
        self.assertEqual(vis.shape[:2], (720, 1280))


class ReportTest(unittest.TestCase):
    def test_clean_frame_passes(self):
        report = rf.build_report(green_square(200, 200), PARAMS, SCALE_H)
        self.assertEqual(report["problems"], [])
        self.assertEqual(report["coverage"]["raw"], 1)
        self.assertEqual(report["coverage"]["outside"], 0)
        self.assertEqual(report["coverage"]["partial"], 0)
        self.assertFalse(rf.print_report(report))

    def test_frame_with_blob_outside_fails(self):
        report = rf.build_report(green_square(100, 20, size=80), PARAMS, SCALE_H)
        self.assertEqual(report["coverage"]["outside"], 1)
        self.assertTrue(rf.print_report(report))


if __name__ == "__main__":
    unittest.main()
