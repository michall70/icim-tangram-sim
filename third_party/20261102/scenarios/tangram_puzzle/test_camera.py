"""相机流策略测试：USB 链路 → (fps, depth) 的自动降级规则。"""

import unittest

from common.realsense_camera import stream_plan


class StreamPlanTests(unittest.TestCase):
    def test_usb3_keeps_full_speed_and_depth(self):
        self.assertEqual(stream_plan("3.2", 30, True), (30, True))
        self.assertEqual(stream_plan("3.1", 30, True), (30, True))

    def test_usb2_degrades_to_color_only_15fps(self):
        # USB2 下 1280×720 彩色只有 6/10/15fps，且带不动双流
        self.assertEqual(stream_plan("2.1", 30, True), (15, False))
        self.assertEqual(stream_plan("2.0", 30, True), (15, False))
        self.assertEqual(stream_plan("1.1", 30, True), (15, False))

    def test_usb2_never_raises_fps(self):
        # 调用方自己要 6fps 就保持 6fps，不会被抬到 15
        self.assertEqual(stream_plan("2.1", 6, True), (6, False))

    def test_unknown_link_keeps_requested_plan(self):
        self.assertEqual(stream_plan(None, 30, True), (30, True))

    def test_depth_flag_respected_on_usb3(self):
        self.assertEqual(stream_plan("3.2", 30, False), (30, False))


if __name__ == "__main__":
    unittest.main()
