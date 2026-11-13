"""roi_pick 纯函数测试：托盘框 → 可用区框的换算。"""

import unittest

from scenarios.tangram_puzzle.tools.roi_pick import usable_from_tray


class UsableFromTrayTests(unittest.TestCase):
    def test_190_to_180_keeps_center(self):
        # 托盘 190mm 对应 570px；可用区 180mm 应为 540px，且与托盘同心
        box = usable_from_tray((700, 20, 570, 570), 190.0, 180.0)
        self.assertEqual(box[2], 540)
        self.assertEqual(box[3], 540)
        self.assertEqual(box[0], 700 + (570 - 540) // 2)
        self.assertEqual(box[1], 20 + (570 - 540) // 2)

    def test_same_size_is_identity(self):
        box = (100, 50, 400, 400)
        self.assertEqual(usable_from_tray(box, 190.0, 190.0), box)

    def test_smaller_usable_shrinks(self):
        box = usable_from_tray((0, 0, 380, 380), 190.0, 95.0)
        self.assertEqual((box[2], box[3]), (190, 190))


if __name__ == "__main__":
    unittest.main()
