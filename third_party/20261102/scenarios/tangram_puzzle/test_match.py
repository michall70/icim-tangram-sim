"""match.py 的纯函数测试：目标图自动发现与错误提示映射。"""

import os
import tempfile
import time
import unittest

from scenarios.tangram_puzzle.match import _hint, choose_target, find_target


class FindTargetTests(unittest.TestCase):
    def test_sorted_by_name_not_mtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("tangram_03.png", "tangram_01.png", "tangram_02.png"):
                open(os.path.join(tmp, name), "wb").close()
            found, candidates = find_target(search_dirs=(tmp,))
            self.assertEqual([os.path.basename(p) for p in candidates],
                             ["tangram_01.png", "tangram_02.png", "tangram_03.png"])
            self.assertEqual(os.path.basename(found), "tangram_01.png")

    def test_ignores_non_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "readme.txt"), "w").close()
            found, candidates = find_target(search_dirs=(tmp,))
            self.assertIsNone(found)
            self.assertEqual(candidates, [])

    def test_missing_dir_is_skipped(self):
        found, candidates = find_target(search_dirs=(r"Z:\definitely-not-here",))
        self.assertIsNone(found)

    def test_accepts_jpg_and_bmp(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("a.JPG", "b.bmp", "c.jpeg"):
                open(os.path.join(tmp, name), "wb").close()
            found, candidates = find_target(search_dirs=(tmp,))
            self.assertEqual(len(candidates), 3)
            self.assertIsNotNone(found)


class ChooseTargetTests(unittest.TestCase):
    def test_number_selects_that_image(self):
        paths = ["F:\\a.png", "F:\\b.png", "F:\\c.png"]
        self.assertEqual(choose_target(paths, ask=lambda _: "2"), "F:\\b.png")

    def test_empty_answer_takes_first(self):
        paths = ["F:\\a.png", "F:\\b.png"]
        self.assertEqual(choose_target(paths, ask=lambda _: ""), "F:\\a.png")

    def test_invalid_then_valid(self):
        paths = ["F:\\a.png", "F:\\b.png"]
        answers = iter(["9", "abc", "1"])
        self.assertEqual(choose_target(paths, ask=lambda _: next(answers)), "F:\\a.png")


class HintTests(unittest.TestCase):
    def test_known_error_maps_to_hint(self):
        self.assertIn("高光", _hint("检测未通过：Purple: multiple substantial candidates"))
        self.assertIn("臂展", _hint("IKPreflightError: ID=2 unable to reach"))
        self.assertIn("相机链路", _hint("RuntimeError: Couldn't resolve requests"))

    def test_unknown_error_returns_none(self):
        self.assertIsNone(_hint("some totally unrelated failure"))


if __name__ == "__main__":
    unittest.main()
