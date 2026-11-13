"""阶段三离线回归；不导入或连接机器人/相机设备。"""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2 as cv
import numpy as np
from common.calibration import CameraMapper
from scenarios.tangram_puzzle.config import CAMERA_MATRIX, DIST_COEFFS, HOMOGRAPHY, DETECT_PARAMS
from scenarios.tangram_puzzle.tangram_detector import TangramDetector, validate_detection
from scenarios.tangram_puzzle import main

DATA = Path(__file__).parent / "data" / "stage3_20261025"


# 测试用的合成图是按下面这套 ROI 生成的；把 ROI 钉在测试里，
# 这样 config 里的实际 ROI 随现场重拍而调整时，不会把检测逻辑的测试带崩。
TEST_ROIS = {"raw_roi": (160, 95, 625, 625), "template_roi": (795, 195, 485, 525)}



# 历史规格：这三个色在**自动曝光的旧图**上用的阈值。旧图（自动曝光）与新图（固定曝光）
# 的饱和度分布不同，同一组数无法同时适配；现网按新图定参数，测试用旧图，故在此钉住旧值。
_LEGACY_RANGES = {
    3: [((23, 90, 120), (34, 255, 255))],   # Yellow
    4: [((35, 60, 120), (84, 255, 255))],   # Green
    6: [((130, 35, 60), (167, 255, 255))],  # Purple
}

def _pinned_params():
    params = copy.deepcopy(DETECT_PARAMS)
    params.update(TEST_ROIS)
    for _sp in params["specs"]:
        if _sp["id"] in _LEGACY_RANGES:
            _sp["ranges"] = _LEGACY_RANGES[_sp["id"]]
    return params


class Stage3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapper = CameraMapper(CAMERA_MATRIX, DIST_COEFFS, HOMOGRAPHY)
        cls.raw, cls.template, cls.blank = [cls.mapper.undistort(cv.imread(str(DATA / f))) for f in
            ("1_Color.png", "template_Color_Color.png", "blank_Color.png")]

    def test_seven_distinct_pieces_in_both_scenes(self):
        d = TangramDetector(_pinned_params())
        for image, detect in ((self.raw,d.detect_raw_material),(self.template,d.detect_template)):
            pieces = detect(image)
            validate_detection(pieces, range(1,8), d.last_issues)
            centers = {p.piece_id:np.array(p.grasp_point) for p in pieces}
            self.assertGreater(np.linalg.norm(centers[2]-centers[3]), 80)
            self.assertEqual(next(p.shape for p in pieces if p.piece_id==7), "square")

    def test_empty_scene_and_wrong_region(self):
        d = TangramDetector(_pinned_params())
        self.assertEqual(d.detect_raw_material(self.blank), [])
        self.assertEqual(d.detect_template(self.blank), [])
        self.assertEqual(d.detect_template(self.raw), [])
        self.assertEqual(d.detect_raw_material(self.template), [])

    def test_truncated_piece_rejected(self):
        params = copy.deepcopy(DETECT_PARAMS)
        params["raw_roi"] = (0,0,600,720)
        d = TangramDetector(params)
        pieces = d.detect_raw_material(self.raw)
        self.assertNotIn(4, [p.piece_id for p in pieces])
        self.assertTrue(any("boundary" in issue for issue in d.last_issues))
        with self.assertRaises(ValueError):
            validate_detection(pieces, range(1,8), d.last_issues)

    def test_cross_color_conflict_rejected(self):
        params = copy.deepcopy(DETECT_PARAMS)
        extra = copy.deepcopy(params["specs"][1])
        extra.update(id=8, name="AmbiguousOrange")
        params["specs"].append(extra)
        d = TangramDetector(params)
        pieces = d.detect_raw_material(self.raw)
        self.assertFalse({2,8} & {p.piece_id for p in pieces})
        self.assertTrue(any("conflict" in issue for issue in d.last_issues))

    def test_wrong_resolution_rejected(self):
        with self.assertRaises(ValueError):
            TangramDetector(_pinned_params()).detect_raw_material(self.raw[::2,::2])

    def test_missing_piece_never_reaches_planner(self):
        incomplete = self.raw.copy()
        incomplete[190:325,430:555] = self.blank[190:325,430:555]
        with tempfile.TemporaryDirectory() as out:
            with patch.object(main,"RESULTS_DIR",out), patch.object(main,"plan_puzzle") as planner:
                with self.assertRaises(ValueError):
                    main._run_vision(TangramDetector(_pinned_params()), self.mapper, self.template, incomplete)
                planner.assert_not_called()

    def test_geometry_rejects_round_blob(self):
        image = np.full_like(self.blank,255)
        cv.circle(image,(350,350),60,(0,0,255),-1)
        d = TangramDetector(_pinned_params())
        self.assertEqual(d.detect_raw_material(image),[])
        self.assertTrue(any("poor polygon fit" in issue for issue in d.last_issues))

if __name__ == "__main__":
    unittest.main()
