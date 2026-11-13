"""规划数学与模拟RPC回归；不连接相机/控制器。"""
import copy
import unittest
from dataclasses import asdict
from pathlib import Path
import cv2 as cv
import numpy as np
from common.calibration import CameraMapper
from common.robot_client import CobotClient, IKPreflightError
from scenarios.tangram_puzzle.puzzle_planner import (
    plan_puzzle, normalize_angle, _transform, _rotation, validate_plan_geometry,
    rotation_offset, predicted_piece_xy)
from scenarios.tangram_puzzle.tangram_detector import TangramPiece, TangramDetector
from scenarios.tangram_puzzle.tool_offset_check import solve_suction_offset
from scenarios.tangram_puzzle.config import CAMERA_MATRIX,DIST_COEFFS,HOMOGRAPHY
from scenarios.tangram_puzzle import main



# 与 test_stage3 一致：合成图是按这套 ROI 生成的，把 ROI 钉住，避免 config 调整影响测试。
_TEST_ROIS = {"raw_roi": (160, 95, 625, 625), "template_roi": (795, 195, 485, 525)}



# 历史规格：这三个色在**自动曝光的旧图**上用的阈值。旧图（自动曝光）与新图（固定曝光）
# 的饱和度分布不同，同一组数无法同时适配；现网按新图定参数，测试用旧图，故在此钉住旧值。
_LEGACY_RANGES = {
    3: [((23, 90, 120), (34, 255, 255))],   # Yellow
    4: [((35, 60, 120), (84, 255, 255))],   # Green
    6: [((130, 35, 60), (167, 255, 255))],  # Purple
}

def _pinned_params():
    import copy as _copy
    from scenarios.tangram_puzzle.config import DETECT_PARAMS as _P
    params = _copy.deepcopy(_P)
    params.update(_TEST_ROIS)
    for _sp in params["specs"]:
        if _sp["id"] in _LEGACY_RANGES:
            _sp["ranges"] = _LEGACY_RANGES[_sp["id"]]
    return params


def piece(vertices, h, shape="triangle", pid=1):
    pixel=_transform(vertices,np.linalg.inv(h))
    center=_transform([np.mean(vertices,axis=0)],np.linalg.inv(h))[0]
    return TangramPiece(pid,"Test",shape,tuple(center),tuple(center),pixel.tolist(),0.0,
                        pixel.astype(np.float32).reshape(-1,1,2))


def fixture(shape="triangle",angle=37,mirror=False):
    # 刻意使用反射/透视映射，避免图像角直接相减也能碰巧通过。
    h=np.array([[-.4,.1,20],[.05,.5,-100],[.0001,.0002,1.]])
    vertices={"triangle":[[0,0],[40,0],[0,40]],
              "square":[[0,0],[20,0],[20,20],[0,20]],
              "parallelogram":[[0,0],[40,0],[60,20],[20,20]]}[shape]
    source=np.asarray(vertices,dtype=float);source-=source.mean(axis=0)
    target=source.copy()
    if mirror:
        target[:,0]*=-1
        target=target[::-1]
    target=target@_rotation(angle).T+np.array([100,100])
    return piece(source,h,shape),piece(target,h,shape),CameraMapper(homography=h)


class FakeRPC:
    def __init__(self, fail=False, tool=0):
        self.fail=fail;self.tool=tool;self.calls=[];self.moves=[]
    def GetActualTCPNum(self):return 0,self.tool
    def GetActualWObjNum(self):return 0,0
    def GetActualJointPosDegree(self):return 0,[0]*6
    def GetInverseKinRef(self,kind,pose,reference):
        self.calls.append((list(pose),list(reference)))
        if self.fail and abs(pose[0]-25.861)<.001 and pose[2]==351.5:
            return 112,None
        return 0,[len(self.calls)]*6
    def MoveJ(self,*args,**kwargs):
        self.moves.append((args,kwargs));return 0


class PlanningTests(unittest.TestCase):
    def test_perspective_robot_rotation(self):
        r,t,m=fixture()
        p=plan_puzzle([r],[t],m,[180,0,0],10,10,0)[0]
        self.assertAlmostEqual(p.delta_angle,37,places=5)
        self.assertLess(p.fit_error_mm,1e-5)
        self.assertTrue(np.allclose(p.place_pose[:2],[100,100]))

    def test_symmetry(self):
        for shape,angle,expected in [("triangle",170,170),("square",100,10),("parallelogram",170,-10)]:
            r,t,m=fixture(shape,angle)
            p=plan_puzzle([r],[t],m,[180,0,0],10,10,0)[0]
            self.assertAlmostEqual(p.delta_angle,expected,places=5)
            self.assertLess(p.fit_error_mm,1e-5)

    def test_mirrored_parallelogram_blocked(self):
        r,t,m=fixture("parallelogram",20,True)
        plans=plan_puzzle([r],[t],m,[180,0,0],10,10,0)
        self.assertTrue(plans[0].suspected_flip)
        with self.assertRaises(ValueError):validate_plan_geometry(plans)

    def test_missing_duplicate_invalid_mapping(self):
        r,t,m=fixture()
        for raw,mapper in [([],m),([r,r],m),([r],CameraMapper())]:
            with self.assertRaises(ValueError):plan_puzzle(raw,[t],mapper,[180,0,0],10,10,0)
        with self.assertRaises(ValueError):normalize_angle(float("inf"))

    def test_offset_applied_once(self):
        r,t,m=fixture()
        p=plan_puzzle([r],[t],m,[180,0,120],10,10,5)[0]
        self.assertAlmostEqual(p.delta_angle,42,places=5)
        self.assertAlmostEqual(p.place_pose[5],162,places=5)

    def test_sample_flip_diagnostic(self):
        data=Path(__file__).parent/"data"/"stage3_20261025"
        mapper=CameraMapper(CAMERA_MATRIX,DIST_COEFFS,HOMOGRAPHY)
        d=TangramDetector(_pinned_params())
        raw=d.detect_raw_material(mapper.undistort(cv.imread(str(data/"1_Color.png"))))
        tpl=d.detect_template(mapper.undistort(cv.imread(str(data/"template_Color_Color.png"))))
        plans=plan_puzzle(raw,tpl,mapper,[180,0,0],10,10,0)
        self.assertEqual(len(plans),7)
        self.assertEqual([p.piece_id for p in plans if p.suspected_flip],[4])

    def test_no_argument_run_does_not_import_hardware(self):
        with self.assertRaises(ValueError):main.run()


class SuctionOffsetTests(unittest.TestCase):
    """吸盘偏心旋转补偿：偏心只在放置旋转后产生平面位移。"""

    def test_default_offset_keeps_old_plan(self):
        r,t,m=fixture()
        base=plan_puzzle([r],[t],m,[180,0,0],10,10,0)[0]
        explicit=plan_puzzle([r],[t],m,[180,0,0],10,10,0,suction_offset=(0.0,0.0))[0]
        self.assertEqual(base.place_pose,explicit.place_pose)
        self.assertEqual(base.grasp_pose,explicit.grasp_pose)
        self.assertEqual(base.rotation_offset_mm,(0.0,0.0))

    def test_no_rotation_no_correction(self):
        # 正方形周期 90°，输入 0° 时 delta=0，偏心不应改变任何位姿。
        r,t,m=fixture("square",0)
        plain=plan_puzzle([r],[t],m,[180,0,0],10,10,0)[0]
        shifted=plan_puzzle([r],[t],m,[180,0,0],10,10,0,suction_offset=(12.0,-7.0))[0]
        self.assertAlmostEqual(shifted.delta_angle,0.0,places=6)
        self.assertEqual(plain.place_pose,shifted.place_pose)
        self.assertEqual(shifted.grasp_pose,plain.grasp_pose)

    def test_correction_matches_rotation_formula(self):
        offset=np.array([9.0,-4.0])
        for shape,angle in (("triangle",37),("triangle",-160),("parallelogram",25)):
            r,t,m=fixture(shape,angle)
            plain=plan_puzzle([r],[t],m,[180,0,0],10,10,0)[0]
            shifted=plan_puzzle([r],[t],m,[180,0,0],10,10,0,suction_offset=offset)[0]
            self.assertEqual(shifted.grasp_pose,plain.grasp_pose)
            expected=rotation_offset(shifted.delta_angle,offset)
            self.assertLess(np.linalg.norm(expected),2*np.linalg.norm(offset)+1e-9)
            np.testing.assert_allclose(shifted.rotation_offset_mm,expected,atol=1e-9)
            np.testing.assert_allclose(np.subtract(plain.place_pose[:2],shifted.place_pose[:2]),
                                       expected,atol=1e-6)
            # 反补后块中心仍应落在模板质心上，与不补偿时的规划落点一致。
            np.testing.assert_allclose(predicted_piece_xy(shifted),plain.place_pose[:2],atol=1e-6)
            self.assertAlmostEqual(shifted.place_pose[5],plain.place_pose[5],places=9)

    def test_invalid_offset_rejected(self):
        r,t,m=fixture()
        for bad in ((0.0,0.0,0.0),(float("inf"),0.0),(500.0,0.0),None):
            if bad is None:
                continue
            with self.assertRaises(ValueError):
                plan_puzzle([r],[t],m,[180,0,0],10,10,0,suction_offset=bad)

    def test_solve_offset_recovers_from_samples(self):
        truth=np.array([7.5,-3.25])
        samples=[]
        for delta in (-174.8,-84.3,13.5,42.9):
            d=rotation_offset(delta,truth)
            samples.append((delta,d[0],d[1]))
        solved,residual=solve_suction_offset(samples)
        np.testing.assert_allclose(solved,truth,atol=1e-9)
        self.assertLess(residual,1e-9)
        with self.assertRaises(ValueError):
            solve_suction_offset(samples[:1])
        with self.assertRaises(ValueError):
            solve_suction_offset([(20.0,1.0,0.0),(20.5,1.0,0.0)])

    def test_sample_offset_plan_still_valid(self):
        # 当前样本七块全出且只有绿块镜像：启用偏心补偿不应改变该结论。
        data=Path(__file__).parent/"data"/"stage3_20261025"
        mapper=CameraMapper(CAMERA_MATRIX,DIST_COEFFS,HOMOGRAPHY)
        d=TangramDetector(_pinned_params())
        raw=d.detect_raw_material(mapper.undistort(cv.imread(str(data/"1_Color.png"))))
        tpl=d.detect_template(mapper.undistort(cv.imread(str(data/"template_Color_Color.png"))))
        plans=plan_puzzle(raw,tpl,mapper,[180,0,0],10,10,0,suction_offset=(10.0,5.0))
        self.assertEqual(len(plans),7)
        self.assertEqual([p.piece_id for p in plans if p.suspected_flip],[4])

    def test_measure_pair_solves_offset(self):
        from scenarios.tangram_puzzle.tool_offset_check import solve_offset_from_pair
        truth=np.array([6.5,-3.25])      # u：基准姿态下测量的偏心向量（mm）
        mark=np.array([100.0,-200.0])    # 尖端对准的标记位置
        for start,turn in ((139.234,180.0),(139.234,90.0),(-20.0,180.0)):
            # 法兰坐标系中的偏心 e 满足 u = R(start)·e；尖端固定时 F = M - R(rz)·e
            e=_rotation(-start)@truth
            fa=mark-(_rotation(start)@e)
            fb=mark-(_rotation(start+turn)@e)
            solved,dz,reported=solve_offset_from_pair(
                [*fa,350.0,-178.633,0.416,start],
                [*fb,350.0,-178.633,0.416,start+turn],reference_rz=start)
            np.testing.assert_allclose(solved,truth,atol=1e-9)
            self.assertAlmostEqual(dz,0.0,places=9)
            self.assertAlmostEqual(reported,turn,places=6)

    def test_measure_pair_rejects_tiny_rotation(self):
        from scenarios.tangram_puzzle.tool_offset_check import solve_offset_from_pair
        with self.assertRaises(ValueError):
            solve_offset_from_pair([0,0,350,0,0,10],[1,1,350,0,0,10.2])

    def test_landing_recovers_offset_from_photo(self):
        # 用同一张模板实拍图当"成品区落点照"：计划若带偏心 u，
        # 复核实测落点与计划的差应等于 (R(Δ)-I)·u，反解应回到 u。
        import json
        import tempfile
        from scenarios.tangram_puzzle.tool_offset_check import measure_landing, report_landing
        truth=np.array([10.0,-5.0])
        data=Path(__file__).parent/"data"/"stage3_20261025"
        mapper=CameraMapper(CAMERA_MATRIX,DIST_COEFFS,HOMOGRAPHY)
        d=TangramDetector(_pinned_params())
        raw=d.detect_raw_material(mapper.undistort(cv.imread(str(data/"1_Color.png"))))
        tpl=d.detect_template(mapper.undistort(cv.imread(str(data/"template_Color_Color.png"))))
        plans=plan_puzzle(raw,tpl,mapper,[180,0,0],10,10,0,suction_offset=truth)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"plans.json"
            path.write_text(json.dumps({"plans":[asdict(p) for p in plans]}),encoding="utf-8")
            rows=measure_landing(data/"template_Color_Color.png",path,echo=lambda *a:None,
                                detect_params=_pinned_params())
            self.assertEqual(len(rows),7)
            for _,delta,error in rows:
                np.testing.assert_allclose(error,rotation_offset(delta,truth),atol=2e-3)
            solved,residual=report_landing(rows,echo=lambda *a:None)
            np.testing.assert_allclose(solved,truth,atol=2e-2)
            self.assertLess(residual,1e-2)


class RobotPreflightTests(unittest.TestCase):
    def setUp(self):
        r,t,m=fixture()
        base=plan_puzzle([r],[t],m,[180,0,0],348.86,351.5,0)[0]
        self.plans=[]
        for i in range(1,8):
            p=copy.deepcopy(base);p.piece_id=i
            p.place_pose[0]=25.861 if i==6 else 100+i
            self.plans.append(p)
        self.robot=CobotClient("not-connected")

    def test_42_endpoints_no_motion_and_chained_references(self):
        rpc=FakeRPC();self.robot.rpc=rpc
        report=self.robot.preflight_pick_place(self.plans,30)
        self.assertEqual(len(report),42)
        self.assertEqual(rpc.moves,[])
        self.assertEqual(report[0]["pose"][2],378.86)
        self.assertEqual(rpc.calls[1][1],[1]*6)

    def test_112_reports_piece_and_pose_without_motion(self):
        rpc=FakeRPC(fail=True);self.robot.rpc=rpc
        with self.assertRaises(IKPreflightError) as cm:
            self.robot.preflight_pick_place(self.plans,30)
        self.assertEqual(cm.exception.details["error_code"],112)
        self.assertIn("ID=6",cm.exception.details["context"])
        self.assertIn("/place",cm.exception.details["context"])
        self.assertEqual(rpc.moves,[])
        self.assertEqual(cm.exception.details["pose"][2],351.5)

    def test_above_ik_and_move_use_same_target(self):
        rpc=FakeRPC();self.robot.rpc=rpc
        pose=self.plans[0].place_pose
        original=list(pose)
        self.robot.move_above(pose,30)
        target=rpc.moves[0][1]["desc_pos"]
        self.assertEqual(rpc.calls[0][0],target)
        self.assertEqual(target[2],381.5)
        self.assertEqual(pose,original)
        self.assertNotIn("offset_flag",rpc.moves[0][1])

    def test_wrong_active_tool_blocks_ik(self):
        rpc=FakeRPC(tool=1);self.robot.rpc=rpc
        with self.assertRaises(RuntimeError):self.robot.preflight_pick_place(self.plans,30)
        self.assertEqual(rpc.calls,[])
        self.assertEqual(rpc.moves,[])

if __name__=="__main__":unittest.main()
