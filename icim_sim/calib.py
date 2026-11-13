# -*- coding: utf-8 -*-
"""仿真工位的"标定值"——全部由模型算出来。

为什么需要这一层
----------------
队友程序里有四组**只对本工位成立**的数：

    CAPTURE_POSE  拍照位（法兰应该停在两块托盘上方多高多远）
    GRIP_Z/PLACE_Z 抓放高度
    HOMOGRAPHY    像素 → 机器人基座 xy
    raw_roi/template_roi  两块托盘在图像里的框

这些是在**他们那台真机**上标出来的。CAD 里的工位和真机不是同一套夹具位置
，所以把这四个数原样搬到仿真里必然错位。

正确做法不是去改模型，而是**像给新工位重新标定一样**，用模型重新算出这四个数：

    拍照位 = 两块托盘中心的正上方，离板面高度与真机一致，姿态沿用真机
    抓放Z  = 板面 + 工具长度                     （见 sim.cell_facts）
    单应   = 仿真相机的真实投影的逆（板面高度上）
    原料框 = **原料托盘**的投影外接框 + 6 mm      （按托盘算，换布局不用重标）
    成品框 = 目标托盘中心、边长 = **托盘垫片**的正方形投影

顺带还有两个"图案通道"要用的量，也由模型算：

    S_PHYS_MM = 七块面积和开根（= 拼成正方形时的边长），2026-11-02 实测 116.364
    usable_mm = 托盘垫片边长（成品区可用范围），260 mm

于是"程序"和"标定"的关系和真机完全一样，只有坐标系里的数字变了。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import mujoco
import numpy as np

from .cell import SCENE, measure
from .paths import CAPTURE_POSE

IMG_W, IMG_H = 1280, 720



@dataclass(frozen=True)
class Board:
    key: str
    centre_base: np.ndarray          # (x, y) mm，基座系
    xy_min: np.ndarray
    xy_max: np.ndarray
    z_top_mm: float = 0.0


#: 抓放姿态：法兰 +z 指向正下方（吸盘轴竖直）。绕竖直轴的转角是自由的 —— 吸盘是圆的，
#: 转多少都吸得住，但**手腕能不能到**取决于它，所以留一个参数让上层去挑。
R_GRASP = np.array([[1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0]])


def grasp_rpy_at(yaw_deg: float = 0.0) -> tuple:
    """吸盘轴竖直、绕竖直轴转 yaw_deg 的抓放姿态（ZYX 欧拉角，度）。"""
    a = np.radians(yaw_deg)
    c, s = np.cos(a), np.sin(a)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return tuple(float(v) for v in _rpy_of(Rz @ R_GRASP))


@dataclass(frozen=True)
class SimCalib:
    capture_pose: list                # [x,y,z,rx,ry,rz] 基座系 mm/deg
    grasp_rpy: tuple                  # 抓放姿态（吸盘朝下），也由模型定
    homography: np.ndarray            # 像素 -> 基座 xy（板面高度）
    raw_roi: tuple                    # (x, y, w, h)
    template_roi: tuple
    raw_board: Board
    target_board: Board
    grip_z: float
    place_z: float
    camera_base_mm: np.ndarray
    s_phys_mm: float = 0.0            # 七块面积和开根 = 等效正方形边长（mm）
    usable_mm: float = 0.0            # 成品区"可用边长"：垫片边长（mm）
    notes: list = field(default_factory=list)

    def report(self) -> str:
        r = [
            f"拍照位（基座系）  {np.round(self.capture_pose, 3).tolist()}",
            f"抓放姿态 rpy      {np.round(self.grasp_rpy, 3).tolist()}",
            f"相机位置（基座系）{np.round(self.camera_base_mm, 1).tolist()} mm",
            f"原料托盘中心      {np.round(self.raw_board.centre_base, 1).tolist()} mm",
            f"目标托盘中心      {np.round(self.target_board.centre_base, 1).tolist()} mm",
            f"raw_roi           {self.raw_roi}",
            f"template_roi      {self.template_roi}",
            f"抓取/放置法兰 z   {self.grip_z:.3f} / {self.place_z:.3f} mm",
            f"实物尺度 S        {self.s_phys_mm:.3f} mm（七块面积和开根）",
            f"成品区可用边长    {self.usable_mm:.1f} mm（= 托盘垫片）",
            "HOMOGRAPHY =",
        ]
        r += [f"  [{v: .9e}, {w: .9e}, {u: .9e}]," for v, w, u in self.homography]
        return "\n".join(r)


# --------------------------------------------------------------------- helpers
@dataclass(frozen=True)
class CollisionPad:
    """托盘下面的**不可见碰撞垫片**（statics 里 contype=1 的 box，rgba 全 0）。

    为什么它是"成品区可用范围"的真值：可见托盘的凹槽比它小，而且七块板实际是
    落在这个垫片的上表面上的。程序里"图案放不放得下"应该按它算 —— 按可见托盘
    外框算会偏小，按"当前那一摞板子的外接框"算则换布局就失效。
    """

    centre_base: np.ndarray          # (x, y) mm，基座系
    half_mm: np.ndarray              # (hx, hy) mm
    z_top_mm: float

    def edge_mm(self) -> float:
        return float(2.0 * float(np.min(self.half_mm)))


def find_pads(m, d, Rb, pb):
    """所有"不可见碰撞垫片"（statics 下的 box，且参与碰撞）。"""
    statics = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "statics")
    out = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != statics:
            continue
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX or m.geom_contype[g] == 0:
            continue
        s = np.asarray(m.geom_size[g], float)
        centre = _to_base(Rb, pb, np.asarray(d.geom_xpos[g], float) * 1000.0)
        R = d.geom_xmat[g].reshape(3, 3)
        top = float((R @ np.array([0.0, 0.0, s[2]]) + d.geom_xpos[g])[2] * 1000.0)
        out.append(CollisionPad(centre_base=centre[:2], half_mm=s[:2] * 1000.0,
                                z_top_mm=top))
    return out


def _piece_area_sum(m, d, Rb, pb):
    """七块板在基座系 xy 平面上的面积和（mm²），用网格顶点的凸包算。"""
    total = 0.0
    for i in range(m.nbody):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if not nm.startswith("piece_"):
            continue
        pts = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != i or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = m.geom_dataid[g]
            adr, num = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            v = (m.mesh_vert[adr:adr + num] @ d.geom_xmat[g].reshape(3, 3).T
                 + d.geom_xpos[g])
            pts.append((Rb.T @ (v - pb).T).T[:, :2] * 1000.0)
        if not pts:
            continue
        hull = cv2.convexHull(np.vstack(pts).astype(np.float32))
        total += abs(float(cv2.contourArea(hull)))
    return total


def _base_frame(m, d):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    return d.xmat[b].reshape(3, 3).copy(), d.xpos[b].copy()


def _to_base(Rb, pb, p_world_mm):
    return Rb.T @ (np.asarray(p_world_mm, float) / 1000.0 - pb) * 1000.0


def _to_world(Rb, pb, p_base_mm):
    return (Rb @ (np.asarray(p_base_mm, float) / 1000.0) + pb) * 1000.0


def _geom_aabb_world(m, d, g):
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        adr, num = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        v = m.mesh_vert[adr:adr + num]
    elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
        s = m.geom_size[g]
        v = np.array([[sx, sy, sz] for sx in (-s[0], s[0])
                      for sy in (-s[1], s[1]) for sz in (-s[2], s[2])])
    else:
        return None
    w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
    return w.min(axis=0), w.max(axis=0)


def find_boards(m, d, Rb, pb):
    """所有够大、够薄、水平放置的静止件（候选托盘）。"""
    statics = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "statics")
    found = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != statics:
            continue
        box = _geom_aabb_world(m, d, g)
        if box is None:
            continue
        lo, hi = box
        size = (hi - lo) * 1000
        if min(size[:2]) < 150 or max(size[:2]) > 400 or size[2] > 15:
            continue
        found.append(Board(key=f"geom{g}",
                           centre_base=_to_base(Rb, pb, ((lo + hi) / 2 * 1000))[:2],
                           xy_min=_to_base(Rb, pb, lo * 1000)[:2],
                           xy_max=_to_base(Rb, pb, hi * 1000)[:2],
                           z_top_mm=float(hi[2] * 1000)))
    return found


def body_aabb_base(m, d, name, Rb=None, pb=None):
    """某个 body 在基座系里的 AABB 中心（mm）。"""
    if Rb is None or pb is None:
        Rb, pb = _base_frame(m, d)
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != b:
            continue
        box = _geom_aabb_world(m, d, g)
        if box is None:
            continue
        lo = np.minimum(lo, box[0])
        hi = np.maximum(hi, box[1])
    return _to_base(Rb, pb, ((lo + hi) / 2 * 1000))


def _tiles_aabb(m, d, Rb, pb):
    """七块七巧板合起来的基座系 AABB（xy），用网格顶点算，不用 body 原点。"""
    lo = np.full(2, np.inf)
    hi = np.full(2, -np.inf)
    for i in range(m.nbody):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if not nm.startswith("piece_"):
            continue
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != i:
                continue
            box = _geom_aabb_world(m, d, g)
            if box is None:
                continue
            a = _to_base(Rb, pb, box[0] * 1000)[:2]
            b = _to_base(Rb, pb, box[1] * 1000)[:2]
            lo = np.minimum(lo, np.minimum(a, b))
            hi = np.maximum(hi, np.maximum(a, b))
    return lo, hi


def pick_boards(candidates, tiles_centre):
    """原料托盘 = 装着七巧板的那块；目标托盘 = 同一高度上离它最近的那块。"""
    def inside(b, p):
        lo = np.minimum(b.xy_min, b.xy_max)
        hi = np.maximum(b.xy_min, b.xy_max)
        return bool(np.all(p >= lo - 1.0) and np.all(p <= hi + 1.0))

    raw = next((b for b in candidates if inside(b, tiles_centre)), None)
    if raw is None:
        raw = min(candidates, key=lambda b: np.linalg.norm(b.centre_base - tiles_centre))
    peers = [b for b in candidates
             if b is not raw and abs(b.z_top_mm - raw.z_top_mm) <= 2.0]
    if not peers:
        raise RuntimeError("只找到一块托盘，模型里应该有两块")
    target = min(peers, key=lambda b: np.linalg.norm(b.centre_base - raw.centre_base))
    return raw, target


def project_points(cam_pos, cam_mat, fovy, pts_world_mm, width=IMG_W, height=IMG_H):
    f = (height / 2.0) / np.tan(np.radians(fovy) / 2.0)
    out = []
    for P in np.asarray(pts_world_mm, float) / 1000.0:
        c = cam_mat.T @ (P - cam_pos)
        out.append([width / 2.0 + f * c[0] / -c[2],
                    height / 2.0 - f * c[1] / -c[2]])
    return np.array(out)


def load_capture_pose(path=None):
    """读固定的拍照位（法兰位姿，mm/度）。没有文件或格式不对就返回 None。

    为什么要有这个：拍照位只取决于"相机怎么装在法兰上"和"两块托盘在哪"。
    这两样都没动的话，位置就是固定的，不必每次都现场搜索——搜索还依赖逆解，
    换臂型时容易受影响。
    """
    p = Path(path or CAPTURE_POSE)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        pose = data.get("pose") if isinstance(data, dict) else data
        if isinstance(pose, (list, tuple)) and len(pose) == 6:
            return [float(v) for v in pose]
    except Exception as exc:                          # noqa: BLE001
        print(f"[calib] {p.name} 读不动，忽略：{exc}")
    print(f"[calib] {p.name} 里不是 6 个数，忽略")
    return None


def load_capture_joints(path=None):
    """读拍照位对应的六个关节角（度）。没有就返回 None，由调用方自己逆解。"""
    p = Path(path or CAPTURE_POSE)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        joints = data.get("joint_deg") if isinstance(data, dict) else None
        if isinstance(joints, (list, tuple)) and len(joints) == 6:
            return [float(v) for v in joints]
    except Exception:                                 # noqa: BLE001
        return None
    return None


def _plane_homography(cam_pos, cam_mat, fovy, z_base, side=200.0):
    """板面高度上的 基座xy → 像素 单应，再取逆得 像素 → 基座xy。

    全部在基座系里算：基座系和世界系之间只差一个刚体变换，投影不受影响。
    """
    quad = np.array([[-side, -side], [side, -side], [side, side], [-side, side]])
    world = np.column_stack([quad, np.full(4, z_base)])
    px = project_points(cam_pos, cam_mat, fovy, world)
    Hpix = cv2.getPerspectiveTransform(quad.astype(np.float32), px.astype(np.float32))
    return np.linalg.inv(Hpix)


def _rpy_of(R):
    """旋转矩阵 → Fairino 口径的 ZYX 欧拉角（度）。"""
    b = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    a = np.arctan2(R[2, 1], R[2, 2])
    c = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees([a, b, c])


def _roi_from_box(px):
    x0, y0 = np.floor(px.min(axis=0)).astype(int)
    x1, y1 = np.ceil(px.max(axis=0)).astype(int)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(IMG_W, x1), min(IMG_H, y1)
    return (int(x0), int(y0), int(max(4, x1 - x0)), int(max(4, y1 - y0)))


# ------------------------------------------------------------------- the calib
@lru_cache(maxsize=2)
def measure_calib(scene: str | Path = SCENE) -> SimCalib:
    facts = measure(scene)
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home"))
    mujoco.mj_forward(m, d)
    Rb, pb = _base_frame(m, d)

    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Link_6")
    Rc = d.cam_xmat[cid].reshape(3, 3).copy()
    pc = d.cam_xpos[cid].copy()
    p_f = d.xpos[fid].copy()
    fovy = float(m.cam_fovy[cid])

    # 相机相对法兰的固定变换（刚体，随便在哪个姿态量都一样，这里用 home 位）
    R_f = d.xmat[fid].reshape(3, 3).copy()
    C_R = R_f.T @ Rc
    C_t = R_f.T @ (pc - p_f)

    # 两块托盘：原料（有板子的那块）与目标
    tiles_c = np.mean([facts.tiles[n][:2] for n in facts.tiles], axis=0)
    raw_board, target_board = pick_boards(find_boards(m, d, Rb, pb), tiles_c)
    tiles_aabb = _tiles_aabb(m, d, Rb, pb)
    raw_centre = (tiles_aabb[0] + tiles_aabb[1]) / 2.0
    half = (tiles_aabb[1] - tiles_aabb[0]) / 2.0

    # 实物尺度与成品区可用范围，都按模型算：
    #   S = 七块面积和开根（一套七巧板拼成正方形时正好是 S²）
    #   可用边长 = 托盘垫片（不可见碰撞板）的边长
    s_phys = float(np.sqrt(_piece_area_sum(m, d, Rb, pb)))
    _pads = find_pads(m, d, Rb, pb)
    if not _pads:
        raise RuntimeError("模型里没找到托盘的碰撞垫片（statics 下 contype=1 的 box）")
    pads = min(_pads, key=lambda p: float(np.linalg.norm(
        p.centre_base - target_board.centre_base)))
    usable = pads.edge_mm()

    # ---- 两个姿态，目的不同，别混 -------------------------------------------
    # 拍照：**相机光轴**必须竖直向下，画面才是俯视。法兰姿态 = Rz(ψ)·C_R^T，
    #       于是相机帧 = Rz(ψ)；ψ 是绕竖直轴的自由度（J6），下面挑一个能到达的。
    # 抓放：**吸盘轴**必须竖直向下（吸盘歪着吸不住）。吸盘轴 = 法兰 +z，
    #       所以法兰 z 要指向正下方 —— 和相机那 2.49° 的安装倾角无关。
    #       差这 2.49° 看着小，252 mm 的杆长会把它放大成 11 mm 的横向偏差。
    from .robot import SimCobotClient, rpy_zyx

    grasp_rpy = grasp_rpy_at(0.0)

    aim = (raw_centre + target_board.centre_base) / 2.0
    off0 = (C_R.T @ C_t) * 1000.0        # 相机相对法兰，在"相机帧=基座帧"下的位移

    def make_look(psi_deg):
        """给定绕竖直轴的转角 ψ，返回 (R_lab, off_base, flange_xy)。"""
        a = np.radians(psi_deg)
        c, s = np.cos(a), np.sin(a)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        R = Rz @ C_R.T
        off = Rz @ off0
        return R, off, aim - off[:2]

    def box_px(cam_pos, cam_mat, centre_xy, half_xy):
        pts = np.array([[x, y] for x in (centre_xy[0] - half_xy[0], centre_xy[0] + half_xy[0])
                        for y in (centre_xy[1] - half_xy[1], centre_xy[1] + half_xy[1])])
        plane = np.column_stack([pts, np.full(len(pts), facts.tile_top_base_mm)])
        return project_points(cam_pos, cam_mat, fovy, plane)

    def solve_capture(psi_deg):
        """在给定 ψ 下，找出"两堆板子都进画面"的最低拍照高度，返回全部中间量。"""
        R_lab, off_base, flange_xy = make_look(psi_deg)

        def look_at(flange_z):
            capture = [float(flange_xy[0]), float(flange_xy[1]), float(flange_z),
                       *_rpy_of(R_lab)]
            T_f = np.eye(4)
            T_f[:3, :3] = R_lab
            T_f[:3, 3] = np.array([flange_xy[0], flange_xy[1], flange_z]) / 1000.0
            T_c = np.eye(4)
            T_c[:3, :3] = C_R
            T_c[:3, 3] = C_t
            T_cam = T_f @ T_c
            return capture, T_cam[:3, 3].copy(), T_cam[:3, :3].copy()

        margin = 14.0
        for z in np.arange(facts.tile_top_base_mm + 250.0,
                           facts.tile_top_base_mm + 900.0, 5.0):
            capture, cam_pos, cam_mat = look_at(float(z))
            boxes = [box_px(cam_pos, cam_mat, raw_centre, half),
                     box_px(cam_pos, cam_mat, target_board.centre_base, half)]
            need = 0.0
            for box in boxes:
                need = max(need,
                           margin - box[:, 0].min(), margin - box[:, 1].min(),
                           box[:, 0].max() - (IMG_W - margin),
                           box[:, 1].max() - (IMG_H - margin))
            if need <= 0.0:
                return dict(psi=psi_deg, capture=capture, cam_pos=cam_pos,
                            cam_mat=cam_mat, off_base=off_base, flange_z=float(z))
        return None

    # ψ 的挑法：优先"图像轴和基座轴对齐"（画面看着正常），并且**真的能逆解**。
    # 都不行才退到"法兰离基座最近"那个角度——相机有 94 mm 偏心，转对了能少伸 94 mm。
    psi_pref = np.arctan2(aim[1], aim[0]) - np.arctan2(off0[1], off0[0])
    candidates = [0.0, 90.0, -90.0, 180.0, float(np.degrees(psi_pref))]
    robot = SimCobotClient(quiet=True)
    robot.connect()

    def from_pose(pose):
        """用给定的法兰位姿反推拍照位相关量（相机变换是刚体，直接算）。"""
        pose = [float(v) for v in pose]
        R_lab = rpy_zyx(*pose[3:])
        Rz = R_lab @ C_R
        T_f = np.eye(4)
        T_f[:3, :3] = R_lab
        T_f[:3, 3] = np.array(pose[:3], float) / 1000.0
        T_c = np.eye(4)
        T_c[:3, :3] = C_R
        T_c[:3, 3] = C_t
        T_cam = T_f @ T_c
        return dict(psi=float(np.degrees(np.arctan2(Rz[1, 0], Rz[0, 0]))),
                    capture=pose, cam_pos=T_cam[:3, 3].copy(),
                    cam_mat=T_cam[:3, :3].copy(),
                    off_base=R_lab @ C_t, flange_z=pose[2])

    chosen = None
    # 0) 优先用 capture_pose.json 里固定的拍照位：桌子、相机、托盘都没动的话，
    #    这个位置就是对的，没必要每次现场搜索（搜索还依赖逆解，容易受臂型影响）。
    pinned = load_capture_pose()
    if pinned is not None:
        sol = from_pose(pinned)
        try:
            sol["joints"] = robot._pose_to_joint(sol["capture"], None, "拍照位")
            chosen = sol
            print(f"[calib] 用 capture_pose.json 里的拍照位"
                  f"（{Path(CAPTURE_POSE).name}）")
        except Exception as exc:                          # noqa: BLE001
            print(f"[calib] capture_pose.json 里的拍照位逆解不了（{exc}），"
                  f"改为现场搜索")

    for psi_deg in candidates:
        if chosen is not None:
            break
        sol = solve_capture(psi_deg)
        if sol is None:
            continue
        try:
            q = robot._pose_to_joint(sol["capture"], None, "拍照位")
        except Exception:                                # noqa: BLE001
            continue
        sol["joints"] = q
        chosen = sol
        break
    robot.disconnect()
    if chosen is None:
        raise RuntimeError("拍照位逆解不出来（capture_pose.json 和自动搜索都不行），"
                           "检查相机安装/托盘位置")

    capture = chosen["capture"]
    cam_pos, cam_mat = chosen["cam_pos"], chosen["cam_mat"]
    off_base, flange_z, psi_deg = chosen["off_base"], chosen["flange_z"], chosen["psi"]
    dz_cam = float(off_base[2])          # 负值：相机比法兰低这么多

    H = _plane_homography(cam_pos, cam_mat, fovy, facts.tile_top_base_mm)
    # 原料框：整块原料托盘 + 6 mm。板子被 scatter 限在托盘内（还留 12 mm 边距），
    # 所以任何合法布局都在框里。**不能**按"当前这一摞板子的外接框"算 ——
    # 那样换一个布局就得重新标定，"任意调整初始位姿"这条硬要求就废了。
    raw_half = np.abs(raw_board.xy_max - raw_board.xy_min) / 2.0 + 6.0
    raw_roi = _roi_from_box(box_px(cam_pos, cam_mat, raw_board.centre_base, raw_half))
    # 成品区框：以目标托盘中心为中心、边长 = 垫片边长的**正方形**。
    # image 通道里"图案顶点是否在成品区内"那条硬门禁就是用它（`tray_quad_mm`）。
    # 画面只差 2.2° 倾斜，所以像素外接框反算回来只大 4% 左右，可以接受。
    template_roi = _roi_from_box(box_px(cam_pos, cam_mat, target_board.centre_base,
                                        np.full(2, usable / 2.0)))

    return SimCalib(
        capture_pose=capture,
        grasp_rpy=tuple(float(v) for v in grasp_rpy),
        homography=H,
        raw_roi=raw_roi,
        template_roi=template_roi,
        raw_board=raw_board,
        target_board=target_board,
        grip_z=facts.grip_z_mm,
        place_z=facts.place_z_mm,
        camera_base_mm=tuple(float(v) for v in cam_pos * 1000),
        s_phys_mm=s_phys,
        usable_mm=usable,
        notes=[
            f"拍照姿态：图像绕竖直轴转了 {psi_deg:+.1f}°（选第一个能逆解的候选）",
            f"相机比法兰低 {-dz_cam:.2f} mm，水平偏 {np.round(off_base[:2], 1).tolist()} mm",
            f"拍照高度由'两堆板子都进画面'定出来：法兰 z = {flange_z:.2f} mm"
            f"（板面上方 {flange_z - facts.tile_top_base_mm:.1f} mm）",
            f"抓放姿态 rpy {np.round(grasp_rpy, 2).tolist()}（法兰 +z 竖直向下）",
            f"实物尺度 S = {s_phys:.3f} mm；成品区可用 {usable:.1f}×{usable:.1f} mm",
            f"拍照位逆解 q = {np.round(chosen['joints'], 1).tolist()}",
        ],
    )


if __name__ == "__main__":
    print(measure_calib().report())
