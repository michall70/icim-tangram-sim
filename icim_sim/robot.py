# -*- coding: utf-8 -*-
"""MuJoCo 里的 CobotClient 替身。

对上层代码零改动：方法名、参数、返回值、异常语义都与
`common.robot_client.CobotClient` 一致，只把"控制器"换成仿真。

    from icim_sim.robot import SimCobotClient
    robot = SimCobotClient()
    robot.connect()
    robot.move_home(HOME_JOINT)
    robot.move_to([x, y, 348.86, -178.633, 0.416, 139.234])

位姿 = [x, y, z, rx, ry, rz]，mm/度，原点是**法兰面中心**，R = Rz·Ry·Rx。
关节角与控制器读数一一对应（零位偏移已由 tools/calibrate_joints.py 标定）。
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.optimize import least_squares

from .paths import SCENE, START_POSE

def load_start_pose(path=START_POSE):
    """读起始关节角（度）。文件不在、或格式不对就返回 None（用场景 keyframe）。"""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        joints = data.get("joint_deg") if isinstance(data, dict) else data
        if isinstance(joints, (list, tuple)) and len(joints) == 6:
            return [float(v) for v in joints]
    except Exception as exc:                      # noqa: BLE001
        print(f"[sim] {p.name} 读不动，忽略：{exc}")
    print(f"[sim] {p.name} 里没有 6 个关节角，忽略")
    return None


def save_start_pose(joint_deg, path=START_POSE):
    """把六个关节角写进 start_pose.json，下次启动就用它。"""
    p = Path(path)
    data = {"joint_deg": [round(float(v), 3) for v in joint_deg],
            "note": "仿真启动时的六个关节角（度，控制器口径）。直接改这个文件即可，不用重建场景。"}
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return p


class IKPreflightError(RuntimeError):
    """和真机 SDK 同名的异常；.details 里带出失败位姿，便于排查。"""

    def __init__(self, code, pose, reference, context):
        self.details = {"action": "GetInverseKinRef", "error_code": code,
                        "pose": list(pose), "reference_joints": list(reference),
                        "context": context}
        super().__init__(f"{context}: 逆解失败, 错误码={code}")


# --------------------------------------------------------------------- helpers
def rpy_zyx(rx, ry, rz):
    """Fairino 口径：R = Rz(rz)·Ry(ry)·Rx(rx)，角度为度。"""
    a, b, c = np.radians([rx, ry, rz])
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rotvec(Ra, Rb):
    dR = Ra.T @ Rb
    ang = float(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    if ang < 1e-12:
        return np.zeros(3)
    v = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
    return v / (2 * math.sin(ang)) * np.degrees(ang)


def _six(values, label):
    if not isinstance(values, (list, tuple, np.ndarray)) or len(values) != 6:
        raise ValueError(f"{label}必须有六个数")
    out = [float(v) for v in values]
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{label}含非有限数值")
    return out


# ------------------------------------------------------------- controller shim
class _Rpc:
    """只实现上层真正用到的那几个 RPC。"""

    def __init__(self, robot):
        self._r = robot

    def SetDO(self, idx, value):
        idx, value = int(idx), int(value)
        self._r._do[idx] = value
        if idx == self._r.gripper_do:
            if value:
                self._r._suction_on()
            else:
                self._r._suction_off()
        return 0

    def GetDO(self):
        lo = 0
        for k, v in self._r._do.items():
            if v:
                lo |= 1 << int(k)
        return 0, (0, lo)

    def GetActualTCPNum(self):
        return 0, self._r.tool

    def GetActualWObjNum(self):
        return 0, self._r.user

    def GetActualJointPosDegree(self):
        return 0, self._r._current_joints()

    def CloseRPC(self):
        return 0


# ------------------------------------------------------------------ the client
class SimCobotClient:
    def __init__(self, ip="sim", tool=0, user=0, sdk_version="auto",
                 scene=None, viewer=False, gripper_do=0, quiet=False, layout=None,
                 flips=None):
        self.ip = ip
        self.tool = tool
        self.user = user
        self.sdk_version = sdk_version
        self.scene = str(scene or SCENE)
        self.show_viewer = viewer
        self.quiet = bool(quiet)
        self.gripper_do = gripper_do
        #: 初始布局：None = 不套用；"auto" = 按 layouts/ 的规则挑一份；也可给名字/路径
        self.layout = layout
        #: 开局就翻面的板（平行四边形手性不符时由 icim_sim.chirality 决定）
        self.flips = tuple(flips or ())
        self.rpc = _Rpc(self)
        self.m = None
        self.d = None
        self._viewer = None
        self._do = {}
        self._eq = {}
        self._gripped = None
        self._rel = None
        # 波纹管压缩的基准面（高度 mm）：吸住时锁定，放开时清掉
        self._press_ref_z = None
        # motion shaping: controller vel is % of full speed
        self.speed_deg_s = 180.0      # full-speed joint rate
        self.time_scale = 0.25        # sim runs 4x faster than the real cell
        #: 一个真实动作至少走多久（秒）。关节几乎不变、但末端要平移几十毫米的动作
        #: （松手后抬 30 mm）本来只要十几毫秒，回放里看着像瞬移；给个下限纯为观感。
        self.min_move_seconds = 0.12
        #: 当前播放倍速（1 = 真机速度）。看窗口时用键盘 1-9 实时调。
        self.speed_now = 4.0
        self._last_sync = 0.0
        self._last_key = 0.0
        self.tol_mm = 0.5
        self.tol_deg = 0.05
        # 吸盘判据：吸盘口要在板子轮廓内、离板面不太远、工具轴与法向夹角不超限
        self.suction_max_gap_mm = 8.0        # 吸盘口最多在板面上方这么多（正数=有缝）
        # 波纹管行程（mm）：由场景里的 tube_joint 范围决定，没这个关节就是 0。
        # 压在板面上时，多出来的下压量被这段行程吃掉，所以"压进 48 mm"是合法的。
        self.suction_stroke_mm = 0.0
        self.suction_max_press_mm = 4.0       # 没有伸缩段时的压入容差
        self._slide = -1                     # tube_joint 的 qpos 下标，-1 = 没有
        self._cup = -1                       # 杯体 body id
        self.suction_max_tilt_deg = 20.0
        #: 每走一步物理就回调一次 `on_step(self)`，用来录回放。
        self.on_step = None

    # ---------------------------------------------------------------- lifecycle
    def connect(self):
        self.m = mujoco.MjModel.from_xml_path(self.scene)
        self.d = mujoco.MjData(self.m)
        self._flange = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "Link_6")
        self._base = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self._tcp = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self._cam = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
        for i in range(self.m.neq):
            nm = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_EQUALITY, i) or ""
            if nm.startswith("grip_"):
                self._eq[nm[5:]] = i
        key = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key >= 0:
            mujoco.mj_resetDataKeyframe(self.m, self.d, key)
        self._slide = -1
        j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "tube_joint")
        if j >= 0 and self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
            self._slide = int(self.m.jnt_qposadr[j])
            self.suction_stroke_mm = float(self.m.jnt_range[j, 1] - self.m.jnt_range[j, 0]) * 1000.0
        # 板子顶面的世界高度上限：波纹管在离它 30 mm 以外不用算（省两次 forward）
        mujoco.mj_forward(self.m, self.d)
        tops = [float(self.d.xpos[i][2]) + 0.01 for i in range(self.m.nbody)
                if (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith("piece_")]
        self._board_top_z = (max(tops) * 1000.0) if tops else 0.0
        self.reset_pose()          # 包根目录的 start_pose.json 优先（队友可以自己改）
        mujoco.mj_forward(self.m, self.d)
        if self.layout is not None:
            # 布局文件里如果带了关节角，会盖掉上面的起始姿态（它更具体）
            from . import layout as _layout
            choice = None if self.layout == "auto" else self.layout
            path = _layout.choose(choice)
            if path is not None:
                data = _layout.read(path)
                _layout.apply(self.m, self.d, data)
                if not self.quiet:
                    print(f"[sim] 初始布局 {path.name}："
                          f"{len(data.get('pieces') or {})} 块板"
                          f"{'，含关节角' if data.get('robot_joint_deg') else ''}")
        if self.flips:
            # 手性不符的板在这里翻面（翻完仍然平放在托盘上，见 cell.flip_piece）
            from . import cell as _cell
            _cell.flip_pieces(self.m, self.d, self.flips)
            if not self.quiet:
                print(f"[sim] 已翻面：{', '.join(self.flips)}")
        if self.show_viewer:
            from mujoco import viewer as _viewer
            self._viewer = _viewer.launch_passive(self.m, self.d)
            self._last_sync = time.perf_counter()
            self._last_key = time.perf_counter()
            print("[sim] 窗口已开：点一下控制台，按 1–9 实时调倍速"
                  "（1 = 慢，9 = 快），+ / - 微调")
        if not self.quiet:
            print(f"[sim] connected: {len(self._eq)} grasp constraints, "
                  f"scene {Path(self.scene).name}")

    # ------------------------------------------------------------- start pose
    def reset_pose(self, joint_deg=None):
        """把六个关节放回起始姿态。

        不给 joint_deg 时读包根目录的 start_pose.json（队友可以自己改）；
        文件不存在就保持场景 keyframe 的姿势。返回是否真的改了。
        """
        if joint_deg is None:
            joint_deg = load_start_pose()
            if joint_deg is None:
                return False
        if self.d is None:
            raise RuntimeError("先 connect()")
        joint_deg = _six(joint_deg, "起始关节")
        self.d.qpos[:6] = np.radians(joint_deg)
        self.d.qvel[:6] = 0.0
        mujoco.mj_forward(self.m, self.d)
        if self._viewer is not None:
            self._viewer.sync()
        return True

    def current_start_pose(self):
        """当前六个关节角（度），可以直接喂给 save_start_pose。"""
        return [round(float(v), 3) for v in np.degrees(self.d.qpos[:6])]

    def save_start_pose(self, joint_deg=None):
        """把当前（或给定）姿态存成新的起始姿态。"""
        joints = self.current_start_pose() if joint_deg is None else _six(joint_deg, "起始关节")
        p = save_start_pose(joints)
        print(f"[sim] 起始姿态已保存到 {p}：{[round(v, 2) for v in joints]}")
        return joints

    def disconnect(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        # 保留 m/d：上层脚本（以及我们的报告）在 disconnect 之后还要读最终状态，
        # 这和真机"断开后读不到"不同，但仿真里没有理由丢掉数据。        self.closed = True

    def check_coordinate_context(self):
        if self.tool != 0 or self.user != 0:
            raise ValueError("仿真只支持工具0/工件0")

    # ------------------------------------------------------------------- state
    def _current_joints(self):
        return [float(v) for v in np.degrees(self.d.qpos[:6])]

    def _flange_pose(self):
        """法兰位姿，相对机器人基座（控制器口径）：(mm, R)。"""
        Rb = self.d.xmat[self._base].reshape(3, 3)
        pb = self.d.xpos[self._base]
        p = Rb.T @ (self.d.xpos[self._flange] - pb) * 1000.0
        R = Rb.T @ self.d.xmat[self._flange].reshape(3, 3)
        return p, R

    def current_pose(self):
        p, R = self._flange_pose()
        b, c = math.asin(np.clip(-R[2, 0], -1, 1)), None
        # ZYX inverse: R = Rz(c)Ry(b)Rx(a)
        a = math.atan2(R[2, 1], R[2, 2])
        c = math.atan2(R[1, 0], R[0, 0])
        return list(p) + list(np.degrees([a, b, c]))

    # ---------------------------------------------------------------------- IK
    def _pose_to_joint(self, pose, reference=None, context="运动逆解"):
        """逆解一个法兰位姿，**不动机械臂**。

        注意：这里只解算，不产生运动。调用前后机械臂必须停在原地——所以入口先记下
        当前关节，出口一律恢复它。以前这里恢复的是 `reference`（参考关节），而预检
        （preflight）是拿"上一个解"当下一个端点的参考的，于是预检跑完机械臂会被直接
        设成最后一个解，回放里就看见"瞬移"。
        """
        pose = _six(pose, "目标位姿")
        self.check_coordinate_context()
        q_before = self.d.qpos[:6].copy()
        v_before = self.d.qvel[:6].copy()
        ref = _six(self._current_joints() if reference is None else reference, "参考关节")
        tp = np.array(pose[:3]) / 1000.0
        tR = rpy_zyx(*pose[3:])
        lo, hi = self.m.jnt_range[:6, 0].copy(), self.m.jnt_range[:6, 1].copy()

        def resid(q):
            self.d.qpos[:6] = q
            mujoco.mj_forward(self.m, self.d)
            p, R = self._flange_pose()
            return np.concatenate([(p / 1000.0 - tp) * 1000.0, rotvec(R, tR) * 20.0])

        try:
            def solve_from(seed_deg):
                q0 = np.clip(np.radians(seed_deg), lo + 1e-4, hi - 1e-4)
                sol = least_squares(resid, q0, bounds=(lo, hi),
                                    xtol=1e-12, ftol=1e-12, max_nfev=600)
                ep = float(np.linalg.norm(sol.fun[:3]))
                ea = float(np.linalg.norm(sol.fun[3:]) / 20.0)
                return sol.x, ep, ea

            def travel(q_rad):
                """这个解离参考姿态有多远（按最短方向算，避免 ±360 的假距离）。"""
                dq = np.degrees(q_rad) - np.asarray(ref)
                dq = ((dq + 180.0) % 360.0) - 180.0
                return float(np.abs(dq).sum())

            # 先用参考关节解（保证动作连续）；解不出来再换几个种子重试。
            # 为什么需要重试：全零位是奇异位形（臂伸直），从那儿去别处经常解不出来。
            # 注意：**所有成功解里取离参考最近的那个**，不是第一个成功的——
            # 否则会换到另一个 IK 分支，J6 在两次动作之间甩一整圈（看着像"一直转"）。
            seeds = [ref,
                     [0.0, -90.0, 0.0, -90.0, 0.0, 0.0],
                     [-90.0, -90.0, 0.0, -90.0, 0.0, 0.0],
                     [0.0, -135.0, 0.0, -135.0, 0.0, 0.0],
                     [90.0, -90.0, 0.0, -90.0, 0.0, 0.0]]
            ok, best, err = [], None, None
            for seed in seeds:
                q, ep, ea = solve_from(seed)
                if ep <= self.tol_mm and ea <= self.tol_deg:
                    ok.append(q)
                elif err is None:
                    best, err = q, (ep, ea)
            if ok:
                best = min(ok, key=travel)
            else:
                ep, ea = err
                if ep > self.tol_mm or ea > self.tol_deg:
                    raise IKPreflightError(112 if ep > 20 else 1, pose, ref, context)
            return [float(v) for v in np.degrees(best)]
        finally:
            self.d.qpos[:6] = q_before          # 解算不留痕迹：机械臂回到调用前
            self.d.qvel[:6] = v_before
            mujoco.mj_forward(self.m, self.d)

    # ---------------------------------------------------------- live speed
    def _poll_speed_keys(self):
        """窗口开着时，从控制台读按键实时调倍速（Windows）。

        1–9 = 直接设成 1×..9×（数字越大越快），`+`/`-` = 在当前基础上加减 1×。
        只在本机有 msvcrt 时启用；控制台没有焦点就读不到，属于正常。
        """
        try:
            import msvcrt
        except ImportError:
            return
        while msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in "123456789":
                self.set_speed(float(ch))
            elif ch in "+=":
                self.set_speed(self.speed_now + 1.0)
            elif ch in "-_":
                self.set_speed(self.speed_now - 1.0)

    def set_speed(self, speed):
        """设置播放倍速：1 = 真机速度，数字越大越快。"""
        self.speed_now = float(np.clip(speed, 0.25, 32.0))
        self.time_scale = 1.0 / self.speed_now
        print(f"[sim] 倍速 -> {self.speed_now:g}×")

    # ------------------------------------------------------------------ motion
    def _settle(self, n=1):
        """跑 n 个物理步，**臂保持不动**（运动学驱动），板子按吸附关系跟随。

        2026-11-02 修：原来只 `mj_step`，不重写臂的 qpos。可臂上没有任何执行器
        （只有 gravcomp），所以**接触反力会把臂"踢"动**——下放到贴着板子那一刻，
        吸盘压到板子上，冲量传回手腕，J6（最轻）一帧能转 13°（实测 677°/s，
        正常 36°/s 的 19 倍），这就是"下放时 J6 突然转一下"。臂既然是运动学驱动，
        就该每步都钉住：进入 `_settle` 时记下臂的 qpos/qvel，每步重新写回去。
        """
        q_arm = self.d.qpos[:6].copy()
        v_arm = self.d.qvel[:6].copy()
        for _ in range(n):
            mujoco.mj_step(self.m, self.d)
            self.d.qpos[:6] = q_arm          # 臂不吃接触反力：运动学驱动
            self.d.qvel[:6] = v_arm
            self._hold()
            if self.on_step is not None:
                self.on_step(self)
            if self._viewer is not None:
                # 渲染限到 60 Hz：物理是 500 Hz，每步都 sync 的话渲染会把时间全吃掉
                # （实测比不渲染慢一倍多）。视觉上 60 fps 已经够顺。
                now = time.perf_counter()
                if now - self._last_sync >= 1.0 / 60.0:
                    self._viewer.sync()
                    self._last_sync = now
                if now - self._last_key >= 0.1:
                    self._poll_speed_keys()
                    self._last_key = now

    def _move_joints(self, q_goal_deg, vel=20.0):
        q0 = np.array(self._current_joints())
        q1 = np.array(_six(q_goal_deg, "目标关节"))
        # 每个关节走**最短方向**：±360 是同一个角度，若直接相减就会绕一大圈
        # （典型表现就是 J6 在两次动作之间甩一整圈，看着像"一直转"，还白耗时）。
        dq = ((q1 - q0 + 180.0) % 360.0) - 180.0
        span = float(np.abs(dq).max())
        if span < 1e-6:
            return
        rate = max(self.speed_deg_s * max(vel, 1.0) / 100.0, 1.0)
        duration = span / rate * self.time_scale
        # 关节只转 1° 的短程（典型是松手后抬 30 mm：主要是平移，关节角几乎不变）
        # 会被算成十几毫秒，回放里就是"瞬移"。给真实动作一个最小时长，纯观感，
        # 不影响落点。小于 0.5° 的微调不套这个下限，否则一整套动作会拖很久。
        if span >= 0.5:
            duration = max(duration, self.min_move_seconds)
        steps = max(3, int(duration / max(self.m.opt.timestep, 0.002) / 2))
        for k in range(1, steps + 1):
            self.d.qpos[:6] = np.radians(q0 + dq * k / steps)
            self.d.qvel[:6] = 0.0
            self._settle(2)
        self.d.qpos[:6] = np.radians(q0 + dq)   # 到终点（等价的最近角度，不是绕远）
        self.d.qvel[:6] = 0.0
        self._settle(6)

    def move_to(self, pose, vel=20.0):
        q = self._pose_to_joint(pose, self._current_joints(), "MoveJ")
        self._move_joints(q, vel)

    def move_home(self, home_joint, vel=20.0):
        self._move_joints(_six(home_joint, "HOME关节"), vel)

    @staticmethod
    def above_pose(pose, dz):
        target = _six(pose, "目标位姿")
        if not math.isfinite(float(dz)):
            raise ValueError("高度偏移必须有限")
        target[2] += float(dz)
        return target

    def move_above(self, pose, dz=10.0, vel=20.0):
        self.move_to(self.above_pose(pose, dz), vel)

    # ------------------------------------------------------------------ suction
    def _body_aabb_centre(self, b):
        """物体在**世界系**的几何中心。

        不能用 `d.xpos[b]`：CAD 里每块板的自身原点离plate中心可达 57 mm
        （是建模原点，不是几何中心），拿它当"这块板在哪"会吸错或吸不到。
        """
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for g in range(self.m.ngeom):
            if self.m.geom_bodyid[g] != b:
                continue
            gt = self.m.geom_type[g]
            if gt == mujoco.mjtGeom.mjGEOM_MESH:
                mid = self.m.geom_dataid[g]
                adr, num = self.m.mesh_vertadr[mid], self.m.mesh_vertnum[mid]
                v = self.m.mesh_vert[adr:adr + num]
            elif gt == mujoco.mjtGeom.mjGEOM_BOX:
                s = self.m.geom_size[g]
                v = np.array([[sx, sy, sz] for sx in (-s[0], s[0])
                              for sy in (-s[1], s[1]) for sz in (-s[2], s[2])])
            else:
                continue
            w = v @ self.d.geom_xmat[g].reshape(3, 3).T + self.d.geom_xpos[g]
            lo = np.minimum(lo, w.min(axis=0))
            hi = np.maximum(hi, w.max(axis=0))
        return (lo + hi) / 2.0

    def _body_world_vertices(self, b):
        """这个 body 所有网格顶点在世界系的坐标（N,3）。"""
        chunks = []
        for g in range(self.m.ngeom):
            if self.m.geom_bodyid[g] != b:
                continue
            gt = self.m.geom_type[g]
            if gt == mujoco.mjtGeom.mjGEOM_MESH:
                mid = self.m.geom_dataid[g]
                adr, num = self.m.mesh_vertadr[mid], self.m.mesh_vertnum[mid]
                v = self.m.mesh_vert[adr:adr + num]
            elif gt == mujoco.mjtGeom.mjGEOM_BOX:
                s = self.m.geom_size[g]
                v = np.array([[sx, sy, sz] for sx in (-s[0], s[0])
                              for sy in (-s[1], s[1]) for sz in (-s[2], s[2])])
            else:
                continue
            chunks.append(v @ self.d.geom_xmat[g].reshape(3, 3).T + self.d.geom_xpos[g])
        return np.vstack(chunks) if chunks else None

    def _cup_body(self):
        """杯体（吸盘口所在的 body）：新 arm 叫 Link_14，旧的叫 suction_link。"""
        if self._cup < 0:
            for nm in ("Link_14", "suction_link"):
                b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, nm)
                if b >= 0:
                    self._cup = b
                    break
            else:
                self._cup = 0
        return self._cup

    def bellows_compression(self):
        """当前波纹管压缩量（mm）：按"杯口被压到板面/托盘下面多少"算，夹在行程内。"""
        if self._slide < 0:
            return 0.0
        return float(self.d.qpos[self._slide]) * 1000.0

    def _update_bellows(self):
        """每步把滑动关节推向目标压缩量（管子始终是活的）。

        目标 = 下压深度，夹在 0..行程 之间：
          - 没吸住：面 = 杯口指令位置 - 间隙（板面或托盘面），压多深缩多少；
          - 吸住了：面 = 吸住那一刻记下的那个面，照样压多少缩多少，抬手就松开。

        两个坑：
          1. 必须按**指令位置**（滑轨当 0）量，不能按杯口当前位置量，否则成反馈环，
             表现是压到板面上之后在 0 和满行程之间来回跳；
          2. 变化要限速，否则吸住/放开那一瞬间杯子上的板子会瞬移。
        """
        if self._slide < 0:
            return

        cur_mm = float(self.d.qpos[self._slide]) * 1000.0

        # ---- 快速跳过：离板子还远、也没吸住、压缩为 0，就没必要每步都算 ----
        # 这里每步要做两次 mj_forward + 一次吸盘判据，无窗口快速跑时不值得；
        # 30 mm 之外不可能压缩，直接返回。
        if self.attached is None and cur_mm < 1e-3:
            tcp_now = float(self.d.site_xpos[self._tcp][2]) * 1000.0
            if tcp_now > self._board_top_z + 30.0:
                return

        # ---- 量一次"指令位置"（滑轨当 0）下的杯口高度 --------------------------
        saved = float(self.d.qpos[self._slide])
        self.d.qpos[self._slide] = 0.0
        mujoco.mj_forward(self.m, self.d)
        tcp_z = float(self.d.site_xpos[self._tcp][2]) * 1000.0
        lift = self.suction_status()["lift_mm"]

        ref = getattr(self, "_press_ref_z", None)
        if self.attached is None:
            ref = None if lift is None else tcp_z - lift      # 面 = 杯口 - 间隙
            self._press_ref_z = ref
        elif ref is None:
            # 刚吸住第一次：把吸住瞬间那个面记下来，之后一直拿它当基准
            ref = None if lift is None else tcp_z - lift
            self._press_ref_z = ref

        self.d.qpos[self._slide] = saved

        # 压入深度 = 面 - 指令杯口高度；负值（杯口在面之上）就是 0
        target = 0.0 if ref is None else float(
            np.clip(ref - tcp_z, 0.0, self.suction_stroke_mm))

        # 限速：波纹管按 ~0.5 m/s 收放，2 ms 一步就是 1 mm
        step_mm = 500.0 * float(self.m.opt.timestep)
        delta = float(np.clip(target - cur_mm, -step_mm, step_mm))
        if abs(delta) > 1e-3:
            self.d.qpos[self._slide] = (cur_mm + delta) / 1000.0
            self.d.qvel[self._slide] = 0.0
            mujoco.mj_forward(self.m, self.d)

    def suction_status(self):
        """吸盘现在对着哪块板、差在哪 —— 给沙盒做实时提示用。

        `lift_mm` 的符号约定：**正数 = 吸盘口在板面上方（还有缝）**，负数 = 压进去了。
        返回 dict：name(最近的一块), lift_mm, tilt_deg, over(是否在轮廓内),
        dxy_mm(离板子中心多远), ok, reason。
        """
        tcp = self.d.site_xpos[self._tcp]
        tool = self._cup_body()
        axis = self.d.xmat[tool].reshape(3, 3)[:, 2]
        tilt = float(np.degrees(np.arccos(np.clip(abs(axis[2]), 0.0, 1.0))))

        best = {"name": None, "lift_mm": None, "tilt_deg": tilt, "over": False,
                "dxy_mm": 1e9, "ok": False, "reason": "没有板子"}
        for name in self._eq:
            b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)
            if b < 0:
                continue
            verts = self._body_world_vertices(b)
            if verts is None or len(verts) < 4:
                continue
            centre = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
            d_xy = float(np.linalg.norm(centre[:2] - tcp[:2])) * 1000.0
            if d_xy >= best["dxy_mm"]:
                continue
            lift = (tcp[2] - float(verts[:, 2].max())) * 1000.0
            over = self._point_over_tile(verts, tcp)
            if tilt > self.suction_max_tilt_deg:
                reason = f"工具轴歪 {tilt:.0f}°"
            elif lift > self.suction_max_gap_mm:
                reason = f"还差 {lift:.1f} mm 才碰到板面"
            elif not over:
                reason = f"吸盘口不在板子上（偏 {d_xy:.0f} mm）"
            elif lift < -max(self.suction_stroke_mm, self.suction_max_press_mm):
                reason = f"压进板面 {-lift:.1f} mm，超过 {self.suction_stroke_mm:.0f} mm 行程"
            else:
                reason = "对口，可以吸"
            best = {"name": name, "lift_mm": lift, "tilt_deg": tilt,
                    "over": over, "dxy_mm": d_xy,
                    "ok": reason == "对口，可以吸", "reason": reason}
        return best

    def _suction_on(self):
        """吸住吸盘口正下方的那块板。

        判据不是"离得最近"，而是三条真的物理条件（橡胶唇要贴上才封得住）：

        1. **吸盘口在板子的轮廓之内**（把板子的顶点投到 xy 上做凸包，看点是否在里面）；
        2. **竖直间隙合适**：吸盘口在板面上方 `suction_max_gap_mm`（默认 8）到下方 4 mm；
        3. **工具轴与板面法向夹角 ≤ `suction_max_tilt_deg`**（默认 20°，歪着漏气）。

        板子在那之后由 `_hold()` 每步强制跟随吸盘（刚性吸附）。
        用 `<weld>` 也行，但它在抓取瞬间要让求解器把板子"拽"过去，容易弹。
        """
        # 和 `_hold()` 同一个道理：判"吸盘口在不在板子上"、以及记下吸住时的
        # 相对位姿，都要用**当前**的 xpos，不能是上一步的。
        mujoco.mj_forward(self.m, self.d)
        st = self.suction_status()
        if not st["ok"]:
            lift = st["lift_mm"]
            lift_txt = "—" if lift is None else f"{lift:+.1f} mm"
            print(f"[sim] suction: 没吸 —— {st['reason']}"
                  f"（最近 {st['name']}，吸盘口离板面 {lift_txt}，"
                  f"倾角 {st['tilt_deg']:.1f}°）")
            return False

        name = st["name"]
        b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)
        tool = self._cup_body()
        R1 = self.d.xmat[tool].reshape(3, 3)
        p1 = self.d.xpos[tool]
        R2 = self.d.xmat[b].reshape(3, 3)
        p2 = self.d.xpos[b]
        # 基准面必须在**贴合之前**量：用"指令杯口位置 - 间隙"推出板子顶面（含板厚）。
        # 如果等吸住之后再量，板子已经被焊在吸盘上跟着动，量出来的面就是杯口自己，
        # 压缩量算成 0，板子会被压进托盘里（这个 bug 踩过）。
        saved = float(self.d.qpos[self._slide]) if self._slide >= 0 else 0.0
        if self._slide >= 0:
            self.d.qpos[self._slide] = 0.0
            mujoco.mj_forward(self.m, self.d)
        tcp_cmd = float(self.d.site_xpos[self._tcp][2]) * 1000.0
        if self._slide >= 0:
            self.d.qpos[self._slide] = saved
            mujoco.mj_forward(self.m, self.d)
        self._press_ref_z = None if st["lift_mm"] is None else tcp_cmd - st["lift_mm"]

        self._rel = (R1.T @ R2, R1.T @ (p2 - p1), tool, b)
        self._gripped = name
        # 吸住那一刻板子的高度：之后压低时不许低于它（托盘顶着）
        self._grab_piece_z = float(p2[2])
        self._settle(3)
        print(f"[sim] suction ON -> {name} (吸盘口离板面 {st['lift_mm']:+.2f} mm, "
              f"倾角 {st['tilt_deg']:.1f}°, 波纹管 {self.bellows_compression():.1f} mm)")
        return True

    def _suction_off(self):
        # 放之前用当前吸盘位姿把板子最后钉一次，让"自由体的初始状态"就是
        # "吸盘命令位姿下的板子"。实测这本身不改变落点（真正的偏差在 `_settle()`
        # 没钉住臂上），但它把"松手瞬间"定义清楚了，不依赖调用顺序。
        if self._rel is not None:
            self._hold()
        if self._gripped:
            print(f"[sim] suction OFF <- {self._gripped}")
        self._gripped = None
        self._rel = None
        self._press_ref_z = None        # 放开后重新按"当前正下方的面"算
        self._settle(3)

    @staticmethod
    def _point_over_tile(verts, point):
        """把点投到 xy，判断它是否落在这块板顶面的凸包里。"""
        xy = verts[:, :2]
        try:
            from scipy.spatial import ConvexHull, Delaunay
        except ImportError:                              # 兜底：用外接矩形
            lo, hi = xy.min(axis=0), xy.max(axis=0)
            return bool(np.all(point[:2] >= lo) and np.all(point[:2] <= hi))
        try:
            hull = ConvexHull(xy)
        except Exception:                                # noqa: BLE001  退化（共线）
            return False
        return bool(Delaunay(xy[hull.vertices]).find_simplex(point[:2]) >= 0)

    def _hold(self):
        """Keep the grasped tile fixed relative to the cup.

        真正的"落点偏 2–12 mm"是 `_settle()` 没钉住臂造成的（接触冲量踢手腕，
        见那边的注释），不是这里。这里加一次 `mj_forward` 只是防御：`_hold` 要在
        "改完波纹管、臂刚被写回"之后读到**当前**的吸盘位姿，而不是上一步的；
        实测它本身不改结果，但不加的话就依赖调用顺序，后面容易踩。
        """
        self._update_bellows()
        if self._rel is None:
            return
        mujoco.mj_forward(self.m, self.d)
        Rp, pp, tool, b = self._rel
        R1 = self.d.xmat[tool].reshape(3, 3)
        p1 = self.d.xpos[tool]
        T = np.eye(4)
        T[:3, :3] = R1 @ Rp
        T[:3, 3] = R1 @ pp + p1
        # 板子不许被压到托盘下面：吸住时它是搁在托盘上的，托盘顶着它。
        # 压缩量有速率限制，法兰压下去的瞬间吸盘会短暂滞后；如果板子严格跟着吸盘，
        # 就会被"拽"下去十几毫米（松开吸盘才发现它已经陷进托盘里）。
        grab_z = getattr(self, "_grab_piece_z", None)
        if grab_z is not None and T[2, 3] < grab_z:
            T[2, 3] = grab_z
        jid = self.m.body_jntadr[b]
        adr = self.m.jnt_qposadr[jid]
        dof = self.m.jnt_dofadr[jid]
        q = trimesh.transformations.quaternion_from_matrix(T)
        self.d.qpos[adr:adr + 3] = T[:3, 3]
        self.d.qpos[adr + 3:adr + 7] = q / np.linalg.norm(q)
        self.d.qvel[dof:dof + 6] = 0.0

    # ------------------------------------------------------- manual sandbox API
    @property
    def attached(self):
        """当前吸住的那块板的名字，没吸住就是 None。"""
        return self._gripped

    def suction_grab(self):
        return self._suction_on()

    def suction_release(self):
        self._suction_off()

    def hold_attached(self):
        self._hold()

    # ------------------------------------------------------- high-level actions
    # 2026-11-02 删掉了 `_free_yaw()`（"抓取时按 90° 整数倍换朝向省 J6"）。
    # 它当时的理由是"吸盘是圆的，转多少都吸得住，物理上完全等价"——**不成立**：
    # 板子是"抓的时候怎么摆，后面就跟着法兰一起转"的，抓取朝向换掉 90k°，
    # 放置时的净转角就变成 delta-90k°，板子最后是转错方向的。
    # 实测（plans.json 逐块复现）：M1/T1/T2 各偏 +90°、P1 偏 +270°，
    # 正好对上"位置对、部分块旋转方向错"的现象。
    # 要省 J6 只能改整体抓放朝向（run.pick_grasp_yaw，对所有块同一个角），不能逐块改。
    def pick(self, pose, gripper, approach_dz=10.0, vel=20.0, dip_dz=0.0):
        pose = _six(pose, "抓取位姿")
        self.move_above(pose, approach_dz, vel)
        self.move_to(pose, vel)
        if dip_dz:
            self.move_above(pose, -dip_dz, vel)
        gripper.grip()
        self.move_above(pose, approach_dz, vel)

    def place(self, pose, gripper, approach_dz=10.0, vel=20.0):
        self.move_above(pose, approach_dz, vel)
        self.move_to(pose, vel)
        gripper.release()
        self.move_above(pose, approach_dz, vel)

    def preflight_pick_place(self, plans, approach_dz):
        self.check_coordinate_context()
        if not math.isfinite(float(approach_dz)) or approach_dz <= 0:
            raise ValueError("接近高度必须为正")
        reference = self._current_joints()
        report = []
        for plan in plans:
            grasp = _six(plan.grasp_pose, "抓取位姿")
            place = _six(plan.place_pose, "放置位姿")
            ga = self.above_pose(grasp, approach_dz)
            pa = self.above_pose(place, approach_dz)
            for stage, pose in (("grasp_above", ga), ("grasp", grasp), ("grasp_retract", ga),
                                ("place_above", pa), ("place", place), ("place_retract", pa)):
                context = f"ID={plan.piece_id}({getattr(plan, 'color_name', '?')})/{stage}"
                joints = self._pose_to_joint(pose, reference, context)
                report.append({"piece_id": plan.piece_id, "stage": stage, "pose": pose,
                               "reference_joints": reference, "joints": joints})
                reference = joints
        return report

    # -------------------------------------------------------------------- utils
    def _base_frame(self):
        return (self.d.xmat[self._base].reshape(3, 3), self.d.xpos[self._base].copy())

    def world_to_base_point(self, p_world_mm):
        """场景世界系 -> 机器人基座系（mm）。控制器给的都是基座系。"""
        Rb, pb = self._base_frame()
        return (Rb.T @ (np.asarray(p_world_mm, dtype=float) / 1000.0 - pb)) * 1000.0

    def base_to_world_point(self, p_base_mm):
        Rb, pb = self._base_frame()
        return (Rb @ (np.asarray(p_base_mm, dtype=float) / 1000.0) + pb) * 1000.0

    def world_to_base_pose(self, pose_world):
        """把世界系位姿换成基座系位姿（位置和姿态都转）。"""
        p = self.world_to_base_point(pose_world[:3])
        Rb, _ = self._base_frame()
        Rw = rpy_zyx(*pose_world[3:])
        R = Rb.T @ Rw
        b = math.asin(np.clip(-R[2, 0], -1, 1))
        a = math.atan2(R[2, 1], R[2, 2])
        c = math.atan2(R[1, 0], R[0, 0])
        return list(p) + list(np.degrees([a, b, c]))

    def get_camera_image(self, width=1280, height=720):
        """wrist_cam 的当前画面，BGR uint8，和 RealSenseCamera.grab_color 一致。"""
        ren = mujoco.Renderer(self.m, height, width)
        ren.update_scene(self.d, camera="wrist_cam")
        rgb = ren.render()
        ren.close()
        return rgb[:, :, ::-1].copy()

    def step(self, seconds=0.2):
        for _ in range(max(1, int(seconds / self.m.opt.timestep))):
            self._settle(1)
