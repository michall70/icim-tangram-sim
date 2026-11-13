"""协作机器人运动控制封装。

基于 common/robot_sdk 下双版本厂商 SDK（3.8.2 / 3.9.3）。两台机器人 IP 相同，
无法按 IP 区分，连接后调用一次 GetSoftwareVersion() 读取控制器实际版本号
自动匹配 SDK；也可通过 sdk_version 显式指定版本跳过探测（用于排查问题）。

位姿格式 [x, y, z, rx, ry, rz]，单位 mm/°；关节格式为 6 关节角列表，单位 °。
抓放动作序列沿用 RobotTest-20250909.py 验证过的规范流程：
上方安全高度接近 → MoveJ 到位 → 夹爪动作 → 抬升回安全高度。
MoveJ 为关节空间运动，末端轨迹不保证为直线。
"""

import threading
import math

from common.robot_sdk import SUPPORTED_VERSIONS, load_robot_sdk


def query_controller_version(rpc, timeout=5.0):
    """查询控制器软件版本，返回去掉错误码后的版本分量列表。

    SDK 内部对 socket 错误无限重试，机器人不在线时会一直挂起，
    此处用守护线程 + 超时兜底，保证探测流程最终能报错返回。
    """
    result = {}

    def _call():
        try:
            result["ret"] = rpc.GetSoftwareVersion()
        except Exception as exc:
            result["exc"] = exc

    worker = threading.Thread(target=_call, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError(f"查询控制器版本超时({timeout}s)")
    if "exc" in result:
        raise result["exc"]
    ret = result["ret"]
    error, parts = ret[0], ret[1:]
    if error != 0:
        raise RuntimeError(f"GetSoftwareVersion 返回错误码 {error}")
    return parts


class IKPreflightError(RuntimeError):
    """保留具体失败位姿和参考关节，供离线排查；不自动调整目标或重试。"""
    def __init__(self, code, pose, reference, context):
        self.details = {"action":"GetInverseKinRef", "error_code":code,
                        "pose":list(pose), "reference_joints":list(reference), "context":context}
        hint = "给定位姿无法到达" if code == 112 else "逆解失败"
        super().__init__(f"{context}: {hint}, 错误码={code}, pose={pose}, reference={reference}")


class CobotClient:

    def __init__(self, ip, tool=0, user=0, sdk_version="auto"):
        self.ip = ip
        self.tool = tool
        self.user = user
        self.sdk_version = sdk_version
        self.rpc = None
        self.controller_version = None

    def connect(self):
        """连接机器人；sdk_version="auto" 时按控制器实际版本自动匹配 SDK。"""
        if self.sdk_version != "auto" and self.sdk_version not in SUPPORTED_VERSIONS:
            raise ValueError(f"不支持的 SDK 版本: {self.sdk_version}，可选: {SUPPORTED_VERSIONS}")
        candidates = [self.sdk_version] if self.sdk_version != "auto" else list(SUPPORTED_VERSIONS)

        for version in candidates:
            rpc = None
            try:
                rpc = load_robot_sdk(version).RPC(self.ip)
                parts = query_controller_version(rpc)
            except Exception as exc:
                print(f"[robot] 使用 {version} SDK 连接 {self.ip} 失败: {exc}")
                self._safe_close(rpc)
                continue

            actual = self._match_version(parts)
            if self.sdk_version == "auto" and actual != version:
                print(f"[robot] 控制器实际版本 {self._format_parts(parts)} 与 {version} SDK 不匹配，尝试其他版本")
                self._safe_close(rpc)
                continue
            if actual is not None and actual != version:
                print(f"[robot] 警告: 控制器实际版本 {actual} 与指定 SDK {version} 不匹配，仍按指定版本继续")

            self.rpc = rpc
            self.controller_version = actual or version
            print(f"[robot] 已连接机器人 {self.ip}，控制器版本 {self.controller_version}，使用 {version} SDK")
            return

        raise ConnectionError(f"无法连接机器人 {self.ip} 或控制器版本无法适配，已尝试 SDK 版本: {candidates}")

    def disconnect(self):
        if self.rpc is not None:
            self.rpc.CloseRPC()
            self.rpc = None

    @staticmethod
    def _match_version(parts):
        """在 GetSoftwareVersion 返回的各版本分量中查找支持的版本号。"""
        for version in SUPPORTED_VERSIONS:
            for comp in parts:
                if comp is not None and version in str(comp):
                    return version
        return None

    @staticmethod
    def _format_parts(parts):
        return "/".join(str(p) for p in parts if p is not None)

    @staticmethod
    def _safe_close(rpc):
        if rpc is not None:
            try:
                rpc.CloseRPC()
            except Exception:
                pass

    def _check(self, error, action):
        if error != 0:
            raise RuntimeError(f"机器人指令失败({action})，错误码: {error}")

    def move_home(self, home_joint, vel=20.0):
        """MoveJ 回 HOME 点。"""
        home_joint=self._six(home_joint,"HOME关节")
        self._check(self.rpc.MoveJ(home_joint, self.tool, self.user, vel=vel), "MoveJ回HOME")

    @staticmethod
    def _six(values, label):
        if not isinstance(values, (list, tuple)) or len(values) != 6:
            raise ValueError(f"{label}必须有六个数")
        result = [float(v) for v in values]
        if not all(math.isfinite(v) for v in result):
            raise ValueError(f"{label}含非有限数值")
        return result

    def check_coordinate_context(self):
        # 本项目单应矩阵以基坐标定义；尚未实现任意工具/工件坐标变换。
        if self.tool != 0 or self.user != 0:
            raise ValueError("当前规划仅核对过工具0/工件0；切换坐标系需先重新验证映射")
        for getter, expected, label in ((self.rpc.GetActualTCPNum,self.tool,"工具"),
                                        (self.rpc.GetActualWObjNum,self.user,"工件")):
            ret=getter()
            if not isinstance(ret,(tuple,list)) or len(ret)!=2:
                raise RuntimeError(f"读取{label}编号失败: {ret}")
            self._check(ret[0],f"读取{label}编号")
            if ret[1]!=expected:
                raise RuntimeError(f"当前{label}编号={ret[1]}，代码要求={expected}；停止，不自动切换")

    def _current_joints(self):
        ret=self.rpc.GetActualJointPosDegree()
        if not isinstance(ret,(tuple,list)) or len(ret)!=2:
            raise RuntimeError(f"读取当前关节失败: {ret}")
        self._check(ret[0],"读取当前关节")
        return self._six(ret[1],"当前关节")

    def get_tcp_pose(self):
        """读取当前 TCP 位姿 [x, y, z, rx, ry, rz]，与 MoveJ 的 desc_pos 同一口径。

        标定用它替代"人眼看示教器再手抄"。返回含工具/工件坐标的 TCP，
        因此必须与运行时 `check_coordinate_context` 保持同一 tool/user。
        """
        ret=self.rpc.GetActualTCPPose()
        if not isinstance(ret,(tuple,list)) or len(ret)!=2:
            raise RuntimeError(f"读取当前 TCP 位姿失败: {ret}")
        self._check(ret[0],"读取当前 TCP 位姿")
        return self._six(ret[1],"当前 TCP 位姿")

    def _pose_to_joint(self, pose, reference=None, context="运动逆解"):
        pose=self._six(pose,"目标位姿")
        reference=self._current_joints() if reference is None else self._six(reference,"参考关节")
        ret=self.rpc.GetInverseKinRef(0,pose,reference)
        error=ret[0] if isinstance(ret,(tuple,list)) and ret else ret
        if error != 0:
            raise IKPreflightError(error,pose,reference,context)
        if not isinstance(ret,(tuple,list)) or len(ret)!=2:
            raise RuntimeError(f"逆解返回格式错误: {ret}")
        return self._six(ret[1],"逆解关节")

    @staticmethod
    def above_pose(pose, dz):
        target=CobotClient._six(pose,"目标位姿")
        if not math.isfinite(float(dz)):
            raise ValueError("高度偏移必须有限")
        target[2]+=float(dz)
        return target

    def preflight_pick_place(self, plans, approach_dz):
        """只读当前状态并逐点逆解，不发送运动/IO；不代表碰撞与轨迹验证。"""
        self.check_coordinate_context()
        if not math.isfinite(float(approach_dz)) or approach_dz<=0:
            raise ValueError("接近高度必须为正")
        reference=self._current_joints()
        report=[]
        for plan in plans:
            grasp=self._six(plan.grasp_pose,"抓取位姿")
            place=self._six(plan.place_pose,"放置位姿")
            ga=self.above_pose(grasp,approach_dz)
            pa=self.above_pose(place,approach_dz)
            for stage,pose in (("grasp_above",ga),("grasp",grasp),("grasp_retract",ga),
                               ("place_above",pa),("place",place),("place_retract",pa)):
                context=f"ID={plan.piece_id}({plan.color_name})/{stage}"
                joints=self._pose_to_joint(pose,reference,context)
                report.append({"piece_id":plan.piece_id,"stage":stage,"pose":pose,
                               "reference_joints":reference,"joints":joints})
                reference=joints
        return report

    def move_to(self, pose, vel=20.0):
        """位姿逆解后 MoveJ 到位（blendT=-1 阻塞至运动完成）。"""
        joint_pos = self._pose_to_joint(pose)
        self._check(self.rpc.MoveJ(joint_pos, self.tool, self.user,
                                   desc_pos=pose, vel=vel, blendT=-1.0), "MoveJ")

    def move_above(self, pose, dz=10.0, vel=20.0):
        """先构造真实上方目标，再对同一目标逆解/MoveJ，避免偏移点与逆解点不同。"""
        self.move_to(self.above_pose(pose,dz),vel)

    def pick(self, pose, gripper, approach_dz=10.0, vel=20.0, dip_dz=0.0):
        """规范抓取：上方接近 → 到位 →（可选再下探 dip_dz）→ 闭合夹爪 → 抬升。"""
        self.move_above(pose, approach_dz, vel)
        self.move_to(pose, vel)
        if dip_dz:
            self.move_above(pose, -dip_dz, vel)
        gripper.grip()
        self.move_above(pose, approach_dz, vel)

    def place(self, pose, gripper, approach_dz=10.0, vel=20.0):
        """规范放置：上方接近 → 到位 → 张开夹爪 → 抬升。"""
        self.move_above(pose, approach_dz, vel)
        self.move_to(pose, vel)
        gripper.release()
        self.move_above(pose, approach_dz, vel)
