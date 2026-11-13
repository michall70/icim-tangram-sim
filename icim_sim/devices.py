# -*- coding: utf-8 -*-
"""把队友的 `main.py` 接到仿真上，且**不改他们一行代码**。

做法是"顶替"而不是"改造"：他们 `run()` 里在函数内部才 import 设备类，
所以在调用之前把模块属性换掉，`from common.realrobot import CobotClient`
拿到的就是仿真替身。

    from icim_sim.devices import bootstrap_env
    env = bootstrap_env()
    env.main.run(execute=True)

三处硬件的替身：
    机器人  common.robot_client.CobotClient   -> SimCobotClient
    相机    common.realsense_camera.RealSenseCamera -> SimCamera（带镜头畸变）
    夹爪    不动，RobotDOGripper 走 robot.rpc.SetDO，仿真里已经接上吸盘

两个器件库没法装（`pyrealsense2` 要真相机，`opcua` 要 PLC），
所以只在 `sys.modules` 里放一个空壳，让 import 能过；真正用到它们的
代码路径全被上面两个替身接管了。

抓放高度
--------
`GRIP_Z` / `PLACE_Z` 一律用 `sim.cell_facts` 从模型里量出来的值覆盖。
这是"以模型为准"的落地：模型改了，这里自动跟着变，不需要谁去改配置。
"""
from __future__ import annotations

import functools
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

from .paths import PACKAGE_ROOT, THIRD_PARTY
from .team import TeamLayout, discover, ensure_importable, resolve_root


# --------------------------------------------------------------------- stubs
class _Missing:
    """占位对象：任何属性/调用都给出明确报错，而不是静默 None。"""

    def __init__(self, what):
        self._what = what

    def __getattr__(self, name):
        raise RuntimeError(f"{self._what} 在仿真环境里不可用（被仿真设备顶替了），"
                           f"不应该访问 .{name}")


def _stub(name, **attrs):
    mod = sys.modules.get(name)
    if mod is not None:
        return mod
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    mod.__doc__ = "simulation stub"
    sys.modules[name] = mod
    return mod


def install_stubs():
    """让 `import pyrealsense2` / `from opcua import ...` 不至于炸。"""
    if "pyrealsense2" not in sys.modules:
        try:
            import pyrealsense2  # noqa: F401
        except ImportError:
            _stub("pyrealsense2")
    if "opcua" not in sys.modules:
        try:
            import opcua  # noqa: F401
        except ImportError:
            _stub("opcua", Client=_Missing("OPC UA 客户端"),
                  ua=_Missing("OPC UA 数据类型"),
                  Node=_Missing("OPC UA 节点"))


# ------------------------------------------------------------------ bootstrap
@dataclass
class TeammateEnv:
    config: types.ModuleType
    main: types.ModuleType
    robot_module: types.ModuleType
    camera_module: types.ModuleType
    CameraMapper: type
    facts: object
    robot: object = None
    #: 机器人替身造出来时回调一下（录像、挂调试钩子用）
    on_robot: object = None
    #: 初始布局的名字或路径（"auto"/"none"/layouts 下的名字），只是字符串
    layout: object = None
    #: 认出来的队友代码布局（各角色 -> 模块名），见 icim_sim/team.py
    team: object = None
    #: 开局要翻面的板（手性不符时由 icim_sim.chirality 填）
    flips: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def report(self) -> str:
        return "\n".join(self.notes)


def _resolve_layout(team_repo=None) -> TeamLayout:
    """确定队友仓库在哪、里面哪些模块是什么（不猜目录名，也不猜有几份）。"""
    root = resolve_root(THIRD_PARTY, team_repo)      # 显式指定 / active.txt / 唯一
    ensure_importable(PACKAGE_ROOT)
    ensure_importable(root)
    layout = discover(root)
    if not layout.modules:
        raise SystemExit(f"{root} 下没认出任何队友代码，检查是不是拷错了目录")
    return layout


def load_config(team_repo=None):
    """只导入队友的 config（它只依赖 os/numpy，不需要任何替身）。"""
    return _resolve_layout(team_repo).require("config")


def bootstrap_env(*, simulate_lens=True, follow_model=True,
                  grip_z=None, place_z=None, viewer=False, team_repo=None,
                  layout="auto"):
    """把仿真环境装起来，返回一个可以调他们 `main.run()` 的句柄。

    顺序很关键：**先打补丁，再导入他们的 main**。这样不管他们的设备 import
    写在文件开头还是写在 `run()` 里面，拿到的都是仿真替身。
    """
    team_layout = _resolve_layout(team_repo)
    print(f"[team] 队友代码：{team_layout.root}")
    print(team_layout.describe())
    install_stubs()

    from icim_sim import cell
    from icim_sim.camera import SimCamera, set_active_robot
    from icim_sim.robot import SimCobotClient

    # 1) 先拿到机器人/相机/标定三个**模块对象**（还没导入 main，补丁一定来得及）
    try:
        robot_module = team_layout.require("robot_client")
        camera_module = team_layout.require("camera")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"导入队友模块失败：{exc}\n"
            f"如果是缺某个硬件 SDK（例如 pyrealsense2/opcua），"
            f"在 icim_sim/devices.py 的 install_stubs() 里加一个空壳即可。") from exc
    calibration = team_layout.get("calibration")

    facts = cell.measure()
    notes: list = []

    # 2) 先建 env（make_robot 里要用），再改模块属性：他们的
    #    `from common.robot_client import CobotClient` 不管写在哪儿，
    #    取到的都是这里换掉的对象
    env = TeammateEnv(config=None, main=None, robot_module=robot_module,
                      camera_module=camera_module,
                      CameraMapper=getattr(calibration, "CameraMapper", None),
                      facts=facts, notes=notes, layout=layout, team=team_layout)

    def make_robot(ip, tool=0, user=0, sdk_version="auto", **_kw):
        r = SimCobotClient(ip=ip, tool=tool, user=user, sdk_version=sdk_version,
                           viewer=viewer, layout=layout, flips=list(env.flips))
        env.robot = r
        set_active_robot(r)
        if env.on_robot is not None:
            env.on_robot(r)
        return r

    robot_module.CobotClient = make_robot

    # 3) 摄像头补丁只需要 config 里的内参，所以先把 config 单独导进来
    config = team_layout.require("config")

    camera_module.RealSenseCamera = functools.partial(
        SimCamera, camera_matrix=config.CAMERA_MATRIX,
        dist_coeffs=config.DIST_COEFFS, simulate_lens=simulate_lens)

    # 4) 两个替身都就位之后，才导入他们的 main
    scene_main = team_layout.require("main")

    env.config = config
    env.main = scene_main

    # ---- heights ---------------------------------------------------------
    if follow_model:
        grip_z = facts.grip_z_mm if grip_z is None else grip_z
        place_z = facts.place_z_mm if place_z is None else place_z
        for mod, label in ((config, "config"), (scene_main, "main")):
            old_g = getattr(mod, "GRIP_Z", None)
            old_p = getattr(mod, "PLACE_Z", None)
            mod.GRIP_Z = grip_z
            mod.PLACE_Z = place_z
            notes.append(f"  {label}.GRIP_Z {old_g} -> {grip_z:.3f}"
                         f"   PLACE_Z {old_p} -> {place_z:.3f}")
        notes.insert(0, f"[env] 抓放高度改用模型值（{facts.scene}）：")
        notes.append(f"       = 板面 {facts.tile_top_base_mm:.3f} + 工具 "
                     f"{facts.flange_to_tip_mm:.3f} mm")

    notes.append(f"[env] 相机：{config.CAMERA_PARAMS['width']}×"
                 f"{config.CAMERA_PARAMS['height']}，"
                 f"{'带' if simulate_lens else '不带'}镜头畸变")
    notes.append(f"[env] 相机横纵内参 fx={config.CAMERA_MATRIX[0,0]:.2f} "
                 f"fy={config.CAMERA_MATRIX[1,1]:.2f} "
                 f"cx={config.CAMERA_MATRIX[0,2]:.2f} cy={config.CAMERA_MATRIX[1,2]:.2f}")

    return env
