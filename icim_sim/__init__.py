# -*- coding: utf-8 -*-
"""ICIM 七巧板工位的 MuJoCo 仿真平台。

给上层的用法（四种）：

    # 1) 只想要一个和真机一样接口的机器人
    from icim_sim.robot import SimCobotClient
    robot = SimCobotClient(); robot.connect()
    robot.move_home(HOME_JOINT); robot.move_to(pose)

    # 2) 想让**你们自己的** main.py 直接跑在仿真上（一行都不改）
    from icim_sim.devices import bootstrap_env
    env = bootstrap_env()            # third_party/ 下自动找他们的代码
    env.main.run(execute=True)

    # 3) 想要工位参数（拍照位、单应、ROI、抓放高度、实物尺度）
    from icim_sim.calib import measure_calib
    c = measure_calib(); print(c.report())

    # 4) 想要一张"赛方给的目标图案"（纯色、白底、按模型尺度）
    from icim_sim import target
    target.main(["--out", "runs/target_sim.png"])

所有几何都来自比赛 STEP 模型；相机、板面高度、单应、ROI、抓放姿态都由模型算，
不依赖任何现场示教值（唯一例外是镜头内参，见 docs/LIMITS.md）。
"""
from .calib import measure_calib, SimCalib          # noqa: F401
from .cell import CellFacts, measure as measure_cell  # noqa: F401
from .paths import ASSETS, RUNS, SCENE             # noqa: F401
from .robot import SimCobotClient                  # noqa: F401

__all__ = ["SimCobotClient", "measure_cell", "measure_calib", "SimCalib",
           "CellFacts", "SCENE", "ASSETS", "RUNS"]
__version__ = "0.2.0"
