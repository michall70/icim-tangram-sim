# -*- coding: utf-8 -*-
"""不需要任何队友代码，演示一次完整的"抓一块 → 放到目标盘"。

用途：先确认这个仿真包自己是活的，再去接你们的 main.py。

    python demo.py                 # 抓红色大三角，放到目标盘对应位置
    python demo.py --piece square  # 换一块（large1/large2/medium/small1/small2/square/para）
    python demo.py --viewer        # 开窗口看
    python demo.py --scatter 7     # 先把七块板随机撒开（seed 固定，可复现）

判断成功的标准（脚本自己会算）：放下去之后这块板离目标点的距离。
"""
from __future__ import annotations

import argparse
import sys

import mujoco
import numpy as np

from icim_sim import cell as cellmod
from icim_sim.calib import measure_calib
from icim_sim.robot import SimCobotClient

PIECES = {
    "large1": "piece_large_1",
    "large2": "piece_large_2",
    "medium": "piece_medium",
    "small1": "piece_small_1",
    "small2": "piece_small_2",
    "square": "piece_square",
    "para": "piece_parallelogram",
}
APPROACH_MM = 30.0


class Cup:
    """吸盘的 grip/release 接口（和队友的 RobotDOGripper 形状一致）。"""

    def __init__(self, robot):
        self.robot = robot

    def grip(self):
        return self.robot.suction_grab()

    def release(self):
        self.robot.suction_release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--piece", default="large1", choices=sorted(PIECES))
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--scatter", type=int, default=None, metavar="SEED",
                    help="先把七块板随机撒开")
    ap.add_argument("--yaw", type=float, default=60.0,
                    help="抓放朝向绕竖直轴转多少度（吸盘是圆的，随便挑一个能到的）")
    args = ap.parse_args()

    name = PIECES[args.piece]
    facts = cellmod.measure()
    calib = measure_calib()
    rpy = list(cellmod_import_grasp_yaw(args.yaw))

    robot = SimCobotClient(viewer=args.viewer, layout="auto")
    robot.connect()
    m, d = robot.m, robot.d

    if args.scatter is not None:
        placed = cellmod.scatter(m, d, seed=args.scatter)
        print(f"已随机撒开（seed={args.scatter}）：")
        for k, (x, y, yaw) in placed.items():
            print(f"   {k:<22} ({x:7.1f}, {y:7.1f})  yaw {yaw:+6.1f}°")

    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    start = robot._body_aabb_centre(b) * 1000.0              # 世界系 mm
    start_base = robot.world_to_base_point(start)
    start_top = start[2] + 1.25                              # 板厚 2.5 mm

    # 目标点：把这块板在原料盘上的位置整体平移到目标盘中心
    raw, tgt = calib.raw_board.centre_base, calib.target_board.centre_base
    goal_base = np.array([start_base[0] + (tgt[0] - raw[0]),
                          start_base[1] + (tgt[1] - raw[1])])
    goal_top = start_top                                     # 目标盘同一高度

    z_grip = robot.world_to_base_point([0, 0, start_top])[2] + facts.flange_to_tip_mm
    z_place = z_grip + cellmod.PLACE_CLEARANCE_MM

    print(f"\n抓 {name}")
    print(f"  抓取位（基座）({start_base[0]:7.1f}, {start_base[1]:7.1f}) 法兰 z {z_grip:.1f}")
    print(f"  放置位（基座）({goal_base[0]:7.1f}, {goal_base[1]:7.1f}) 法兰 z {z_place:.1f}")

    cup = Cup(robot)
    # 从"起始姿态"出发（包根目录 start_pose.json，队友可以自己改）
    robot.reset_pose()
    robot.pick([start_base[0], start_base[1], z_grip] + rpy, cup,
               approach_dz=APPROACH_MM)
    if robot.attached != name:
        raise SystemExit(f"没吸住（吸住的是 {robot.attached}），看看吸盘判据")
    robot.place([goal_base[0], goal_base[1], z_place] + rpy, cup,
                approach_dz=APPROACH_MM)
    robot.step(0.5)

    got = robot._body_aabb_centre(b) * 1000.0
    got_base = robot.world_to_base_point(got)
    err = float(np.hypot(got_base[0] - goal_base[0], got_base[1] - goal_base[1]))
    print(f"\n结果：板中心 ({got_base[0]:7.1f}, {got_base[1]:7.1f})，"
          f"离目标 {err:.2f} mm")
    print("OK" if err < 2.0 else "偏差偏大，检查吸盘/高度")
    robot.disconnect()
    return 0 if err < 2.0 else 1


def cellmod_import_grasp_yaw(yaw_deg):
    from icim_sim.calib import grasp_rpy_at

    return grasp_rpy_at(yaw_deg)


if __name__ == "__main__":
    sys.exit(main())
