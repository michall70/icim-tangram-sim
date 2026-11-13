# -*- coding: utf-8 -*-
"""（旧链路）生成"相机模板照片"——只在 `--target-source photo` 时用。

2026-11-02 起比赛走的是**彩色图案图片**通道（`TARGET_SOURCE="image"`），
那份图由 `icim_sim/target.py` 直接画（纯色、白底、无描边），**不再需要拍照片**。
这里保留的是旧的 photo 通道：把七块板整体搬到目标托盘上、绕托盘中心转一个角度，
从拍照位拍一张，当成"相机实拍模板"。万一现场给的图解析不了，可以用它回退。

    用法：`python run.py --target-source photo`（配合 `--template` 指定路径）

做法：把七块板整体搬到目标托盘上、绕托盘中心转一个角度，从拍照位拍一张，
存成 PNG。板与板之间的相对位置保持不变（它们本来就不重叠），所以搬过去仍然
是一个合法的图案；整体转动只是为了让规划层必须算旋转，而不是纯平移。

    python -m icim_sim.template                 # 默认转 15°，存 runs/
    python -m icim_sim.template --angle 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from icim_sim import calib as sim_calib
from icim_sim import cell as sim_cell
from icim_sim.camera import SimCamera, set_active_robot
from icim_sim.paths import RUNS

OUT = RUNS


def tile_qpos_adr(m, name):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    if b < 0:
        raise KeyError(name)
    j = m.body_jntadr[b]
    return m.jnt_qposadr[j], m.jnt_dofadr[j]


def move_tiles(m, d, names, src_centre, dst_centre, angle_deg):
    """整体刚体搬运：绕 src_centre 转 angle，再平移到 dst_centre。

    src/dst 都是**基座系 mm**（和七巧板中心的记录口径一致），qpos 是世界系，
    所以先换到基座系算，再换回来。
    """
    a = np.radians(angle_deg)
    Rz = np.array([[np.cos(a), -np.sin(a), 0.0],
                   [np.sin(a), np.cos(a), 0.0],
                   [0.0, 0.0, 1.0]])
    qz = np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])
    src = np.array([src_centre[0], src_centre[1], 0.0]) / 1000.0
    dst = np.array([dst_centre[0], dst_centre[1], 0.0]) / 1000.0

    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    Rb = d.xmat[base].reshape(3, 3).copy()
    pb = d.xpos[base].copy()

    for nm in names:
        adr, dof = tile_qpos_adr(m, nm)
        p = d.qpos[adr:adr + 3].copy()
        q = d.qpos[adr + 3:adr + 7].copy()
        p_base = Rb.T @ (p - pb)
        rel = p_base - src
        rel[2] = 0.0
        p_base_new = dst + Rz @ rel
        p_base_new[2] = p_base[2]
        p_new = Rb @ p_base_new + pb
        w1, x1, y1, z1 = qz
        w2, x2, y2, z2 = q
        q_new = np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                          w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                          w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                          w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
        d.qpos[adr:adr + 3] = p_new
        d.qpos[adr + 3:adr + 7] = q_new / np.linalg.norm(q_new)
        d.qvel[dof:dof + 6] = 0.0


def spin_tiles(m, d, spins, centres):
    """把某几块绕自己的**几何中心**转一下（只对 90°/180° 对称的块用）。

    必须绕中心转：CAD 里每块板的自身原点不一定在中心，绕原点转会让轮廓平移，
    图案就重叠了。
    """
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    Rb = d.xmat[base].reshape(3, 3).copy()
    pb = d.xpos[base].copy()
    for nm, deg in spins.items():
        a = np.radians(deg)
        Rz = np.array([[np.cos(a), -np.sin(a), 0.0],
                       [np.sin(a), np.cos(a), 0.0],
                       [0.0, 0.0, 1.0]])
        qz = np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])
        adr, dof = tile_qpos_adr(m, nm)
        p = d.qpos[adr:adr + 3].copy()
        q = d.qpos[adr + 3:adr + 7].copy()
        p_base = Rb.T @ (p - pb)
        c = np.array([centres[nm][0], centres[nm][1], p_base[2]]) / 1000.0
        p_base_new = c + Rz @ (p_base - c)
        p_new = Rb @ p_base_new + pb
        w1, x1, y1, z1 = qz
        w2, x2, y2, z2 = q
        q_new = np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                          w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                          w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                          w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
        d.qpos[adr:adr + 3] = p_new
        d.qpos[adr + 3:adr + 7] = q_new / np.linalg.norm(q_new)
        d.qvel[dof:dof + 6] = 0.0
        print(f"  {nm} 绕几何中心自转 {deg:+.0f}°")


def main(argv=None):
    """生成仿真模板图。

    `argv` 传 `[]` 时按默认参数跑——`run.py` 内部调用必须这样，
    否则 argparse 会把 `run.py` 自己的参数（如 `--no-replay`）当成自己的，
    直接报错退出（解压后第一次跑就会踩到）。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--angle", type=float, default=15.0,
                    help="七块板整体转多少度（15° 左右既能测旋转又不超出托盘）")
    ap.add_argument("--spin", action="store_true", default=True,
                    help="正方形转 90°、平行四边形转 180°（轮廓不变，但规划要算转角）")
    ap.add_argument("--no-spin", dest="spin", action="store_false")
    ap.add_argument("--out", default=str(OUT / "template_sim.png"))
    args = ap.parse_args(argv)

    from icim_sim.robot import SimCobotClient
    from icim_sim.devices import load_config

    calib = sim_calib.measure_calib()
    facts = sim_cell.measure()
    cfg = load_config()

    # 注意：模板必须在**板子在 CAD 位姿**的状态下生成——`move_tiles` 的 src 就是
    # CAD 中心，如果这里套用了初始布局（板子已经被挪过），图案会整体偏掉，
    # 之后规划出来的目标位会跟着偏十几毫米。所以这里明确不套布局。
    robot = SimCobotClient(layout=None)
    set_active_robot(robot)
    robot.connect()
    robot.move_to(calib.capture_pose)

    names = list(facts.tiles)
    src = np.mean([facts.tiles[n][:2] for n in names], axis=0)
    dst = np.asarray(calib.target_board.centre_base, float)
    move_tiles(robot.m, robot.d, names, src, dst, args.angle)
    if args.spin:
        # 只用"轮廓不变"的自转：正方形 90°、平行四边形 180°，
        # 这样图案依然是合法摆放，但规划层必须真的去算转角。
        mujoco.mj_forward(robot.m, robot.d)
        centres = {n: sim_calib.body_aabb_base(robot.m, robot.d, n)[:2]
                   for n in names}
        spin_tiles(robot.m, robot.d,
                   {"piece_square": 90.0, "piece_parallelogram": 180.0}, centres)
    mujoco.mj_forward(robot.m, robot.d)

    cam = SimCamera(camera="wrist_cam", camera_matrix=cfg.CAMERA_MATRIX,
                    dist_coeffs=cfg.DIST_COEFFS, simulate_lens=True)
    cam.open()
    img = cam.grab_color()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    import cv2

    cv2.imwrite(args.out, img)
    print(f"目标图案：七块板整体从 {np.round(src,1).tolist()} 搬到 "
          f"{np.round(dst,1).tolist()}，转 {args.angle:+.1f}°")
    print(f"已存 {args.out}")

    # 记下这套摆放参数，报告最终结果时要用同一套来对照
    meta = {
        "angle_deg": float(args.angle),
        "spin": {"piece_square": 90.0, "piece_parallelogram": 180.0} if args.spin else {},
        "src_base_mm": [float(v) for v in src],
        "dst_base_mm": [float(v) for v in dst],
        "image": str(args.out),
    }
    meta_path = Path(args.out).with_name("template_arrangement.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    print(f"摆放参数已存 {meta_path}")
    robot.disconnect()


if __name__ == "__main__":
    main()
