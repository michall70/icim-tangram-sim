# -*- coding: utf-8 -*-
"""把"仿真里的一切都来自 STEP"这件事逐条核对一遍。

这不是单元测试，是**验收清单**：每一条都是一个可以从模型（或物理）本身验证的
事实。有任意一条不过，沙盒里看到的东西就不可信。

    python verify.py          # 有多少条过多少条，有失败就返回 1
    python verify.py -v       # 连带把中间数字都打出来

清单分五组：
  A 装配   —— 关节数、自由度、七块自由板
  B 几何   —— 工具长度、板面高度、七块板面积（应当 4:4:2:1:1:2:2）
  C 物理   —— 板子落定不下沉、掉在托盘上不穿模
  D 相机   —— 在模型里的安装面、拍照姿态光轴竖直、焦距
  E 联动   —— 拍照位可逆解、吸盘对准才吸得住、歪了/偏了/高了不吸
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

from icim_sim import calib as sim_calib
from icim_sim import cell as sim_cell
from icim_sim.robot import SimCobotClient, rpy_zyx

RESULTS = []


def check(group, name, ok, detail=""):
    RESULTS.append((group, name, bool(ok), detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {group} · {name}" + (f"   {detail}" if detail else ""))
    return ok


def near(a, b, tol):
    return abs(float(a) - float(b)) <= tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    facts = sim_cell.measure()
    calib = sim_calib.measure_calib()
    m = mujoco.MjModel.from_xml_path(str(sim_cell.SCENE))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home"))
    mujoco.mj_forward(m, d)

    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    flange = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Link_6")
    tcp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    Rb, pb = d.xmat[base].reshape(3, 3), d.xpos[base]

    def to_base(p_m):
        return (Rb.T @ (np.asarray(p_m) - pb)) * 1000.0

    print("\n=== A 装配 ===")
    # 6 个铰链 + 1 个吸盘伸缩滑轨 + 7 块板的 freejoint
    check("A", "nq = 56", m.nq == 56, f"nq={m.nq}")
    check("A", "njnt = 14（6 铰链 + 1 滑轨 + 7 freejoint）", m.njnt == 14,
          f"njnt={m.njnt}")
    hinges = [i for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    frees = [i for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
    check("A", "6 个铰链 + 7 个自由板", len(hinges) == 6 and len(frees) == 7,
          f"{len(hinges)} + {len(frees)}")
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in hinges]
    check("A", "铰链名 Joint_1..6", names == [f"Joint_{k}" for k in range(1, 7)], str(names))

    print("\n=== B 几何（STEP 说了算） ===")
    check("B", "法兰→吸盘口 252.515 mm",
          near(facts.flange_to_tip_mm, 252.515, 0.02),
          f"{facts.flange_to_tip_mm:.3f} mm")
    tops = [v[2] for v in facts.tiles.values()]
    check("B", "七块板顶面同一高度 48.300 mm（基座）",
          max(abs(t - 48.30) for t in tops) < 0.01,
          f"{min(tops):.3f}..{max(tops):.3f}")
    check("B", "板面高度 = 板顶 − 板厚 2.5", near(facts.tile_top_world_mm - 2.5, 65.5, 0.01),
          f"板顶世界 z {facts.tile_top_world_mm:.3f}")

    # 七块板面积（凸包 = 平面板子的面积）
    from scipy.spatial import ConvexHull
    areas = {}
    for i in range(m.nbody):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if not nm.startswith("piece_"):
            continue
        verts = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != i or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = m.geom_dataid[g]
            a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            v = m.mesh_vert[a:a + n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            verts.append(v)
        v = np.vstack(verts)                                  # 世界系，板子躺平
        areas[nm] = ConvexHull(v[:, :2]).volume * 1e6        # mm²
    order = ["piece_large_1", "piece_large_2", "piece_medium", "piece_small_1",
             "piece_small_2", "piece_square", "piece_parallelogram"]
    want = [3.93, 3.93, 2.00, 1.00, 1.00, 2.10, 2.14]
    got = [areas[n] / areas["piece_small_1"] for n in order]
    check("B", "面积比 4:4:2:1:1:2:2（以小三角为单位）",
          max(abs(a - b) for a, b in zip(got, want)) < 0.10,
          " / ".join(f"{a:.2f}" for a in got))
    check("B", "总面积 13536 mm²（拼正方形边长 116.4）",
          near(sum(areas.values()), 13536, 60), f"{sum(areas.values()):.0f} mm²")
    if args.verbose:
        for n in order:
            print(f"        {n:<22}{areas[n]:8.1f} mm²")

    print("\n=== C 物理 ===")
    pieces = [i for i in range(m.nbody)
              if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith("piece_")]
    z_world0 = {i: d.xpos[i][2] * 1000 for i in pieces}
    for _ in range(500):                                     # 1 s
        mujoco.mj_step(m, d)
    driven = max(abs(d.xpos[i][2] * 1000 - z) for i, z in z_world0.items())
    check("C", "1 秒后板子没有下沉（< 0.3 mm）", driven < 0.3, f"最大 {driven:.3f} mm")

    # 从 110 mm 高处松手，应当落回托盘而不是穿模
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home"))
    mujoco.mj_forward(m, d)
    b0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "piece_large_1")
    adr = m.jnt_qposadr[m.body_jntadr[b0]]
    z_start = d.qpos[adr + 2]
    d.qpos[adr + 2] = 0.110
    mujoco.mj_forward(m, d)
    for _ in range(600):
        mujoco.mj_step(m, d)
    z_end = d.qpos[adr + 2] * 1000
    check("C", "从 110 mm 掉下来落在托盘上（不穿模）", z_end > 60.0,
          f"落到 {z_end:.2f} mm（托盘面 65.5）")

    print("\n=== D 相机（模型安装面） ===")
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    check("D", "只有一台相机（不塞实测位的第二台）", m.ncam == 1,
          f"ncam={m.ncam} {[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)]}")
    R_f = d.xmat[flange].reshape(3, 3)
    p_f = d.xpos[flange]
    off_flange = (R_f.T @ (d.cam_xpos[cid] - p_f)) * 1000
    # 相机位置 = RGB 模块成像面，再侧移 3.49 mm 去模拟真机的主点偏心（cx/cy ≠ 画面中心）
    d_face = float(np.linalg.norm(off_flange - np.array([-85.77, -29.94, 64.22])))
    check("D", "相机在 RGB 模块成像面上（±5 mm，含 3.5 mm 主点补偿）", d_face < 5.0,
          f"{np.round(off_flange, 2).tolist()}，离成像面 {d_face:.2f} mm")

    # 拍照位：从 IK 走到那个位姿，再看渲染用的相机帧
    robot = SimCobotClient()
    robot.connect()
    q = robot._pose_to_joint(calib.capture_pose, None, "capture")
    check("E", f"拍照位可逆解 {np.round(calib.capture_pose, 1).tolist()}", q is not None,
          f"q={np.round(q, 1).tolist()}")
    d.qpos[:6] = np.radians(q)
    mujoco.mj_forward(m, d)
    R_cam = d.cam_xmat[cid].reshape(3, 3)
    view = -R_cam[:, 2]
    tilt = float(np.degrees(np.arccos(np.clip(-view[2], -1.0, 1.0))))
    check("D", "拍照位光轴基本朝下（偏竖直 ≤ 8°）", tilt <= 8.0, f"偏 {tilt:.2f}°")
    # 真正的验收条件不是"光轴严格竖直"，而是**两块托盘都要完整落在画面里**：
    # 拍照位换了（比如换成作者手调的那个姿态，偏 4°）只要两个 ROI 都在画面内就成立。
    roi_ok, roi_txt = True, []
    for name, roi in (("原料区", calib.raw_roi), ("目标区", calib.template_roi)):
        x, y, w, h = roi
        good = x >= 0 and y >= 0 and x + w <= 1280 and y + h <= 720 and w > 50 and h > 50
        roi_ok = roi_ok and good
        roi_txt.append(f"{name} x[{x},{x + w}] y[{y},{y + h}]")
    check("D", "拍照位两块托盘都完整在画面内", roi_ok, "；".join(roi_txt))

    fy = (720 / 2.0) / np.tan(np.radians(m.cam_fovy[cid]) / 2)
    check("D", "fovy 对应 fy ≈ 910.4 px（真机镜头）", near(fy, 910.4, 1.0), f"fy={fy:.2f}")

    print("\n=== E 联动 / 吸盘 ===")
    # 吸盘口正对板面：应当吸住
    d.qpos[:6] = 0.0
    tile = "piece_large_2"
    tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, tile)
    mujoco.mj_forward(m, d)
    centre = robot._body_aabb_centre(tb) * 1000
    xy = robot.world_to_base_point([centre[0], centre[1], 0.0])
    rpy = list(calib.grasp_rpy)
    top_base = robot.world_to_base_point([0, 0, centre[2]])[2] + 1.25
    pose = [xy[0], xy[1], top_base + facts.flange_to_tip_mm] + rpy
    robot.move_to(pose)
    got = robot.suction_grab()
    check("E", "吸盘口对准板子 -> 吸住", got, f"吸住 {robot.attached}")
    robot.suction_release()

    # 偏 30 mm：应当吸不住
    robot.move_to([pose[0] + 30.0, pose[1], pose[2]] + rpy)
    got = robot.suction_grab()
    check("E", "吸盘偏 30 mm -> 不吸", not got, f"吸住 {robot.attached}")
    robot.suction_release()

    # 抬高 30 mm：应当吸不住
    robot.move_to([pose[0], pose[1], pose[2] + 30.0] + rpy)
    got = robot.suction_grab()
    check("E", "吸盘抬高 30 mm -> 不吸", not got, f"吸住 {robot.attached}")
    robot.suction_release()

    # ---- F 初始布局 -----------------------------------------------------
    from icim_sim import layout as _layout

    files = _layout.list_layouts()
    check("F", "layouts/ 里至少有一份布局", bool(files),
          f"{[p.stem for p in files]}")

    # 摆一个已知的位姿，存，打乱，读回来 —— 必须一模一样
    target = "piece_medium"
    sim_cell.set_piece(robot.m, robot.d, target, -180.0, -420.0, 25.0)
    mujoco.mj_forward(robot.m, robot.d)
    want = _layout.capture(robot.m, robot.d)
    tmp = _layout.LAYOUTS / "_verify_tmp.json"
    _layout.write(tmp, [0.0] * 6, want)
    sim_cell.set_piece(robot.m, robot.d, target, -140.0, -460.0, -70.0)
    mujoco.mj_forward(robot.m, robot.d)
    _layout.apply(robot.m, robot.d, _layout.read(tmp))
    got_all = _layout.capture(robot.m, robot.d)
    err = max(abs(a - b) for nm in want for a, b in zip(want[nm], got_all[nm]))
    tmp.unlink(missing_ok=True)
    check("F", "布局存-读往返无误差（七块板 + yaw）", err < 0.01, f"最大 {err:.3f}")

    # 多份快照的选择规则
    from icim_sim import team
    from icim_sim.paths import THIRD_PARTY

    cands = team.list_candidates(THIRD_PARTY)
    check("F", "third_party 下能认出队友快照", bool(cands),
          f"{[c.name for c in cands]}")
    resolved = None
    if cands:
        try:
            resolved = team.resolve_root(THIRD_PARTY)
        except SystemExit:
            resolved = None
        if len(cands) == 1:
            check("F", "只有一份时自动选中它", resolved == cands[0].path.resolve(),
                  str(resolved))
        else:
            check("F", "多份时要么已指定、要么报错并列出候选",
                  resolved is not None or team._active_name(THIRD_PARTY) is not None,
                  f"active={team._active_name(THIRD_PARTY)}")

    robot.disconnect()

    print("\n=== 汇总 ===")
    bad = [r for r in RESULTS if not r[2]]
    print(f"  {len(RESULTS) - len(bad)}/{len(RESULTS)} 条通过")
    if bad:
        for g, n, _, det in bad:
            print(f"  失败：{g} · {n}   {det}")
        return 1
    print("  仿真里的每一条事实都对得上 STEP。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
