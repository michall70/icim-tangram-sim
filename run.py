# -*- coding: utf-8 -*-
"""让队友的 `main.py` 在 MuJoCo 里跑一遍，一行代码都不改。

    python run.py            # 最常用：算完自动回放
    python run.py --dry-run  # 只取图+检测+规划，不动机械臂
    python run.py --help     # 全部参数

参数说明统一维护在 `README.md` §4，**这里不再重复**（以前两处各写一份，改一处漏一处）。

它做四件事：
1. 把三种硬件换成仿真替身（见 `icim_sim/devices.py`）；
2. 把他们的标定值换成由模型算出来的值（见 `icim_sim/calib.py`），
   并把改了什么、改前改后各是多少全部打印出来；
3. 把目标图案通道的三件事喂进去（图 / 锚点 / 可用区 + 实物尺度），
   顺带按他们的判据决定平行四边形要不要翻面（见 `icim_sim/chirality.py`）；
4. 调他们的 `run(execute=True)`，跑完把每块板的落点和 `plans.json` 的 place 点对一遍。

所有图片、录像、结果都落在 `runs/`，不写他们的仓库。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from icim_sim.paths import RUNS

OUT = RUNS


def pick_grasp_yaw(calib, verbose=True):
    """挑一个"手腕真的到得了"的抓放朝向。

    吸盘是圆的，绕竖直轴转多少都吸得住，但机器人能到哪取决于这个角。
    判据：原料堆上方、目标区上方各取一个"抓住/放下"和"上方 30 mm"的点，
    四个点都逆解成功才算这个角可用。取第一个能用的。

    注意：**不能**为了"腕子舒服"去换这个角。他们的放置角是
    `delta = 目标朝向 − 板子朝向` 再作用于法兰的，抓取朝向一变，板子在吸盘里的
    相对朝向就跟着变；实测换成 180° 后落位偏差从 0.5 mm 变成 16.6 mm。
    """
    from icim_sim.calib import grasp_rpy_at
    from icim_sim.robot import SimCobotClient

    probes = []
    for centre, z in ((calib.raw_board.centre_base, calib.grip_z),
                      (calib.target_board.centre_base, calib.place_z)):
        probes.append([centre[0], centre[1], z])
        probes.append([centre[0], centre[1], z + 30.0])

    robot = SimCobotClient()
    robot.connect()
    chosen = None
    try:
        # 从 60° 开始试：这是这套标定下验证过的值（0.5 mm）。换别的角会让板子
        # 绕**自己的原点**转（七块板的建模原点离几何中心最多 57 mm），落位偏差
        # 实测 15-17 mm，所以顺序不能随便改。
        for yaw in [60] + [y for y in range(0, 360, 30) if y != 60]:
            rpy = grasp_rpy_at(yaw)
            ok = True
            for p in probes:
                try:
                    robot._pose_to_joint(list(p) + list(rpy), None, "probe")
                except Exception:                        # noqa: BLE001
                    ok = False
                    break
            if ok:
                chosen = (float(yaw), list(rpy))
                break
    finally:
        robot.disconnect()
    if chosen is None:
        if verbose:
            print("[env] 抓放朝向：没有找到四个探测点都能到的角度，用 0°")
        return 0.0, list(grasp_rpy_at(0))
    if verbose:
        print(f"[env] 抓放朝向：吸盘轴竖直，绕竖直轴选 {chosen[0]:.0f}°"
              f"（rpy {[round(v, 1) for v in chosen[1]]}，四个探测点都逆解成功）")
    return chosen[0], chosen[1]


def apply_overlay(env, *, target=None, source="image", template=None):
    """把标定值换成仿真工位的，返回改动清单。

    2026-11-02：队友的主方案改成"读现场给的彩色图案图片"（`TARGET_SOURCE="image"`），
    所以这里除了标定，还要把**目标通道**的几个量喂进去：目标图路径、图案锚点、
    成品区可用尺寸、实物尺度。他们 `main.py` 是 `from config import ...`，
    所以同一个名字 **config 和 main 两处都要改**，只改 config 不会生效。
    """
    from icim_sim import calib as sim_calib

    calib = sim_calib.measure_calib()
    cfg, main = env.config, env.main
    changed = []

    def setv(mod, name, new):
        old = getattr(mod, name, None)
        setattr(mod, name, new)
        changed.append((name, old, new))

    def setb(name, new):
        """同时改 config 和 main 两处的模块级名字。"""
        setv(cfg, name, new)
        if hasattr(main, name):
            setv(main, name, new)

    setb("HOMOGRAPHY", calib.homography)
    setb("CAPTURE_POSE", list(calib.capture_pose))
    setb("GRIP_Z", calib.grip_z)
    setb("PLACE_Z", calib.place_z)
    # 抓放姿态：吸盘轴竖直（模型），绕竖直轴的转角挑一个能到的
    _yaw, grasp_rpy = pick_grasp_yaw(calib)
    setb("TCP_RPY", grasp_rpy)
    # HOME 关节：他们示教的那个收尾姿态在这个模型里 J2/J4 扭得很厉害，臂会穿过自己。
    # 仿真里没有碰撞检测，穿模只是"看着不对"，但收尾姿态不该这么摆。
    # 改成"回拍照位"：位置正常，而且相机在那儿本来就看得见两块托盘。
    home = sim_calib.load_capture_joints()
    if home is None:
        home = [0.0] * 6
    setb("HOME_JOINT", list(home))

    det = cfg.DETECT_PARAMS
    changed.append(("raw_roi", det.get("raw_roi"), calib.raw_roi))
    changed.append(("template_roi", det.get("template_roi"), calib.template_roi))
    det["raw_roi"] = calib.raw_roi
    det["template_roi"] = calib.template_roi

    # ---- 目标通道 --------------------------------------------------------
    # 三件事一起换，缺一件就错位：
    #   TARGET_ANCHOR_MM  图案形心落在台面哪（基座系 mm）
    #   TARGET_AREA_MM    "放不放得下"的判据（按托盘垫片，不是可见托盘外框）
    #   S_PHYS_MM         图案缩放用的实物尺度（按模型七块面积和开根）
    setb("TARGET_SOURCE", source)
    anchor = [float(calib.target_board.centre_base[0]),
              float(calib.target_board.centre_base[1])]
    setb("TARGET_ANCHOR_MM", anchor)
    setb("TARGET_AREA_MM", (float(calib.usable_mm), float(calib.usable_mm)))
    setb("S_PHYS_MM", float(calib.s_phys_mm))
    # 2026-11-02 版把锚点换成了"成品区四角"（`TARGET_AREA_QUAD_MM`），而且**优先于**
    # `TARGET_ANCHOR_MM`。那四角是他们真机台面上的坐标，放到我们模型里整幅图案会跑偏
    # 150 mm（实测锚点变成 (232,-294)），所以这里换成我们托盘垫片的四角。
    # 旧版没有这个常量，`setb` 会自动跳过。
    half = float(calib.usable_mm) / 2.0
    setb("TARGET_AREA_QUAD_MM",
         [(anchor[0] - half, anchor[1] - half), (anchor[0] + half, anchor[1] - half),
          (anchor[0] + half, anchor[1] + half), (anchor[0] - half, anchor[1] + half)])
    # 朝向：None = 先按 0°，放不下再自动转到与成品区平行。我们的成品区只差 2.2°，
    # 图案又比可用区小得多，所以实际不会触发自动转 —— 图案保持目标图自己的朝向。
    setb("TARGET_YAW_DEG", None)
    if target is not None:
        setb("TARGET_IMAGE_PATH", str(target))
    if template is not None:
        setb("TEMPLATE_PATH", str(template))

    # 结果写进 runs/results，别污染他们的仓库
    setv(main, "RESULTS_DIR", str(OUT / "results"))
    return calib, changed


def show(calib, changed):
    print("\n================ 仿真工位标定（全部由模型算出） ================")
    print(calib.report())
    print("\n相比队友 config 的改动：")
    seen = set()
    for name, old, new in changed:
        if name in seen:          # config 和 main 两处都改了，只报一次
            continue
        seen.add(name)
        if name == "HOMOGRAPHY":
            print(f"  {name:<14} 3×3 矩阵已替换")
        elif isinstance(new, (list, tuple)) and isinstance(old, (list, tuple)) and len(new) == 6:
            d = max(abs(a - b) for a, b in zip(new, old))
            print(f"  {name:<14} {old} -> {new}   (最大分量差 {d:.1f})")
        elif isinstance(new, float):
            print(f"  {name:<14} {old} -> {new:.3f}   (差 {new - old:+.3f})")
        else:
            print(f"  {name:<14} {old} -> {new}")
    print("===============================================================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只做视觉+规划，不执行抓放")
    ap.add_argument("--viewer", action="store_true",
                    help="边跑边看：开 MuJoCo 窗口，按真机速度播放（能不能跑满看机器）")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="配合 --viewer：1 = 真机速度（默认），数字越大越快")
    ap.add_argument("--replay", action=argparse.BooleanOptionalAction, default=True,
                    help="跑完自动回放（默认开；不想看窗口就用 --no-replay，只算不放）")
    ap.add_argument("--replay-speed", type=float, default=1.0,
                    help="回放倍速，1.0 = 实时；播放时也能按 1–9 实时改")
    ap.add_argument("--cam-window", action="store_true",
                    help="回放时另开一个腕相机窗口（多一个 GL 窗口很吃资源，默认不开）")
    ap.add_argument("--replay-file", default=str(OUT / "replay.npz"),
                    help="回放录像存哪（--replay 时同时存下来）")
    ap.add_argument("--target", default=str(OUT / "target_sim.png"),
                    help="目标图案图片（比赛给的那张）。不给就用仿真生成的正方形")
    ap.add_argument("--target-source", default="image", choices=("image", "photo"),
                    help="image = 读彩色图案图片（队友现在的主方案）；"
                         "photo = 回退到'相机实拍模板照片'那条旧链路")
    ap.add_argument("--template", default=str(OUT / "template_sim.png"),
                    help="只在 --target-source photo 时用：相机模板照片")
    ap.add_argument("--team-repo", default=None,
                    help="用哪份队友代码（third_party 下的目录名，或绝对路径）；"
                         "不给就用 active.txt / 唯一的那份")
    ap.add_argument("--list", action="store_true",
                    help="列出 third_party 下的所有快照，然后退出")
    ap.add_argument("--set-active", default=None, metavar="NAME",
                    help="把某份快照设为默认（写 third_party/active.txt），然后退出")
    ap.add_argument("--gripper-time", type=float, default=0.2,
                    help="吸盘动作等待秒数；真机是 2.0，仿真里没必要等")
    ap.add_argument("--layout", default="auto", metavar="NAME",
                    help="初始布局：layouts/ 下的名字；auto=按规则挑；none=保持场景原样")
    ap.add_argument("--list-layouts", action="store_true", help="列出所有布局，然后退出")
    ap.add_argument("--no-lens", action="store_true", help="不给图像加镜头畸变")
    args = ap.parse_args()

    from icim_sim.paths import THIRD_PARTY
    from icim_sim.team import format_candidates, set_active

    if args.list:
        print(format_candidates(THIRD_PARTY))
        return
    if args.list_layouts:
        from icim_sim import layout as _layout
        files = _layout.list_layouts()
        print("layouts/ 下的布局：" if files else "layouts/ 下还没有布局")
        for f in files:
            d = _layout.read(f)
            print(f"  {f.stem:<22} {len(d.get('pieces') or {})} 块板"
                  f"{'，含关节角' if d.get('robot_joint_deg') else ''}")
        return
    if args.set_active:
        p = set_active(THIRD_PARTY, args.set_active)
        print(f"默认快照 -> {args.set_active}（已写入 {p}）")
        return

    from icim_sim.devices import bootstrap_env

    # 第一次跑（或刚解压）时还没有目标图案图：现场生成一张。
    # 注意这是**画图**不是拍照：比赛给的就是一张彩色图案图片，我们照那个样子
    # 用模型量出来的尺度画一张（七块纯色、白底、无描边），见 icim_sim/target.py。
    target_path = Path(args.target)
    if args.target_source == "image" and not target_path.is_file():
        print(f"[run] 还没有 {target_path.name}，先按模型尺度画一张（约 1 秒）")
        from icim_sim import target as target_mod
        target_mod.main(["--out", str(target_path)])

    env = bootstrap_env(simulate_lens=not args.no_lens, viewer=args.viewer,
                        team_repo=args.team_repo,
                        layout=None if args.layout == "none" else args.layout)
    print("\n".join(env.notes))
    calib, changed = apply_overlay(env, target=target_path, source=args.target_source,
                                   template=args.template)
    show(calib, changed)

    # 平行四边形手性：用他们自己的判据量一次，不符就让开局把那块翻面
    from icim_sim import chirality
    env.flips.extend(chirality.check(env, calib))
    if args.target_source == "image" and chirality.PARALLELOGRAM not in env.flips:
        # 还有第二条判据：**目标图案**里那块平行四边形要不要翻。
        # 队友的 assign 判据（像素空间）和 plan_puzzle 的 suspected_flip（毫米空间）
        # 是分开算的；换了目标图以后经常是"assign 说没事、plan_puzzle 说要翻"，
        # 实测 target_samples 的三张图都是这样，不翻就直接被 validate_plan_geometry 拦。
        need, fit, mir = chirality.pattern_handedness(env, calib, target_path)
        if need:
            env.flips.append(chirality.PARALLELOGRAM)
            print(f"[chirality] 目标图案要求这块平行四边形**翻面**"
                  f"（旋转拟合 {fit:.2f} mm / 镜像 {mir:.2f} mm）→ 开局翻面")
        elif need is not None:
            print(f"[chirality] 目标图案与模型同手（拟合 {fit:.2f} / 镜像 {mir:.2f} mm），"
                  f"不用翻")

    cfg = env.config
    cfg.GRIPPER_ACTION_TIME = args.gripper_time
    env.main.GRIPPER_ACTION_TIME = args.gripper_time

    if args.dry_run:
        _dry_run(env, calib, target_path)
        return

    print(">>> 调用队友的 main.run(execute=True)")
    rec = None
    t_wall0 = time.perf_counter()

    def _view_hook(robot):
        """看窗口时按真机速度播放（能不能跑满取决于机器，末尾会报实测倍速）。"""
        robot.set_speed(args.speed)

    if args.replay:
        from icim_sim.replay import Recorder

        rec = Recorder(every=10)         # 每 10 个物理步记一帧 = 20 ms 一帧，正好 50 fps

        def _hook(robot, _rec=rec):
            robot.on_step = _rec
            if args.viewer:              # 边录边看：两个钩子都挂上
                _view_hook(robot)

        env.on_robot = _hook           # 机器人是他们的代码在 run() 里造的，等它造出来再挂
        print("[replay] 正在录制整段搬运过程……")
        if args.viewer:
            print(f"[viewer] 实时播放（{args.speed:g}×）")
    elif args.viewer:
        env.on_robot = _view_hook
        print(f"[viewer] 按 {args.speed:g}× 播放（真机速度 = 1；实际能跑多快看机器）")

    env.main.run(execute=True)
    print("\n[run] 全流程返回")

    # 实际达成倍速：仿真时间 / 墙钟时间。窗口开着时通常会低于设定值。
    if env.robot is not None:
        wall = max(time.perf_counter() - t_wall0, 1e-6)
        sim_t = float(env.robot.d.time)
        if sim_t > 0:
            print(f"[速度] 仿真 {sim_t:.1f} s / 实际 {wall:.1f} s "
                  f"→ 约 {sim_t / wall:.2f}× 真机速度")

    if rec is not None:
        if env.robot is not None:
            env.robot.on_step = None
        path = rec.save(args.replay_file)
        print(f"[replay] 录到 {len(rec)} 帧 / {rec.seconds:.1f} s 仿真时间，已存 {path}")
    report_final(env, calib, target_path)
    if rec is not None:
        rec.play(env.robot.m, speed=args.replay_speed,
                 camera="wrist_cam" if args.cam_window else None)


def _latest_results(env):
    """这次跑出来的结果目录（他们每跑一次建一个时间戳子目录）。"""
    from pathlib import Path

    root = Path(env.main.RESULTS_DIR)
    dirs = [d for d in root.iterdir() if d.is_dir()] if root.is_dir() else []
    return max(dirs, key=lambda d: d.stat().st_mtime) if dirs else root


def _polygon_centroid(points):
    """多边形的形心（面积加权），坐标系不变。"""
    import numpy as np

    p = np.asarray(points, float)
    x, y = p[:, 0], p[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = cross.sum() / 2.0
    if abs(area) < 1e-9:
        return p.mean(axis=0)
    return np.array([((x + np.roll(x, -1)) * cross).sum() / (6.0 * area),
                     ((y + np.roll(y, -1)) * cross).sum() / (6.0 * area)])


def report_final(env, calib, target):
    """跑完以后，把每块板实际落在哪里和**目标图案**对一遍。

    类别按**占地面积**排序取档（两个大三角 / 一个中三角 / 两个小三角 / 正方形 /
    平行四边形）。为什么不用"凸包有几个顶点"：CAD 的件有倒角，而且放下后常有
    1–2° 倾斜，投影出来一律是 5–6 个点的多边形，按顶点数判三角会直接判错。
    七块的面积三档之间差 2% 以上，倾斜只会让它们一起小幅变大，不改变顺序。
    """
    import json
    from pathlib import Path

    import cv2
    import numpy as np

    from icim_sim import calib as sim_calib
    from icim_sim import cell as sim_cell
    from icim_sim.camera import SimCamera

    robot = env.robot
    if robot is None or robot.d is None:
        return None
    facts = env.facts
    res = _latest_results(env)
    layout_path = Path(str(target)).with_name(Path(str(target)).stem + "_layout.json")

    # 目标点优先取**计划自己下发的 place 点**（plans.json）。比拿目标图重算更靠谱：
    # 一是这才是"摆放 vs 规划"的直接对照，二是避开"目标图像素坐标 y 向下当 mm 用"
    # 带来的上下镜像（那件事优先级低，见 docs/对接问题清单_1102.md §4）。
    plan_path = res / "plans.json"
    if plan_path.is_file():
        plans = json.loads(plan_path.read_text(encoding="utf-8"))["plans"]
        slots = {}
        for p in plans:
            kind = str(p["color_name"])[0]
            slots.setdefault(kind, []).append(
                (np.asarray(p["place_pose"][:2], float), str(p["color_name"])))
        source = f"{plan_path.parent.name}/plans.json"
    elif layout_path.is_file():
        meta = json.loads(layout_path.read_text(encoding="utf-8"))
        anchor = np.asarray(getattr(env.main, "TARGET_ANCHOR_MM", None)
                            or calib.target_board.centre_base, float)
        slots = {}
        for item in meta["pieces"]:
            poly = np.asarray(item["polygon"], float)
            slots.setdefault(item["kind"], []).append(
                (poly.mean(axis=0) + anchor[:2], item["kind"] + str(item["index"])))
        source = layout_path.name
    else:
        print("\n[report] 既没有 plans.json 也没有图案布局，跳过落点比对")
        return None

    items = []
    for nm in facts.tiles:
        hull = sim_cell._hull_xy(robot.m, robot.d, nm)
        if len(hull) < 3:
            continue
        items.append((nm, abs(float(cv2.contourArea(
            np.asarray(hull, np.float32).reshape(-1, 1, 2)))), hull))
    items.sort(key=lambda item: -item[1])
    if len(items) != 7:
        print(f"\n[report] 只量到 {len(items)} 块板，跳过落点比对")
        return None
    got = {}
    for nm, _area, _hull in items[:2]:
        got[nm] = "L"
    for nm, _area, _hull in items[5:]:
        got[nm] = "T"
    for nm, kind in zip([i[0] for i in items[2:5]], ("P", "Q", "M")):
        got[nm] = kind

    print("\n================ 最终摆放 vs 计划下发的落点 ================")
    print(f"  目标来源 {source}；对照的是**计划里的 place 点**"
          f"（块形心，不是包围盒中心）")
    worst = 0.0
    unused = {k: list(v) for k, v in slots.items()}
    for nm, area, hull in items:
        kind = got[nm]
        centre = _polygon_centroid(np.asarray(hull, float))
        pool = unused.get(kind) or []
        if not pool:
            print(f"  {nm:<22} {kind}  计划里没有同类落点，跳过")
            continue
        idx = min(range(len(pool)),
                  key=lambda i: float(np.linalg.norm(pool[i][0] - centre)))
        want, label = pool.pop(idx)
        err = float(np.linalg.norm(centre - np.asarray(want)))
        worst = max(worst, err)
        print(f"  {nm:<22} {label:<3} 面积 {area:7.1f} mm²  "
              f"实际形心 ({centre[0]:7.1f},{centre[1]:7.1f})  "
              f"计划 ({want[0]:7.1f},{want[1]:7.1f})  偏差 {err:5.2f} mm")
    print(f"  最大偏差 {worst:.2f} mm")

    # 成品图：一定要**先回拍照位、再拍**，而且要和他们相机走同一条链路
    # （加镜头畸变）。2026-11-02 实测：以前直接 move_to 拍，拍出来的托盘比
    # 他们的 raw_frame.png 偏 20–50 px、还多转 4.6°——因为收工时臂停在
    # HOME（= 拍照位关节），这次 move_to 几乎是零位移，姿态没真正落到目标上。
    # 现在先 move_home 再 move_to，并把实际位姿打出来对账。
    robot.move_home(env.config.HOME_JOINT)
    robot.move_to(calib.capture_pose)
    pose = robot.current_pose()
    err_pos = max(abs(a - b) for a, b in zip(pose[:3], calib.capture_pose[:3]))
    err_ang = max(abs(((a - b + 180.0) % 360.0) - 180.0)
                  for a, b in zip(pose[3:], calib.capture_pose[3:]))
    print(f"  成品图拍照位：法兰 {[round(v, 2) for v in pose]}"
          f"（与目标差 {err_pos:.3f} mm / {err_ang:.3f}°）")
    cam = SimCamera(camera_matrix=env.config.CAMERA_MATRIX,
                    dist_coeffs=env.config.DIST_COEFFS, simulate_lens=True)
    cam.open()
    img = cam.grab_color()
    out = OUT / "after.png"
    cv2.imwrite(str(out), img)
    print(f"  成品图已存 {out}  （结果目录 {res}，含镜头畸变，可直接和 "
          f"raw_frame.png 对比）")
    return worst


def _dry_run(env, calib, target):
    """不动机械臂：走到拍照位拍一张，然后**用他们自己的离线流程**跑一遍。

    以前这里是我们把检测/规划复刻一遍，两边容易走偏（他们的判据一改，复刻就过时）。
    现在只做一件事：把"相机原帧"存成文件，再调 `main.run(raw_image_path=...)` —
    这就是他们平时调试跑的那条离线分支，检测、目标图解析、规划、几何门禁
    全是他们的代码。
    """
    import json
    import shutil

    import cv2

    from icim_sim.camera import SimCamera, set_active_robot
    from icim_sim.robot import SimCobotClient

    cfg = env.config
    out = OUT / "dry"
    out.mkdir(parents=True, exist_ok=True)

    robot = SimCobotClient(layout=None if env.layout == "none" else env.layout,
                           flips=env.flips)
    set_active_robot(robot)
    robot.connect()
    robot.move_home(cfg.HOME_JOINT)
    robot.move_to(calib.capture_pose)
    print(f"[dry] 拍照位到位，法兰 {[round(v, 1) for v in robot.current_pose()]}")

    cam = SimCamera(camera_matrix=cfg.CAMERA_MATRIX, dist_coeffs=cfg.DIST_COEFFS,
                    simulate_lens=True)
    cam.open()
    raw = cam.grab_color()
    raw_path = out / "raw.png"
    cv2.imwrite(str(raw_path), raw)
    print(f"[dry] 相机原帧（未去畸变）-> {raw_path}")
    cam.close()
    robot.disconnect()

    print(">>> 调用队友的离线分支 main.run(raw_image_path=..., target_source='image')")
    env.main.run(raw_image_path=str(raw_path), target_source="image",
                 target_image_path=str(target))

    res = _latest_results(env)
    for name in ("raw_annotated.png", "raw_frame.png", "plan_preview.png",
                 "target_summary.json", "plans.json", "plans.txt"):
        src = res / name
        if src.exists():
            shutil.copy2(src, out / name)
    plans_path = out / "plans.json"
    if plans_path.is_file():
        plans = json.loads(plans_path.read_text(encoding="utf-8"))["plans"]
        print(f"\n[dry] {len(plans)} 条计划（目标图 = {target}）：")
        for p in plans:
            g, q = p["grasp_pose"], p["place_pose"]
            print(f"   ID{p['piece_id']} {p['color_name']:<4} "
                  f"抓 ({g[0]:7.1f},{g[1]:7.1f}) 放 ({q[0]:7.1f},{q[1]:7.1f}"
                  f",{q[5]:6.1f}°) 转 {p['delta_angle']:+6.1f}° "
                  f"RMS {p['fit_error_mm']:5.2f} mm"
                  f"{'  疑似翻面' if p.get('suspected_flip') else ''}"
                  f"{'  几何不符' if p.get('geometry_mismatch') else ''}")
    print(f"\n[dry] 中间图都在 {out}")
    print("      要看的：raw_annotated.png（原料七块标注）、plan_preview.png"
          "（黄=预测落位 / 绿=目标槽位）")


if __name__ == "__main__":
    main()
