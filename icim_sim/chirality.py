# -*- coding: utf-8 -*-
"""平行四边形手性：用**队友自己的判据**量两遍，任一说要翻就把实物翻面。

背景
----
七巧板里只有平行四边形"翻过来和没翻"是两种真实状态（镜像不能靠旋转得到）。
他们那边有**两条互相独立**的手性判据，口径不同：

1. `planning/assign.py::handedness_mismatch` —— **像素**空间（检测到的顶点 vs 一个
   y 向上的件库），判不符就抛 `AssignError`，整批规划中止；
2. `puzzle_planner.plan_puzzle` 的 `suspected_flip` —— **毫米**空间（块轮廓 vs
   模板多边形的最小二乘，只允许旋转），`main --execute` 的
   `validate_plan_geometry` 会因此抛 `ValueError`。

两条经常一条说没事、另一条说要翻（实测他们 `data/target_samples` 三张图都是这样），
所以这里两个都算：

    check(env, calib)                      # 第 1 条
    pattern_handedness(env, calib, 目标图)  # 第 2 条（调一次他们的 plan_puzzle，只喂这一块）

判"不符合"时，合法的修法不是改数，而是**把这块板翻个面**：七巧板是等厚棱柱，
翻面后外形正好是镜像，落回托盘仍然是平的 —— 这是现场真能做的动作，
所以仿真里做它不算作弊。`run.py` 把结果塞进 `env.flips`，机器人 `connect()` 时执行。

目前实测：
    · 我们自画的默认目标图：两条都"不用翻"；
    · 他们 `target_samples` 的三张图：第 1 条不用翻、第 2 条要翻 → 会自动翻。
"""
from __future__ import annotations

import numpy as np

#: 只有这块有手性问题
PARALLELOGRAM = "piece_parallelogram"


def _team_assign(env):
    """把他们的 `planning.assign` 模块拿到手（拿不到就返回 None）。"""
    import importlib

    planner = env.team.get("planner") if env.team is not None else None
    names = []
    if planner is not None:
        names.append(planner.__name__.rsplit(".", 1)[0] + ".assign")
    names.append("scenarios.tangram_puzzle.planning.assign")
    for name in names:
        try:
            return importlib.import_module(name)
        except Exception:                              # noqa: BLE001
            continue
    return None


def _pixel_outline(m, d, name, homography):
    """一块板在基座系 xy 上的轮廓 → 去畸变图像像素（和检测器同一口径）。"""
    from . import cell as sim_cell

    hull = sim_cell._hull_xy(m, d, name)
    if len(hull) < 4:
        return None
    h = np.linalg.inv(np.asarray(homography, float))
    pts = np.column_stack([hull, np.ones(len(hull))]) @ h.T
    pts = pts[:, :2] / pts[:, 2, None]
    centre = pts.mean(axis=0)
    order = np.argsort(np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0]))
    return pts[order]


def check(env, calib, *, verbose=True):
    """返回需要翻面的 body 名列表（现在通常是空的）。"""
    import mujoco

    from . import cell as sim_cell

    out = []
    assign = _team_assign(env)
    if assign is None:
        if verbose:
            print("[chirality] 没找到队友的 planning.assign，跳过手性检查")
        return out

    m = mujoco.MjModel.from_xml_path(str(sim_cell.SCENE))
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(m, d, key)
    mujoco.mj_forward(m, d)

    pts = _pixel_outline(m, d, PARALLELOGRAM, calib.homography)
    if pts is None:
        return out
    mismatch, direct, mirror = assign.handedness_mismatch(pts)
    if verbose:
        print(f"[chirality] 平行四边形手性：旋转拟合 {direct:.2f} mm / "
              f"镜像拟合 {mirror:.2f} mm → "
              f"{'不符，将自动翻面' if mismatch else '正常，不用动'}")
    if mismatch:
        out.append(PARALLELOGRAM)
        if verbose:
            print("[chirality] 说明：翻面是把这块板绕水平轴转 180°（等厚棱柱，"
                  "翻过来外形就是镜像），落回托盘仍然是平的")
    return out


def _team_module(env, role, suffix):
    """按角色取队友模块（`planning.target_image` 这类同目录的兄弟模块按名字推）。"""
    import importlib

    base = env.team.get(role) if env.team is not None else None
    names = []
    if base is not None:
        names.append(base.__name__.rsplit(".", 1)[0] + "." + suffix)
    names.append("scenarios.tangram_puzzle.planning." + suffix)
    for name in names:
        try:
            return importlib.import_module(name)
        except Exception:                              # noqa: BLE001
            continue
    return None


def pattern_handedness(env, calib, target_image_path):
    """目标图案里那块平行四边形，和模型里这块**是不是同一只手**。

    做法完全用他们的判据：把物理块的真轮廓造成一个 `TangramPiece`，
    再调一次 `plan_puzzle`（只喂这一块），看它说的 `suspected_flip`。
    这样一旦他们的门槛改了，我们自动跟着改，不会各说各话。

    返回 (needs_flip, fit, mirror_fit)；判断不了时返回 (None, None, None)。
    """
    import mujoco
    import numpy as np

    from . import cell as sim_cell

    planner = _team_module(env, "planner", "puzzle_planner") or env.team.get("planner")
    target_image = _team_module(env, "planner", "target_image")
    detector = env.team.get("detector") if env.team is not None else None
    if planner is None or target_image is None or detector is None:
        return None, None, None
    try:
        placements, _summary = target_image.load_target(str(target_image_path))
        slot = next(i for i, item in enumerate(placements) if item["kind"] == "P")
        pieces = target_image.build_template_pieces(
            placements, float(calib.s_phys_mm),
            [float(calib.target_board.centre_base[0]),
             float(calib.target_board.centre_base[1])],
            0.0, np.asarray(calib.homography, float), piece_ids={slot: 4})
        tpl = next(p for p in pieces if p.shape == "parallelogram")

        m = mujoco.MjModel.from_xml_path(str(sim_cell.SCENE))
        d = mujoco.MjData(m)
        key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key >= 0:
            mujoco.mj_resetDataKeyframe(m, d, key)
        mujoco.mj_forward(m, d)
        hull = sim_cell._hull_xy(m, d, PARALLELOGRAM)
        px = _mm_to_px(hull, calib.homography)
        if px is None:
            return None, None, None
        g = px.mean(axis=0)
        raw = detector.TangramPiece(
            piece_id=4, color_name="P1", shape="parallelogram",
            center=(int(round(g[0])), int(round(g[1]))),
            grasp_point=(int(round(g[0])), int(round(g[1]))),
            vertices=[(float(a), float(b)) for a, b in px],
            angle=0.0, contour=None)
        plans = planner.plan_puzzle([raw], [tpl], calib, calib.grasp_rpy,
                                    calib.grip_z, calib.place_z, 0.0, (0.0, 0.0))
    except Exception as exc:                           # noqa: BLE001
        print(f"[chirality] 目标图案手性没算出来（{type(exc).__name__}: {exc}），跳过")
        return None, None, None
    plan = plans[0]
    return bool(plan.suspected_flip), float(plan.fit_error_mm), float(plan.mirror_fit_error_mm)


def _mm_to_px(points_mm, homography):
    """台面 mm → 去畸变图像像素，并按检测器的规则（绕形心极角升序）排好。"""
    import numpy as np

    if homography is None:
        return None
    inv = np.linalg.inv(np.asarray(homography, float))
    p = np.asarray(points_mm, float).reshape(-1, 2)
    h = np.column_stack([p, np.ones(len(p))]) @ inv.T
    if np.any(np.abs(h[:, 2]) < 1e-9):
        return None
    px = h[:, :2] / h[:, 2, None]
    centre = px.mean(axis=0)
    return px[np.argsort(np.arctan2(px[:, 1] - centre[1], px[:, 0] - centre[0]))]
