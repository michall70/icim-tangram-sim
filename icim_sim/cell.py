# -*- coding: utf-8 -*-
"""从 MuJoCo 场景里"量"出上层代码需要的数字。

原则：**模型是唯一真值**。抓取高度不抄任何示教记录，而是由

    法兰在基座系的高度 = 七巧板上表面高度 + 法兰到吸盘口的长度

算出来。两块数据都来自 `assets/scene.xml`：

- 七巧板上表面：七块板的网格顶点 AABB 顶面，`qpos` 取 `home` 关键帧；
- 法兰到吸盘口：`<site name="tcp">` 相对 `Link_6` 的位移（CAD 量出来的 252.515 mm）。

这样只要模型改一个字，抓取高度自动跟着变，不会出现"配置和模型各说各话"。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

from .paths import SCENE

#: 放置时吸盘口离板面再高一点，避免一放下就撞。和队友 config 里
#: `PLACE_Z - GRIP_Z = 2` 的口径一致，但基准改成模型。
#: 2026-11-02：改成 0。那 2 mm 会变成"松手 → 掉 2 mm → 落到邻居边上被顶开"，
#: 实测有 3 块因此偏 6–8 mm。七巧板是平的、托盘不参与碰撞，贴面松手就够了。
PLACE_CLEARANCE_MM = 0.0


@dataclass(frozen=True)
class CellFacts:
    scene: str
    tile_top_world_mm: float
    tile_top_base_mm: float
    flange_to_tip_mm: float
    grip_z_mm: float
    place_z_mm: float
    home_tcp_base_mm: tuple
    tiles: dict = field(default_factory=dict)     # 名字 -> 基座系 (x, y, z_top)

    def report(self) -> str:
        lines = [
            f"场景            {self.scene}",
            f"七巧板上表面    世界 z = {self.tile_top_world_mm:8.3f} mm   "
            f"基座 z = {self.tile_top_base_mm:8.3f} mm",
            f"法兰→吸盘口     {self.flange_to_tip_mm:8.3f} mm（沿工具 +Z，来自 tcp site）",
            f"抓取位法兰 z    基座 {self.grip_z_mm:8.3f} mm",
            f"放置位法兰 z    基座 {self.place_z_mm:8.3f} mm",
            f"home 时吸盘口   基座 {np.round(self.home_tcp_base_mm, 3).tolist()} mm"
            f"（离板面 {self.home_tcp_base_mm[2] - self.tile_top_base_mm:+.3f} mm）",
            "七巧板中心（基座系 mm）：",
        ]
        for name, (x, y, z) in sorted(self.tiles.items()):
            lines.append(f"  {name:<22} x {x:8.2f}  y {y:8.2f}  上表面 z {z:8.3f}")
        return "\n".join(lines)


def _body_aabb(m, d, b):
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != b:
            continue
        R = d.geom_xmat[g].reshape(3, 3)
        p = d.geom_xpos[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            adr, num = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            v = m.mesh_vert[adr:adr + num]
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            s = m.geom_size[g]
            v = np.array([[sx, sy, sz] for sx in (-s[0], s[0])
                          for sy in (-s[1], s[1]) for sz in (-s[2], s[2])])
        else:
            continue
        w = v @ R.T + p
        lo = np.minimum(lo, w.min(axis=0))
        hi = np.maximum(hi, w.max(axis=0))
    return lo, hi


@lru_cache(maxsize=4)
def measure(scene: str | Path = SCENE) -> CellFacts:
    """加载场景并测出上面那些数字（结果缓存）。"""
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(m, d, key)
    mujoco.mj_forward(m, d)

    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    flange = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Link_6")
    tcp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    if min(base, flange, tcp) < 0:
        raise RuntimeError("场景缺少 base_link / Link_6 / tcp")

    Rb = d.xmat[base].reshape(3, 3)
    pb = d.xpos[base].copy()
    p_f = d.xpos[flange].copy()
    R_f = d.xmat[flange].reshape(3, 3).copy()
    p_t = d.site_xpos[tcp].copy()
    # 法兰→吸盘口：取**几何长度**（两个点的距离），不按关节轴取符号。
    # 吸盘带伸缩段时，"压缩方向"和"工具朝下的方向"正好相反，用关节轴定方向会一会儿
    # 正一会儿负；而下游只把它当"法兰要高多少才能让杯口贴到板面"，长度就够。
    tip = float(np.linalg.norm(p_t - p_f) * 1000.0)

    tiles, tops = {}, []
    for i in range(m.nbody):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if not nm.startswith("piece_"):
            continue
        lo, hi = _body_aabb(m, d, i)
        tops.append(hi[2] * 1000.0)
        cx, cy = ((lo + hi) / 2)[:2] * 1000.0
        c = Rb.T @ (np.array([cx, cy, hi[2] * 1000.0]) / 1000.0 - pb) * 1000.0
        tiles[nm] = (float(c[0]), float(c[1]), float(c[2]))

    top_w = float(np.mean(tops))
    top_b = float((Rb.T @ (np.array([0, 0, top_w / 1000.0]) - pb))[2] * 1000.0)
    grip = top_b + tip
    home_tcp = Rb.T @ (p_t - pb) * 1000.0

    return CellFacts(
        scene=str(scene),
        tile_top_world_mm=top_w,
        tile_top_base_mm=top_b,
        flange_to_tip_mm=tip,
        grip_z_mm=grip,
        place_z_mm=grip + PLACE_CLEARANCE_MM,
        home_tcp_base_mm=tuple(float(v) for v in home_tcp),
        tiles=tiles,
    )


if __name__ == "__main__":
    print(measure().report())


# ===========================================================================
# 摆位：任意调整七块板的初始位姿
# ===========================================================================
# 比赛要求"任意调整七巧板的初始位姿"，所以这里给三件事：
#   set_piece(m, d, name, x, y, yaw)  —— 把一块板摆到基座系的 (x, y)，绕竖直轴转 yaw
#   scatter(m, d, seed=...)           —— 在原料盘上随机撒开，保证互不重叠
#   overlaps(m, d)                    —— 检查有没有两块板压在一起
# 位置用**基座系 mm**，和程序里其它位姿同一个口径。yaw 是相对 CAD 原始朝向的增量。

TILE_NAMES = ("piece_large_1", "piece_large_2", "piece_medium", "piece_small_1",
              "piece_small_2", "piece_square", "piece_parallelogram")


def _base_frame(m, d):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    return d.xmat[b].reshape(3, 3).copy(), d.xpos[b].copy()


def _tile_joint(m, name):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    if b < 0:
        raise KeyError(f"场景里没有 {name}（可用的：{', '.join(TILE_NAMES)}）")
    j = m.body_jntadr[b]
    return (m.jnt_qposadr[j], m.jnt_dofadr[j], b)


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def cad_poses(m, d):
    """关键帧里每块板的原始位姿（世界系）。摆位都以它为基准，便于"再摆一次"。"""
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key < 0:
        raise RuntimeError('场景里没有 <keyframe name="home">')
    saved = d.qpos.copy()
    mujoco.mj_resetDataKeyframe(m, d, key)
    out = {}
    for nm in TILE_NAMES:
        adr, _, _ = _tile_joint(m, nm)
        out[nm] = (d.qpos[adr:adr + 3].copy(), d.qpos[adr + 3:adr + 7].copy())
    d.qpos[:] = saved
    mujoco.mj_forward(m, d)
    return out


def set_piece(m, d, name, x=None, y=None, yaw_deg=0.0, z_mm=None, cad=None):
    """把一块板摆到基座系 (x, y) 的位置并绕竖直轴转 yaw_deg（相对 CAD 朝向）。

    x / y 只给一个时，另一个保持 CAD 原值。z 默认取 CAD 原值（板子落在托盘面上）。
    返回摆放后该板在基座系的实际位置 (x, y, z)。
    """
    cad = cad or cad_poses(m, d)
    Rb, pb = _base_frame(m, d)
    adr, dof, _ = _tile_joint(m, name)
    p_cad, q_cad = cad[name]
    p_cad_b = Rb.T @ (p_cad - pb) * 1000.0
    x = p_cad_b[0] if x is None else float(x)
    y = p_cad_b[1] if y is None else float(y)
    z = p_cad_b[2] if z_mm is None else float(z_mm)

    a = np.radians(yaw_deg) / 2.0
    q_yaw = np.array([np.cos(a), 0.0, 0.0, np.sin(a)])
    q = _quat_mul(q_yaw, q_cad)
    d.qpos[adr:adr + 3] = Rb @ (np.array([x, y, z]) / 1000.0) + pb
    d.qpos[adr + 3:adr + 7] = q / np.linalg.norm(q)
    d.qvel[dof:dof + 6] = 0.0
    mujoco.mj_forward(m, d)
    return x, y, z


def _body_min_z(m, d, b):
    lo = np.full(3, np.inf)
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != b:
            continue
        box = _geom_aabb(m, d, g)
        if box is not None:
            lo = np.minimum(lo, box[0])
    return float(lo[2])


def _geom_aabb(m, d, g):
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


def flip_piece(m, d, name):
    """把一块板**翻面**（绕一条水平轴转 180°），然后重新落回原来的高度。

    七巧板是等厚棱柱，翻面后外形正好是镜像，落到托盘上仍然是平的 —— 现场
    把实物翻过来就是这个效果，所以这是合法的物理状态，不是穿模作弊。
    绕的是"这块板轮廓的形心"，不是 body 原点（CAD 原点不在中心，绕它会平移）。
    """
    Rb, pb = _base_frame(m, d)
    adr, dof, b = _tile_joint(m, name)
    z_before = _body_min_z(m, d, b)

    p = d.qpos[adr:adr + 3].copy()
    q = d.qpos[adr + 3:adr + 7].copy()
    hull = _hull_xy(m, d, name)
    centre_xy = hull.mean(axis=0)

    # 世界系的水平轴（基座系的 y 轴，只差一个绕竖直轴的转角）
    axis = Rb @ np.array([0.0, 1.0, 0.0])
    q_flip = np.array([0.0, axis[0], axis[1], axis[2]])
    q_flip /= np.linalg.norm(q_flip)

    centre_world = Rb @ (np.array([centre_xy[0], centre_xy[1], 0.0]) / 1000.0) + pb
    centre_world = np.array([centre_world[0], centre_world[1], p[2]])
    # 绕中心翻：p' = c + R_flip (p - c)
    v = p - centre_world
    p = centre_world + (2.0 * np.dot(q_flip[1:], v) * q_flip[1:] - v)
    d.qpos[adr:adr + 3] = p
    d.qpos[adr + 3:adr + 7] = _quat_mul(q_flip, q)
    d.qpos[adr + 3:adr + 7] /= np.linalg.norm(d.qpos[adr + 3:adr + 7])
    d.qvel[dof:dof + 6] = 0.0
    mujoco.mj_forward(m, d)

    # 翻完高度会变（mesh 原点不一定在中间），按"最低点回到原位"重新落一次
    d.qpos[adr + 2] += (z_before - _body_min_z(m, d, b))
    mujoco.mj_forward(m, d)
    return name


def flip_pieces(m, d, names):
    for nm in names or ():
        flip_piece(m, d, nm)


def _hull_xy(m, d, name):
    """这块板在**基座系** xy 平面上的凸包顶点（单位 mm）。"""
    Rb, pb = _base_frame(m, d)
    _, _, b = _tile_joint(m, name)
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != b or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        adr, num = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        v = m.mesh_vert[adr:adr + num] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        pts.append((Rb.T @ (v - pb).T).T[:, :2] * 1000.0)
    if not pts:
        return np.zeros((0, 2))
    from scipy.spatial import ConvexHull

    p = np.vstack(pts)
    return p[ConvexHull(p).vertices]


def _inside(poly, pts):
    """射线法：pts 是否在多边形 poly 内。"""
    inside = np.zeros(len(pts), bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = (y1 > pts[:, 1]) != (y2 > pts[:, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (x2 - x1) * (pts[:, 1] - y1) / (y2 - y1) + x1
        inside ^= cond & (pts[:, 0] < xint)
    return inside


def overlaps(m, d, step_mm=2.0, names=None):
    """两两检查板子有没有压在一起，返回 [(a, b, 重叠面积 mm²), ...]。

    做法是把每块板的凸包撒成网格，数同时落在两块板里的格子——够用且不依赖
    额外库（板子都是平的薄片，投影到 xy 就是它们的轮廓）。
    """
    names = list(names or TILE_NAMES)
    hulls = {}
    for nm in names:
        h = _hull_xy(m, d, nm)
        if len(h) >= 3:
            hulls[nm] = h                  # 基座系 mm
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a not in hulls or b not in hulls:
                continue
            lo = np.maximum(hulls[a].min(axis=0), hulls[b].min(axis=0))
            hi = np.minimum(hulls[a].max(axis=0), hulls[b].max(axis=0))
            if np.any(hi <= lo):
                continue
            gx = np.arange(lo[0], hi[0], step_mm) + step_mm / 2
            gy = np.arange(lo[1], hi[1], step_mm) + step_mm / 2
            if not len(gx) or not len(gy):
                continue
            gxx, gyy = np.meshgrid(gx, gy)
            pts = np.column_stack([gxx.ravel(), gyy.ravel()])
            both = _inside(hulls[a], pts) & _inside(hulls[b], pts)
            area = float(both.sum()) * step_mm * step_mm
            if area > 0.5:
                out.append((a, b, area))
    return out


def scatter(m, d, seed=0, *, names=None, yaw_step_deg=45.0, margin_mm=12.0,
            step_mm=4.0, tries=300):
    """在原料托盘上把七块板随机撒开，**保证互不重叠**。

    - 位置只允许落在原料托盘（装着板子的那块）里，四周留 margin_mm；
    - 姿态取 yaw_step_deg 的整数倍（七巧板习惯用 45°）；
    - 大的先放，每放一块都用栅格占位图检查与已放板子的重叠；
    - 返回每块板的基座系 (x, y, yaw)。

    撒完之后可以用 `overlaps(m, d)` 复核（它用 2 mm 栅格，比这里更细）。
    """
    rng = np.random.default_rng(seed)
    names = list(names or TILE_NAMES)
    cad = cad_poses(m, d)
    Rb, pb = _base_frame(m, d)

    # ---- 原料托盘的范围（大、薄、装着板子的那块）---------------------------
    # 注意：基座系相对世界系转了 2.6°，所以**不能**只把 AABB 的两个角点转过去
    # （旋转角点会把 x/y 混在一起，算出来的范围是歪的）。必须转所有顶点再取 min/max。
    statics = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "statics")
    boards = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != statics or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        adr, num = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        v = m.mesh_vert[adr:adr + num] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        base = (Rb.T @ (v - pb).T).T * 1000.0
        lo, hi = base.min(axis=0), base.max(axis=0)
        size = hi - lo
        if min(size[:2]) >= 150.0 and size[2] <= 15.0:
            boards.append((lo, hi))
    if not boards:
        raise RuntimeError("找不到托盘，无法撒板子")
    tiles0 = measure().tiles
    cx = np.mean([tiles0[n][:2] for n in names], axis=0)
    board = min(boards, key=lambda b: np.linalg.norm((b[0][:2] + b[1][:2]) / 2 - cx))
    tray_lo, tray_hi = board[0][:2].copy(), board[1][:2].copy()

    # ---- 每块板的形状（相对自身原点，单位 mm）与面积 ----------------------
    shapes, areas = {}, {}
    # 形状要按 **CAD 位姿**量：连撒两次时，第二次不能拿上一次的姿态当形状，
    # 否则算出来的轮廓和实际摆下去的对不上（会莫名其妙地压在一起）。
    _key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    if _key < 0:
        raise RuntimeError('场景里没有 <keyframe name="home">')
    _saved = d.qpos.copy()
    mujoco.mj_resetDataKeyframe(m, d, _key)
    mujoco.mj_forward(m, d)
    try:
        for nm in names:
            adr, _, _ = _tile_joint(m, nm)
            p_cad = d.qpos[adr:adr + 3]
            hull = _hull_xy(m, d, nm)
            origin_xy = (Rb.T @ (p_cad - pb) * 1000.0)[:2]
            shapes[nm] = hull - origin_xy
            xs = shapes[nm][:, 0]
            ys = shapes[nm][:, 1]
            areas[nm] = 0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))
    finally:
        d.qpos[:] = _saved
        mujoco.mj_forward(m, d)

    def poly_at(nm, x, y, yaw):
        a = np.radians(yaw)
        Rz = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        return shapes[nm] @ Rz.T + (x, y)

    order = sorted(names, key=lambda n: -areas[n])      # 大的先放，好塞
    n_yaw = max(1, int(round(360.0 / yaw_step_deg)))

    # 从"留够边距"往"贴边"退：先试 margin_mm，挤不下就缩小边距再来。
    # 每档都要求：① 板子在托盘范围内 ② 两两不重叠，否则换一档。
    margins = [margin_mm] + [m for m in (9.0, 6.0, 3.0) if m < margin_mm]
    last = None
    for margin in margins:
        blo = tray_lo + margin
        bhi = tray_hi - margin
        if np.any(blo >= bhi):
            continue

        gx = np.arange(blo[0], bhi[0], step_mm) + step_mm / 2
        gy = np.arange(blo[1], bhi[1], step_mm) + step_mm / 2
        GX, GY = np.meshgrid(gx, gy)
        grid = np.column_stack([GX.ravel(), GY.ravel()])
        occupied = np.zeros(len(grid), bool)

        def cells_in_bbox(poly):
            lo = np.maximum(poly.min(axis=0), blo)
            hi = np.minimum(poly.max(axis=0), bhi)
            if np.any(hi <= lo):
                return None
            sel = ((grid[:, 0] >= lo[0]) & (grid[:, 0] <= hi[0])
                   & (grid[:, 1] >= lo[1]) & (grid[:, 1] <= hi[1]))
            return sel

        for attempt in range(6):        # 4 mm 栅格偶尔漏一丝，重摆一遍就好
            occupied[:] = False
            out, ok_all = {}, True
            for nm in order:
                ok = False
                for _ in range(tries):
                    yaw = float(rng.integers(0, n_yaw)) * yaw_step_deg
                    # 这块板在 yaw 下的外接框，**相对自身原点**。七块板的 CAD 原点
                    # 都不在几何中心（最远差 57 mm），所以必须用 min/max 而不是
                    # ±半尺寸，否则按"中心 ± 半宽"采样会让板子从托盘边缘漏出去。
                    rel = poly_at(nm, 0.0, 0.0, yaw)
                    rlo, rhi = rel.min(axis=0), rel.max(axis=0)
                    x_lo, x_hi = blo[0] - rlo[0], bhi[0] - rhi[0]
                    y_lo, y_hi = blo[1] - rlo[1], bhi[1] - rhi[1]
                    if x_hi <= x_lo or y_hi <= y_lo:
                        continue
                    x = float(rng.uniform(x_lo, x_hi))
                    y = float(rng.uniform(y_lo, y_hi))
                    poly = poly_at(nm, x, y, yaw)
                    sel = cells_in_bbox(poly)
                    if sel is None:
                        continue
                    inside = _inside(poly, grid[sel])
                    if np.any(occupied[sel] & inside):
                        continue
                    occupied[sel] |= inside
                    out[nm] = (x, y, yaw)
                    ok = True
                    break
                if not ok:
                    ok_all = False
                    break
            if not ok_all:
                last = f"margin {margin:.0f} mm 时挤不下"
                break

            for nm, (x, y, yaw) in out.items():
                set_piece(m, d, nm, x, y, yaw, cad=cad)
            mujoco.mj_forward(m, d)

            bad = overlaps(m, d, step_mm=2.0, names=names)
            outside = []
            for nm in names:
                hv = _hull_xy(m, d, nm)
                over = max(tray_lo[0] - hv[:, 0].min(), hv[:, 0].max() - tray_hi[0],
                           tray_lo[1] - hv[:, 1].min(), hv[:, 1].max() - tray_hi[1])
                if over > 0.5:
                    outside.append((nm, round(float(over), 1)))
            last = f"margin {margin:.0f} mm 时：重叠 {bad}，越界 {outside}"
            if not bad and not outside:
                return {nm: out[nm] for nm in names}

    raise RuntimeError(f"摆了多档边距仍不满足：{last}")
