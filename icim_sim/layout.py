# -*- coding: utf-8 -*-
"""初始布局：七块板 + 机器人关节角，可存多份、运行时任选。

为什么单独放 `layouts/`（不放 `assets/`）
------------------------------------------
`assets/` 是工作间从 STEP 生成的东西，改了会被重新发布覆盖。
布局是**使用者的东西**：你摆好一套，存下来，下次直接读。

    layouts/
      default.json      出厂那份（作者发布；没有它就用场景里烘焙的散乱摆放）
      flat.json         自己存的
      active.txt        一行：默认读哪份（可选）

选哪份（`choose()`）：
    1. 命令行 `--layout 名字`
    2. `layouts/active.txt`
    3. `default.json`
    4. 只有一份 → 就是它
    5. 一份都没有 → 返回 None（保持场景原样），不报错

文件格式（角度：度；位置：基座系 mm；yaw 是**相对 CAD 朝向**的转角）：

    {
      "robot_joint_deg": [-98.3, -110.0, -30.5, -140.1, 90.0, 90.0],
      "pieces": {
        "piece_large_1": {"x": -151.48, "y": -453.82, "yaw": -30.0}
      }
    }
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from . import cell
from .paths import PACKAGE_ROOT

LAYOUTS = PACKAGE_ROOT / "layouts"


# --------------------------------------------------------------------- files
def list_layouts() -> list:
    if not LAYOUTS.is_dir():
        return []
    return sorted(p for p in LAYOUTS.glob("*.json"))


def active_name():
    f = LAYOUTS / "active.txt"
    if not f.is_file():
        return None
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return None


def set_active(name: str) -> Path:
    LAYOUTS.mkdir(parents=True, exist_ok=True)
    p = LAYOUTS / "active.txt"
    p.write_text(f"{name}\n", encoding="utf-8")
    return p


def choose(choice=None):
    """按优先级挑一份布局；都没有就返回 None（保持场景原样）。"""
    if choice:
        cand = Path(choice)
        if cand.is_file():
            return cand
        for p in list_layouts():
            if p.stem == str(choice) or p.name == str(choice):
                return p
        raise SystemExit(f"找不到布局 {choice}（可用：{[p.stem for p in list_layouts()]}）")
    act = active_name()
    if act:
        for p in list_layouts():
            if p.stem == act or p.name == act:
                return p
        print(f"[layout] active.txt 写的是 {act!r}，但没有这个文件，忽略")
    default = LAYOUTS / "default.json"
    if default.is_file():
        return default
    files = list_layouts()
    if len(files) == 1:
        return files[0]
    if not files:
        return None
    raise SystemExit("layouts/ 下有多份布局，请指定：\n  "
                     + "\n  ".join(p.stem for p in files)
                     + "\n用法：--layout <名字>，或写进 layouts/active.txt")


# ------------------------------------------------------------------ read/write
def read(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, robot_deg, pieces) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "_note": "初始布局：robot_joint_deg 是六个关节角（度）；pieces 是七块板的"
                 "基座系位置（mm）和相对 CAD 朝向的 yaw（度）。直接编辑也行。",
        "robot_joint_deg": [round(float(v), 3) for v in robot_deg],
        "pieces": {k: {"x": round(float(v[0]), 3),
                       "y": round(float(v[1]), 3),
                       "yaw": round(float(v[2]), 3)}
                   for k, v in pieces.items()},
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return path


def _cad_yaw(m, d, name):
    """这块板 CAD 位姿下绕竖直轴的朝向（度）。"""
    cad = cell.cad_poses(m, d)
    _, q = cad[name]
    w, x, y, z = q
    return float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))


def _yaw_of(R):
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


# ------------------------------------------------------------------- apply/cap
def capture(m, d) -> dict:
    """把当前七块板读成布局字典（yaw 是相对 CAD 的转角）。"""
    Rb, pb = cell._base_frame(m, d)
    pieces = {}
    for nm in cell.TILE_NAMES:
        adr, _, b = cell._tile_joint(m, nm)
        p = (Rb.T @ (d.qpos[adr:adr + 3] - pb)) * 1000.0
        yaw_now = _yaw_of(d.xmat[b].reshape(3, 3))
        yaw = ((yaw_now - _cad_yaw(m, d, nm) + 180.0) % 360.0) - 180.0
        pieces[nm] = (float(p[0]), float(p[1]), float(yaw))
    return pieces


def apply(m, d, data: dict) -> None:
    """把一份布局套到场景上（板子 + 关节角）。"""
    cad = cell.cad_poses(m, d)
    for nm, pose in (data.get("pieces") or {}).items():
        if nm not in cell.TILE_NAMES:
            print(f"[layout] 跳过不认识的板 {nm}")
            continue
        if isinstance(pose, dict):
            x, y, yaw = pose.get("x"), pose.get("y"), pose.get("yaw", 0.0)
        else:                                   # 也接受 [x, y, yaw]
            x, y, yaw = pose
        cell.set_piece(m, d, nm, x, y, yaw, cad=cad)
    joints = data.get("robot_joint_deg")
    if joints:
        if len(joints) != 6:
            raise SystemExit("robot_joint_deg 必须是 6 个数")
        d.qpos[:6] = np.radians([float(v) for v in joints])
        d.qvel[:6] = 0.0
    mujoco.mj_forward(m, d)


def apply_to(robot, choice=None, *, verbose=True):
    """给一个已连接的 SimCobotClient 套用默认/指定布局。返回用到的文件或 None。"""
    path = choose(choice)
    if path is None:
        if verbose:
            print("[layout] 没有布局文件，保持场景原样")
        return None
    data = read(path)
    apply(robot.m, robot.d, data)
    if verbose:
        n = len(data.get("pieces") or {})
        print(f"[layout] 已套用 {path.name}：{n} 块板"
              f"{'，含关节角' if data.get('robot_joint_deg') else ''}")
    return path


def save_from(robot, name="my_layout", *, with_joints=True):
    """把当前场景存成一份布局；`name` 可以是文件名或路径。"""
    path = Path(name)
    if path.suffix != ".json":
        path = LAYOUTS / f"{path.name}.json" if not path.is_absolute() else path.with_suffix(".json")
    pieces = capture(robot.m, robot.d)
    joints = [float(v) for v in np.degrees(robot.d.qpos[:6])] if with_joints else [0.0] * 6
    return write(path, joints, pieces)
