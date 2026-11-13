# -*- coding: utf-8 -*-
"""手玩沙盒：六个关节各一根滑条 + 吸盘开关 + 3D 视图 + 腕上相机。

    python play.py
    python play.py --selftest     # 不开窗口，自检用

两个窗口
--------
* **MuJoCo**（3D）：鼠标左键拖动旋转、中键平移、滚轮缩放。想复位视角就按 F9 再按 F5。
* **控制台**（Tk）：六根滑条 + 吸盘 + 腕上相机画面。

滑条上的数字是**控制器口径的关节角**（度）。因为关节零位已经用示教点标定过，
滑条上的读数和示教器上看到的数就是同一个数——这是这个场景最值得信任的一层。

用法
----
* 拖滑条 → 机械臂以 150°/s 平滑跟过去（不是瞬移），松开就停。
* 点「吸盘」或按 `s` → 吸住离吸盘口最近的一块板（45 mm 内），再按一次放开。
  吸住之后板子跟着手腕走，转手腕它也会跟着转。
* 「回 HOME」→ 回到场景里的 `home` 关键帧（= 你们的拍照位附近）。
* 「打印位姿」或按 `p` → 打印六个关节角 + 法兰位姿（mm/度），可直接贴进代码。
* `g` 开关重力（关掉的话板子会浮在空中，方便摆位）。
* 鼠标把某块板拖走也可以：3D 窗口里 Ctrl + 右键拖动 = 施加外力扰动。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from icim_sim import cell as cellmod
from icim_sim.robot import SimCobotClient

NAMES = [f"J{i}" for i in range(1, 7)]
VMAX_DEG_S = 150.0          # 滑条跟过去的最大角速度
TICK_MS = 16                # 约 60 Hz


class PlayCell:
    def __init__(self, robot: SimCobotClient, cam_size=(480, 270)):
        self.robot = robot
        self.m, self.d = robot.m, robot.d
        self.key = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_KEY, "home")
        if self.key < 0:
            raise SystemExit('场景里没有 <keyframe name="home">，先跑 tools/build_scene.py')
        self.home_qpos = None            # 由 reset() 填
        self.reset()

        self.lo = np.degrees(self.m.jnt_range[:6, 0])
        self.hi = np.degrees(self.m.jnt_range[:6, 1])
        self.target = np.degrees(self.d.qpos[:6]).copy()

        self.cam_w, self.cam_h = cam_size
        self.ren = None
        self.gravity = True
        self.t_last = time.time()
        self.fps = 0.0
        # 摆位模式
        self.place_on = False
        self.sel = 0
        self.dz = {nm: 0.0 for nm in cellmod.TILE_NAMES}
        self._rgba = {nm: self._geom_rgba(nm) for nm in cellmod.TILE_NAMES}

    # -------------------------------------------------------------- 摆位模式
    def _geom_of(self, name):
        b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)
        for g in range(self.m.ngeom):
            if self.m.geom_bodyid[g] == b:
                return g
        raise RuntimeError(f"{name} 没有 geom")

    def _geom_rgba(self, name):
        return self.m.geom_rgba[self._geom_of(name)].copy()

    def _paint(self, name, rgba=None):
        g = self._geom_of(name)
        self.m.geom_rgba[g] = self._rgba[name] if rgba is None else rgba

    def _pose_of(self, name):
        from icim_sim import layout as _layout
        return _layout.capture(self.m, self.d)[name]

    def enter_place(self):
        """进入摆位模式：关重力、选中第一块、高亮它。"""
        self.place_on = True
        self.sel = 0
        if self.gravity:
            self.gravity = False
            self.m.opt.gravity[:] = (0, 0, 0)
        self._paint_sel()

    def exit_place(self):
        """退出摆位模式：恢复重力、取消高亮。"""
        self.place_on = False
        for nm in cellmod.TILE_NAMES:
            self._paint(nm)
        if not self.gravity:
            self.gravity = True
            self.m.opt.gravity[:] = (0, 0, -9.81)
        mujoco.mj_forward(self.m, self.d)

    def _paint_sel(self):
        for i, nm in enumerate(cellmod.TILE_NAMES):
            self._paint(nm, (1.0, 1.0, 1.0, 1.0) if i == self.sel else None)

    def select_next(self):
        self.sel = (self.sel + 1) % len(cellmod.TILE_NAMES)
        self._paint_sel()

    def selected(self):
        return cellmod.TILE_NAMES[self.sel]

    def _z_of(self, name):
        """这块板现在（基座系）的 z，mm。"""
        adr, _, _ = cellmod._tile_joint(self.m, name)
        Rb, pb = cellmod._base_frame(self.m, self.d)
        return float((Rb.T @ (self.d.qpos[adr:adr + 3] - pb))[2] * 1000.0)

    def move_selected(self, dx=0.0, dy=0.0, dz=0.0, dyaw=0.0):
        """平移/转动/升降当前选中的板（dx,dy,dz 单位 mm，dyaw 单位度）。

        z 一律**沿用板子当前的 z**（以前这里用 CAD 的 z，而自由关节的原点在托盘
        面上，结果一按方向键板子就沉到 z=0、看着像掉进托盘里）。
        """
        nm = self.selected()
        x, y, yaw = self._pose_of(nm)
        z = self._z_of(nm) + dz
        cad = cellmod.cad_poses(self.m, self.d)
        cellmod.set_piece(self.m, self.d, nm, x + dx, y + dy, yaw + dyaw,
                          z_mm=z, cad=cad)
        mujoco.mj_forward(self.m, self.d)
        self._paint_sel()
        return self._pose_of(nm)

    # ------------------------------------------------------------------ 布局
    def save_layout(self, name=None):
        from icim_sim import layout as _layout
        target = name or (_layout.choose(None) or "my_layout")
        path = _layout.save_from(self.robot, getattr(target, "stem", target))
        return path

    def load_layout(self, name=None):
        from icim_sim import layout as _layout
        path = _layout.apply_to(self.robot, name, verbose=False)
        if path is not None:
            self.home_qpos = self.d.qpos.copy()
            self.target = np.degrees(self.d.qpos[:6]).copy()
        return path

    # ------------------------------------------------------------------ state
    def reset(self):
        """回到初始状态：六个关节 + 七块板的位置一起复位。

        优先级：`layouts/` 里选中的那份布局（含关节角和七块板）→
        包根目录 `start_pose.json`（只管关节）→ 场景里烘焙的 keyframe。
        """
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.key)
        self.robot.reset_pose()          # 有 start_pose.json 就套上去
        from icim_sim import layout as _layout
        path = _layout.choose(None)
        if path is not None:
            _layout.apply(self.m, self.d, _layout.read(path))
        mujoco.mj_forward(self.m, self.d)
        self.home_qpos = self.d.qpos.copy()
        self.robot.suction_release()
        self.target = np.degrees(self.d.qpos[:6]).copy()

    def joints_deg(self):
        return np.degrees(self.d.qpos[:6]).copy()

    def reseed(self, seed=None):
        """把七块板在原料盘上随机撒开（互不重叠）。"""
        self.seed = np.random.default_rng().integers(0, 10_000) if seed is None else seed
        cellmod.scatter(self.m, self.d, seed=int(self.seed))
        self.robot.suction_release()
        return int(self.seed)

    def flange(self):
        return self.robot.current_pose()

    # ------------------------------------------------------------------ drive
    def advance(self, dt):
        """把六个关节往滑条目标上推，最多 VMAX_DEG_S；再走一步物理。"""
        q = np.degrees(self.d.qpos[:6])
        dq = np.clip(self.target - q, -VMAX_DEG_S * dt, VMAX_DEG_S * dt)
        self.d.qpos[:6] = np.radians(np.clip(q + dq, self.lo, self.hi))
        self.d.qvel[:6] = 0.0
        mujoco.mj_step(self.m, self.d)
        self.robot.hold_attached()        # 吸住的板跟着手腕走
        mujoco.mj_forward(self.m, self.d)

    def status(self):
        p = self.flange()
        suction = self.robot.attached or "—"
        text = (f"法兰 x{p[0]:8.2f} y{p[1]:8.2f} z{p[2]:8.2f} mm   "
                f"rx{p[3]:7.2f} ry{p[4]:6.2f} rz{p[5]:7.2f}°\n"
                f"吸盘：{suction}     重力：{'开' if self.gravity else '关'}      "
                f"{self.fps:4.1f} fps")
        return text

    # ------------------------------------------------------------------- suct
    def toggle_suction(self):
        if self.robot.attached:
            self.robot.suction_release()
        else:
            self.robot.suction_grab()
        return self.robot.attached


# --------------------------------------------------------------------- selftest
def selftest():
    """不开窗口跑一遍，确认驱动/吸盘/复位都真的动。"""
    robot = SimCobotClient(layout="auto")
    robot.connect()
    cell = PlayCell(robot)
    print(f"home 关节 {np.round(cell.joints_deg(), 2).tolist()}")
    print(f"home 法兰 {np.round(cell.flange(), 2).tolist()}")

    # 1) 关节能按目标动，且不超过限速
    cell.target = cell.joints_deg() + np.array([10.0, 0, 0, 0, 0, 0])
    before = cell.joints_deg()[0]
    cell.advance(1 / 60)
    moved = cell.joints_deg()[0] - before
    print(f"一步 J1 动了 {moved:.3f}°（限速 {VMAX_DEG_S / 60:.2f}°/帧）")
    for _ in range(20):
        cell.advance(1 / 60)
    print(f"20 帧后 J1 = {cell.joints_deg()[0]:.2f}°（目标 {cell.joints_deg()[0] + 0:.2f}）")

    # 2) 走到某块板上方、贴下去、吸住、抬起来
    tile = "piece_large_1"
    b = mujoco.mj_name2id(robot.m, mujoco.mjtObj.mjOBJ_BODY, tile)
    centre = robot._body_aabb_centre(b) * 1000
    base_z = robot.world_to_base_point([0, 0, centre[2]])[2]
    xy = robot.world_to_base_point([centre[0], centre[1], 0.0])
    rpy = [-178.633, 0.416, 139.234]

    def goto(pose, label, tol=0.25):
        q = robot._pose_to_joint(pose, cell.joints_deg(), label)
        cell.target = np.array(q)
        for _ in range(500):
            cell.advance(1 / 60)
            if np.abs(cell.target - cell.joints_deg()).max() < tol:
                break
        gap = np.linalg.norm(robot._body_aabb_centre(b)
                             - robot.d.site_xpos[robot._tcp]) * 1000
        print(f"{label}: 法兰 z {cell.flange()[2]:7.2f} mm，吸盘口离板中心 {gap:5.2f} mm")

    goto([xy[0], xy[1], base_z + 252.5 + 60.0] + rpy, "上方 60 mm")
    goto([xy[0], xy[1], base_z + 252.5] + rpy, "贴到板面")

    z0 = robot._body_aabb_centre(b)[2] * 1000
    print(f"吸附：{cell.toggle_suction()}")
    goto([xy[0], xy[1], base_z + 252.5 + 40.0] + rpy, "抬起 40 mm")
    z1 = robot._body_aabb_centre(b)[2] * 1000
    print(f"板子 z {z0:.2f} → {z1:.2f} mm（跟上了 {z1 - z0:+.2f} mm）")

    print(f"放开：{cell.toggle_suction()}")
    for _ in range(180):
        cell.advance(1 / 60)
    z2 = robot._body_aabb_centre(b)[2] * 1000
    print(f"放开后自由下落，板子 z = {z2:.2f} mm")

    cell.reset()
    print(f"复位后 J1 = {cell.joints_deg()[0]:.2f}°")
    print("selftest 通过")


# ------------------------------------------------------------------------ GUI
def run_gui(args):
    import tkinter as tk
    from PIL import Image, ImageTk

    import mujoco.viewer

    robot = SimCobotClient(layout="auto")
    robot.connect()
    cell = PlayCell(robot, cam_size=(args.size[0], args.size[1]))
    cell.ren = mujoco.Renderer(cell.m, cell.cam_h, cell.cam_w)

    viewer = mujoco.viewer.launch_passive(cell.m, cell.d)
    viewer.cam.lookat[:] = (0.33, 0.40, 0.20)
    viewer.cam.distance = 1.5
    viewer.cam.azimuth = 130
    viewer.cam.elevation = -18

    root = tk.Tk()
    root.title("ICIM 仿真沙盒 — 六轴手动 + 吸盘")
    root.configure(bg="#1b1b1f")

    # 顶部两行：第一行是状态（大、亮），第二行是常驻操作提示（小、灰）
    bar = tk.Label(root, text="", justify="left", anchor="w", bg="#1b1b1f",
                   fg="#c8e6ff", font=("Consolas", 12, "bold"))
    bar.pack(fill="x", padx=8, pady=(6, 0))
    hints = tk.Label(
        root,
        text="Tab 摆位 · 方向键移 1mm（Shift 10mm）· , . 转 15° · [ ] 升降 · "
             "Ctrl+S 存布局 · Ctrl+L 读布局 · s 吸放 · n 重新撒开 · g 重力",
        justify="left", anchor="w", bg="#1b1b1f", fg="#7f8c9b",
        font=("Consolas", 9))
    hints.pack(fill="x", padx=8, pady=(0, 4))

    panel = tk.Frame(root, bg="#1b1b1f")
    panel.pack(fill="x", padx=8)

    sliders, labels = [], []
    for i in range(6):
        row = tk.Frame(panel, bg="#1b1b1f")
        row.pack(fill="x")
        tk.Label(row, text=NAMES[i], width=3, bg="#1b1b1f", fg="#9fd0ff",
                 font=("Consolas", 11, "bold")).pack(side="left")
        s = tk.Scale(row, from_=cell.lo[i], to=cell.hi[i], resolution=0.1,
                     orient="horizontal", length=420, bg="#26262c", fg="#dddddd",
                     troughcolor="#3a3a44", highlightthickness=0,
                     command=lambda v, i=i: on_slide(i, v))
        s.set(np.degrees(cell.d.qpos[i]))
        s.pack(side="left", padx=4)
        lab = tk.Label(row, text="", width=11, anchor="e", bg="#1b1b1f",
                       fg="#ffe082", font=("Consolas", 11))
        lab.pack(side="left")
        sliders.append(s)
        labels.append(lab)

    def on_slide(i, v):
        cell.target[i] = float(v)

    state = {"suction": False}

    def sync_suction():
        on = cell.robot.attached is not None
        state["suction"] = on
        btn_suc.configure(text="吸盘：吸住" if on else "吸盘：放开",
                          bg="#2e7d32" if on else "#455a64")

    def do_suction():
        cell.toggle_suction()
        sync_suction()

    def do_reset():
        cell.reset()
        for i, s in enumerate(sliders):
            s.set(np.degrees(cell.d.qpos[i]))
        sync_suction()

    def do_print():
        p = cell.flange()
        data = {"joint_deg": [round(float(v), 3) for v in cell.joints_deg()],
                "flange": [round(float(v), 3) for v in p]}
        print(json.dumps(data, indent=1))
        bar.configure(text="已打印到终端")

    def do_gravity():
        cell.gravity = not cell.gravity
        cell.m.opt.gravity[:] = (0, 0, -9.81) if cell.gravity else (0, 0, 0)
        btn_g.configure(text="重力：开" if cell.gravity else "重力：关")

    def do_seed():
        s = cell.reseed()
        sync_suction()
        bar.configure(text=f"重新撒开（seed={s}）")

    # ---------------------------------------------------------- 摆位 / 布局
    def set_place(on):
        if on and not cell.place_on:
            cell.enter_place()
            for s in sliders:
                s.configure(state="disabled")
            root.focus_force()
            btn_place.configure(text="摆位：开", bg="#6d4c41")
        elif not on and cell.place_on:
            cell.exit_place()
            for s in sliders:
                s.configure(state="normal")
            btn_place.configure(text="摆位：关", bg="#455a64")
        show_place()

    def toggle_place():
        if not cell.place_on:
            set_place(True)
        else:
            cell.select_next()
            show_place()

    def show_place():
        if not cell.place_on:
            bar.configure(text=cell.status())
            return
        x, y, yaw = cell._pose_of(cell.selected())
        bar.configure(text=f"摆位：{cell.selected()}   x={x:7.2f} y={y:7.2f} "
                           f"yaw={yaw:6.1f}°   "
                           f"[方向键]平移 [,. ]转 [ ] 升降  Tab 换块  Esc 退出")

    def nudge(dx=0.0, dy=0.0, dz=0.0, dyaw=0.0):
        if not cell.place_on:
            return
        cell.move_selected(dx=dx, dy=dy, dz=dz, dyaw=dyaw)
        show_place()

    def do_save_layout():
        """存到当前默认布局 —— 会覆盖它，所以先问一句。"""
        from tkinter import messagebox

        from icim_sim import layout as _layout

        target = _layout.choose(None)
        if target is None:
            name = "default"
            target = _layout.LAYOUTS / "default.json"
        else:
            name = target.stem
        if not messagebox.askyesno(
                "存布局",
                f"把当前七块板 + 六个关节写进布局 “{name}” 吗？\n\n"
                f"{target}\n\n"
                f"这个布局会被覆盖，读回它就会变成现在这个样子。",
                parent=root):
            bar.configure(text="已取消，没有改动布局")
            return
        try:
            p = cell.save_layout(name)
        except Exception as exc:                       # noqa: BLE001
            bar.configure(text=f"存布局失败：{exc}")
            return
        bar.configure(text=f"已存布局 → {p}")

    def do_save_layout_as(_e=None):
        from tkinter import messagebox, simpledialog

        from icim_sim import layout as _layout

        name = simpledialog.askstring("另存为", "布局名字（不用写 .json）：",
                                      parent=root)
        if not name:
            return
        target = _layout.LAYOUTS / f"{name}.json"
        if target.exists() and not messagebox.askyesno(
                "覆盖？", f"布局 “{name}” 已经存在，要覆盖它吗？", parent=root):
            return
        p = cell.save_layout(name)
        bar.configure(text=f"已存布局 → {p}")

    def do_load_layout(_e=None):
        p = cell.load_layout()
        for i, s in enumerate(sliders):
            s.set(np.degrees(cell.d.qpos[i]))
        sync_suction()
        bar.configure(text=(f"已读布局 ← {p}" if p else "layouts/ 里没有布局文件"))

    # 按钮分两行：上面是"动作"，下面是"文件"
    btns1 = tk.Frame(root, bg="#1b1b1f")
    btns1.pack(fill="x", padx=8, pady=(0, 2))
    btns2 = tk.Frame(root, bg="#1b1b1f")
    btns2.pack(fill="x", padx=8, pady=(0, 6))

    btn_suc = tk.Button(btns1, text="吸盘：放开", width=14, command=do_suction,
                        bg="#455a64", fg="white", relief="flat")
    btn_suc.pack(side="left", padx=3)
    tk.Button(btns1, text="回 HOME", width=10, command=do_reset,
              relief="flat").pack(side="left", padx=3)
    tk.Button(btns1, text="重新撒开", width=10, command=do_seed,
              relief="flat").pack(side="left", padx=3)
    btn_place = tk.Button(btns1, text="摆位：关", width=12, command=toggle_place,
                          bg="#455a64", fg="white", relief="flat")
    btn_place.pack(side="left", padx=3)
    btn_g = tk.Button(btns1, text="重力：开", width=10, command=do_gravity,
                      relief="flat")
    btn_g.pack(side="left", padx=3)

    tk.Button(btns2, text="存布局", width=10, command=do_save_layout,
              relief="flat").pack(side="left", padx=3)
    tk.Button(btns2, text="另存为", width=8, command=do_save_layout_as,
              relief="flat").pack(side="left", padx=3)
    tk.Button(btns2, text="读布局", width=10, command=do_load_layout,
              relief="flat").pack(side="left", padx=3)
    tk.Button(btns2, text="打印位姿", width=10, command=do_print,
              relief="flat").pack(side="left", padx=3)

    # 腕上相机 = 模型里 RGB 模块的成像面（见 tools/build_scene.py）
    cam_row = tk.Frame(root, bg="#1b1b1f")
    cam_row.pack(padx=8, pady=(2, 8))
    lab_cam = tk.Label(cam_row, text="wrist_cam：RGB 模块成像面", bg="#1b1b1f",
                       fg="#9fd0ff", font=("Consolas", 9))
    lab_cam.pack()
    cam_label = tk.Label(cam_row, bd=0, bg="#1b1b1f")
    cam_label.pack()

    root.bind("<Key-s>", lambda e: do_suction())
    root.bind("<Key-r>", lambda e: do_reset())
    root.bind("<Key-n>", lambda e: do_seed())
    root.bind("<Key-p>", lambda e: do_print())
    root.bind("<Key-g>", lambda e: do_gravity())
    root.bind("<Tab>", lambda e: (toggle_place(), "break")[1])
    root.bind("<Control-s>", lambda e: (do_save_layout(), "break")[1])
    root.bind("<Control-S>", lambda e: (do_save_layout_as(), "break")[1])
    root.bind("<Control-l>", lambda e: (do_load_layout(), "break")[1])
    for key, args in (("<Left>", dict(dx=-1.0)), ("<Right>", dict(dx=1.0)),
                      ("<Up>", dict(dy=1.0)), ("<Down>", dict(dy=-1.0)),
                      ("<Shift-Left>", dict(dx=-10.0)),
                      ("<Shift-Right>", dict(dx=10.0)),
                      ("<Shift-Up>", dict(dy=10.0)),
                      ("<Shift-Down>", dict(dy=-10.0)),
                      ("<comma>", dict(dyaw=-15.0)),
                      ("<period>", dict(dyaw=15.0)),
                      ("<Shift-comma>", dict(dyaw=-45.0)),
                      ("<Shift-period>", dict(dyaw=45.0)),
                      ("<bracketleft>", dict(dz=-1.0)),
                      ("<bracketright>", dict(dz=1.0))):
        root.bind(key, lambda e, a=args: (nudge(**a), "break")[1])

    def on_escape(_e=None):
        if cell.place_on:
            set_place(False)
        else:
            viewer.close()
            root.destroy()
        return "break"

    root.bind("<Escape>", on_escape)

    def tick():
        if not viewer.is_running():
            root.destroy()
            return
        cell.advance(TICK_MS / 1000.0)
        viewer.sync()

        now = time.time()
        dt = now - cell.t_last
        if dt > 0.4:
            cell.fps = 0.9 * cell.fps + 0.1 * (1.0 / max(dt, 1e-3)) if cell.fps else 1.0 / dt
            cell.t_last = now

        for i, lab in enumerate(labels):
            lab.configure(text=f"{np.degrees(cell.d.qpos[i]):+8.2f}°")
        if cell.robot.attached is not None and not state["suction"]:
            sync_suction()
        if cell.place_on:
            show_place()
        else:
            bar.configure(text=cell.status())

        cell.ren.update_scene(cell.d, camera="wrist_cam")
        img = Image.fromarray(cell.ren.render())
        cam_label.img = ImageTk.PhotoImage(img)
        cam_label.configure(image=cam_label.img)
        root.after(TICK_MS, tick)

    tick()
    root.mainloop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="不开窗口，自检驱动与吸附")
    ap.add_argument("--size", type=int, nargs=2, default=(480, 270), help="腕相机预览尺寸")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    else:
        run_gui(args)


if __name__ == "__main__":
    main()
