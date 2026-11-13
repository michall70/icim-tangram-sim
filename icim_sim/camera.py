# -*- coding: utf-8 -*-
"""把 MuJoCo 的画面对着 `RealSenseCamera` 的接口伪装成真相机。

上层 `main.py` 只用到三样：`RealSenseCamera(**CAMERA_PARAMS)`、`open()`、
`grab_color(warmup=...)`。这里全部照做，返回 BGR uint8 的 1280×720 图像。

两个必须做对的地方
------------------
1. **相机挂在手腕上**，所以必须从"机器人当前姿态"渲染。相机体是
   `rgb_camera`，`<camera name="wrist_cam">` 挂在它下面，MuJoCo 会自己
   跟着 FK 走。
2. **交出去的是"原图"**：MuJoCo 渲的是理想针孔，而队友的代码会自己调
   `CameraMapper.undistort()`。所以这里先用 `sim.lens` 按真实镜头畸变把
   图打歪，他们去畸变之后正好回到理想针孔（见 `sim/lens.py` 的说明）。
"""
from __future__ import annotations

import numpy as np

from . import lens


class _Registry:
    """当前活动的仿真机器人（相机要跟着它的姿态走）。"""

    robot = None


def set_active_robot(robot) -> None:
    _Registry.robot = robot


def get_active_robot():
    return _Registry.robot


class SimCamera:
    """`common.realsense_camera.RealSenseCamera` 的仿真替身。"""

    def __init__(self, width=1280, height=720, fps=30, exposure=None, gain=None,
                 brightness=None, contrast=None, white_balance=None,
                 camera="wrist_cam", camera_matrix=None, dist_coeffs=None,
                 simulate_lens=True, warmup_steps=0, **_ignored):
        # **_ignored：队友的 `CAMERA_PARAMS` 会往里加键（2026-11-01 加了
        # `depth=True` / `auto_usb=True`），他们是 `RealSenseCamera(**CAMERA_PARAMS)`
        # 直接展开的，签名对不上就 TypeError。仿真是"能收就收、用不到的忽略"，
        # 这样他们加参数不用先来问我们。
        self.width = int(width)
        self.height = int(height)
        self.fps = fps
        self.camera = camera
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.simulate_lens = bool(simulate_lens and camera_matrix is not None)
        self.warmup_steps = int(warmup_steps)
        self._opened = False
        self._renderer = None
        self.frames = 0

    # ------------------------------------------------------------- lifecycle
    def open(self):
        # 队友的 main.py 先 camera.open() 再 robot.connect()，所以这里不能要求
        # 机器人已经连上；真正渲染的时候再用，那时一定连上了。
        robot = get_active_robot()
        self._opened = True
        ready = robot is not None and robot.m is not None
        print(f"[sim.camera] {self.camera} {self.width}×{self.height}"
              f"{'，带镜头畸变' if self.simulate_lens else ''}"
              f"{'' if ready else '（机器人尚未连接，出图时再绑定）'}")

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._opened = False

    # ---------------------------------------------------------------- grabbing
    def _render(self):
        import mujoco

        robot = get_active_robot()
        if robot is None or robot.m is None:
            raise RuntimeError("仿真机器人未连接")
        if self._renderer is None:
            self._renderer = mujoco.Renderer(robot.m, self.height, self.width)
        if self.warmup_steps:
            robot.step(self.warmup_steps)
        self._renderer.update_scene(robot.d, camera=self.camera)
        rgb = self._renderer.render()
        return rgb[:, :, ::-1].copy()          # RGB -> BGR

    def grab_color(self, warmup=10):
        """采一帧彩色图（BGR）。`warmup` 在仿真里没有曝光要等，直接忽略。"""
        img = self._render()
        if self.simulate_lens:
            img = lens.distort_image(img, self.camera_matrix, self.dist_coeffs)
        self.frames += 1
        return img

    def get_aligned_frames(self):
        """`(color, depth, intrinsics)`；仿真没有深度流，depth 返回 None。"""
        return self.grab_color(), None, self.camera_matrix

    # ------------------------------------------------------------------ extras
    def render_ideal(self):
        """未加畸变的理想针孔图，仅用于自检。"""
        return self._render()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()


def save_png(path, bgr):
    """用 cv2 存图（venv 里已有 opencv-python-headless）。"""
    import cv2

    cv2.imwrite(str(path), np.ascontiguousarray(bgr))
