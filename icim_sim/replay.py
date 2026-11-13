# -*- coding: utf-8 -*-
"""录一段仿真过程，跑完用 MuJoCo 窗口慢慢重放。

为什么不在跑的时候直接开窗口：机器人是**运动学驱动**的，
一步只走 2 ms 物理时间，跑起来比实时快 4 倍（`time_scale`），窗口里就是一片残影。
先录下来、再按帧回放，才看得清"它到底怎么搬的"。

用法（由 `run.py` 默认的 `--replay` 调，`--no-replay` 关掉）：

    rec = Recorder(every=10)         # 每 10 个物理步记一帧 = 20 ms
    robot.on_step = rec
    ...跑...
    rec.save("runs/replay.npz")
    rec.play(model)                  # 在窗口里按 ~1x 播，1–9 实时改倍速
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np


class Recorder:
    def __init__(self, every: int = 4, limit: int = 120_000):
        self.every = max(1, int(every))
        self.limit = int(limit)
        self.frames: list[np.ndarray] = []
        self.count = 0

    # ---------------------------------------------------------------- capture
    def __call__(self, robot):
        self.count += 1
        if self.count % self.every:
            return
        if len(self.frames) >= self.limit:
            return
        self.frames.append(robot.d.qpos.copy())

    def __len__(self):
        return len(self.frames)

    @property
    def seconds(self):
        """录到的仿真时长（秒）。"""
        return len(self.frames) * self.every * 0.002

    # ------------------------------------------------------------------ files
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, frames=np.array(self.frames), every=self.every)
        return path

    @classmethod
    def load(cls, path):
        z = np.load(Path(path))
        rec = cls(every=int(z["every"]))
        rec.frames = [row.copy() for row in z["frames"]]
        return rec

    # ----------------------------------------------------------------- replay
    def play(self, model, *, speed=1.0, loop=True, start_pause=0.0, camera=None):
        """在 MuJoCo 窗口里回放。

        `speed` = 相对仿真时间的倍速（1.0 = 实时）。播放时可以按数字键实时改：
        1–9 直接设成 1×..9×，`0` = 0.5×，`+`/`-` 微调。

        `camera` 给了名字才会另开一个相机窗口（默认不开：多一个 GL 窗口很吃资源，
        实测会把整机拖卡）。想看相机画面，用 `--cam-window`，或者直接看流水线
        存下来的 runs/raw_annotated.png 之类。
        """
        import msvcrt
        import mujoco
        import mujoco.viewer

        if not self.frames:
            print("[replay] 没有录到任何帧")
            return

        d = mujoco.MjData(model)
        viewer = mujoco.viewer.launch_passive(model, d)
        viewer.cam.lookat[:] = (0.33, 0.40, 0.20)
        viewer.cam.distance = 1.35
        viewer.cam.azimuth = 130
        viewer.cam.elevation = -18

        # 只有明确要求才开第二个窗口
        cam_view = None
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera or "")
        if camera and cid >= 0:
            cam_view = mujoco.viewer.launch_passive(model, d)
            cam_view.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam_view.cam.fixedcamid = cid
        elif camera:
            print(f"[replay] 场景里没有相机 {camera!r}，只显示 3D 视图")

        frame_dt = self.every * model.opt.timestep
        print(f"[replay] {len(self.frames)} 帧 / {self.seconds:.1f} s 仿真时间，"
              f"每帧 {frame_dt*1000:.0f} ms，按 {speed:g}× 播放")
        print("[replay] 改倍速：点一下控制台按 1–9（0 = 0.5×，+/- 微调）；"
              "关掉窗口即结束")

        def poll_keys():
            nonlocal speed
            while msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in "123456789":
                    speed = float(ch)
                elif ch == "0":
                    speed = 0.5
                elif ch in "+=":
                    speed = min(speed + 1.0, 32.0)
                elif ch in "-_":
                    speed = max(speed - 1.0, 0.1)
                else:
                    continue
                print(f"[replay] 倍速 -> {speed:g}×")

        if start_pause:
            time.sleep(start_pause)
        try:
            while viewer.is_running():
                t0 = time.perf_counter()
                for q in self.frames:
                    if not viewer.is_running():
                        break
                    try:
                        poll_keys()
                    except Exception:            # noqa: BLE001
                        pass
                    d.qpos[:] = q
                    mujoco.mj_forward(model, d)
                    viewer.sync()
                    if cam_view is not None and cam_view.is_running():
                        cam_view.sync()
                    slack = frame_dt / max(speed, 1e-6) - (time.perf_counter() - t0)
                    if slack > 0:
                        time.sleep(slack)
                    t0 = time.perf_counter()
                if not loop:
                    break
                time.sleep(0.6)          # 每圈停一下，看得出这是重播
        except KeyboardInterrupt:
            pass
        finally:
            for v in (viewer, cam_view):
                if v is None:
                    continue
                try:
                    v.close()
                except Exception:            # noqa: BLE001
                    pass
