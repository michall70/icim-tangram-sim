"""RealSense D435i 相机封装：深度-彩色对齐采集。

默认全部参数为 None，即自动曝光/自动白平衡；但本项目在 `config.CAMERA_PARAMS`
中显式给出固定曝光 312、增益 0、白平衡 3300（2026-10-31 现场锁定），
因为自动模式会让同一块塑料在不同时刻呈现不同 HSV，颜色阈值随之漂移。

深度流会与彩色流对齐后一并返回，但当前主流程只使用彩色图；深度留作后续
"吸空检测 / 高度修正"的接口，未参与现有规划。
"""

import cv2
import numpy as np
import pyrealsense2 as rs


def usb_descriptor():
    """当前相机的 USB 链路描述（如 '3.2' / '2.1'）；没有相机时返回 None。"""
    try:
        devices = rs.context().query_devices()
        if len(devices) == 0:
            return None
        return str(devices[0].get_info(rs.camera_info.usb_type_descriptor))
    except Exception:
        return None


def stream_plan(usb, fps=30, depth=True):
    """按 USB 链路挑流配置，返回 (fps, depth)。

    - USB 3.x：用调用方给的值（本项目 1280×720 / 30fps / 深度+彩色双流）；
    - USB 2.x：1280×720 的彩色只有 6/10/15fps，且带不动双流
      （同时请求会 `RuntimeError: Couldn't resolve requests`）
      ⇒ 自动降到 **15fps + 只开彩色流**（主流程只用彩色图，功能无损）。
    """
    if usb and str(usb)[0] in ("1", "2"):
        return min(int(fps), 15), False
    return int(fps), bool(depth)


class RealSenseCamera:
    """深度流对齐到彩色流的 RealSense 相机。

    默认全自动（自动曝光 + 自动白平衡）。传 exposure（非 None）则关闭
    自动曝光改用固定值，传 white_balance（非 None）则锁定白平衡；
    gain/brightness/contrast 仅在显式传入时写入。

    `auto_usb=True`（默认）时，开流前会读 USB 链路并按 `stream_plan` 自动选流：
    USB2 降到 15fps 单彩色流，USB3 用调用方给的全速双流。
    """

    def __init__(self, width=1280, height=720, fps=30,
                 exposure=None, gain=None, brightness=None, contrast=None,
                 white_balance=None, depth=True, auto_usb=True):
        self.width = width
        self.height = height
        self.fps = fps
        self.exposure = exposure
        self.gain = gain
        self.brightness = brightness
        self.contrast = contrast
        self.white_balance = white_balance
        self.depth = depth            # 期望值；USB2 下会被自动关掉
        self.auto_usb = auto_usb
        self.usb = None               # 实际链路描述，open() 后填充
        self.active_fps = None
        self.active_depth = None
        self.pipeline = None
        self.align = None
        self.profile = None

    def open(self):
        """启动采集流水线，写入固定曝光/白平衡（传 None 的项交给相机自动）。"""
        fps, use_depth = self.fps, self.depth
        if self.auto_usb:
            self.usb = usb_descriptor()
            if self.usb is None:
                raise RuntimeError("没有枚举到 RealSense 相机：检查 USB 线/口，或是否被其他程序占用")
            fps, use_depth = stream_plan(self.usb, self.fps, self.depth)
        self.active_fps, self.active_depth = fps, use_depth
        if self.usb:
            print("[camera] USB %s → %dfps%s" % (self.usb, fps, "" if use_depth else "（仅彩色流）"))

        self.pipeline = rs.pipeline()
        config = rs.config()
        if use_depth:
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(config)

        color_sensor = self.profile.get_device().first_color_sensor()
        if self.exposure is None:
            color_sensor.set_option(rs.option.enable_auto_exposure, 1)
        else:
            color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            self._apply_exposure(color_sensor, self.exposure)
        if self.white_balance is None:
            color_sensor.set_option(rs.option.enable_auto_white_balance, 1)
        else:
            color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
            color_sensor.set_option(rs.option.white_balance, self.white_balance)
        if self.brightness is not None:
            color_sensor.set_option(rs.option.brightness, self.brightness)
        if self.contrast is not None:
            color_sensor.set_option(rs.option.contrast, self.contrast)
        if self.gain is not None:
            color_sensor.set_option(rs.option.gain, self.gain)

        self.align = rs.align(rs.stream.color) if self.active_depth else None

    def _apply_exposure(self, sensor, value):
        """写曝光并**读回校验**；写不进去时重试，最后仍不一致就明确告警。

        为什么要读回：D435i 的 RGB 传感器在关掉自动曝光后，**紧接着**写 exposure
        有时不生效（驱动还停在自动曝光算出来的值上，例如 625）。旧代码写完就不再过问，
        于是"config 改了 446、相机仍是 625"却毫无提示。
        这里改成：写 → 读回 → 不一致就抓一帧再写，最多 3 次；仍不一致则打印真实值与差值。
        """
        last = None
        for attempt in range(3):
            try:
                sensor.set_option(rs.option.exposure, float(value))
            except Exception as exc:
                print("[camera] 写曝光失败（第 %d 次）：%s" % (attempt + 1, exc))
            try:
                last = float(sensor.get_option(rs.option.exposure))
            except Exception:
                last = None
            if last is not None and abs(last - float(value)) <= 1.0:
                print("[camera] 曝光 = %.0f（已生效）" % last)
                return True
            # 让驱动真正提交一次状态：抓一帧再重写
            try:
                self.pipeline.wait_for_frames()
            except Exception:
                pass
        print("[camera] [警告] 曝光写入未生效：请求 %.0f，实际仍为 %s"
              % (float(value), "%.0f" % last if last is not None else "未知"))
        print("[camera]        原因通常是该传感器在关自动曝光后需要先出一帧再写；"
              "也可能被同一台相机的其它程序覆盖（关掉 Viewer/其它 python 再试）")
        return False

    def close(self):
        if self.pipeline is not None:
            # open() 里 start() 失败时 pipeline 还没起来，stop() 会抛
            # "stop() cannot be called before start()"，把真实错误盖掉，这里吞掉即可。
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None

    def get_aligned_frames(self):
        """采集一帧并做深度-彩色对齐，返回 (color_img, depth_img, color_intrinsics)。

        只开彩色流时 `depth_img` 为 None。
        """
        if self.align is None:
            frames = self.pipeline.wait_for_frames()
        else:
            frames = self.align.process(self.pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame() if self.active_depth else None
        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        color_img = np.asanyarray(color_frame.get_data()).copy()
        depth_img = np.asanyarray(depth_frame.get_data()).copy() if depth_frame else None
        return color_img, depth_img, intrinsics

    def grab_color(self, warmup=10):
        """采集一帧彩色图；warmup>0 时先丢弃若干帧等待曝光稳定。"""
        for _ in range(warmup):
            self.pipeline.wait_for_frames()
        color_img, _, _ = self.get_aligned_frames()
        return color_img

    def print_current_settings(self):
        """打印当前相机参数，用于调试。"""
        color_sensor = self.profile.get_device().first_color_sensor()
        print("当前相机设置:")
        print(f"自动曝光: {color_sensor.get_option(rs.option.enable_auto_exposure)}")
        print(f"曝光值: {color_sensor.get_option(rs.option.exposure)}")
        print(f"自动白平衡: {color_sensor.get_option(rs.option.enable_auto_white_balance)}")
        print(f"白平衡: {color_sensor.get_option(rs.option.white_balance)}")
        print(f"亮度: {color_sensor.get_option(rs.option.brightness)}")
        print(f"对比度: {color_sensor.get_option(rs.option.contrast)}")
        print(f"增益: {color_sensor.get_option(rs.option.gain)}")

    @staticmethod
    def colorize_depth(depth_img, alpha=0.03):
        """深度图伪彩化，用于可视化。"""
        return cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=alpha), cv2.COLORMAP_JET)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
