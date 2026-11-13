"""相机拍照参数调试：实时取景显示，键盘调曝光/增益/白平衡，保存图像核对。

用途：工位无额外打光光源，不同时段光照不同，固定参数需按当前环境整定。
用本工具实时调参目视确认亮度/颜色合适后，把参数写回 config.py 的
CAMERA_PARAMS（保存时会打印可直接粘贴的参数行）。

用法（项目根目录执行）:
    python -m scenarios.tangram_puzzle.capture_check

按键:
    + / -   曝光 ±5        g / G   增益 ±16       w / W   白平衡 ±200
    s       保存当前帧（文件名含参数）        q / ESC  退出
图像顶栏实时显示亮度统计（V 均值 / 过曝% / 欠曝%），调到亮度适中、无大面积
过曝欠曝、块颜色与实物一致即可。
"""

import datetime
import os

import cv2 as cv
import numpy as np
import pyrealsense2 as rs

from common.realsense_camera import stream_plan, usb_descriptor
from scenarios.tangram_puzzle.config import CAMERA_PARAMS

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "capture_check")

STEP_EXPOSURE = 5.0
STEP_GAIN = 16.0
STEP_WB = 200.0
WARMUP_FRAMES = 10


def clamp(sensor, option, value):
    """把参数值限制在相机允许范围内。"""
    r = sensor.get_option_range(option)
    return max(r.min, min(r.max, value))


def main():
    p = dict(CAMERA_PARAMS)
    width = p.get("width", 1280)
    height = p.get("height", 720)
    fps = p.get("fps", 30)
    # 与 RealSenseCamera 用同一套策略：USB2 下 1280×720 彩色最高 15fps，
    # 直接请求 30 会 RuntimeError: Couldn't resolve requests。
    usb = usb_descriptor()
    fps, _ = stream_plan(usb, fps, False)
    exposure = p.get("exposure") if p.get("exposure") is not None else 25.0
    gain = p.get("gain") if p.get("gain") is not None else 64.0
    wb = p.get("white_balance") if p.get("white_balance") is not None else 4600.0
    # brightness/contrast 也必须像 RealSenseCamera 那样写进相机，
    # 否则 config 里改了它们，这里的预览却看不到效果（两边不一致）。
    brightness = p.get("brightness")
    contrast = p.get("contrast")

    print("[capture_check] 链路 %s → %dfps" % (usb or "未知", fps))

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(config)
    sensor = profile.get_device().first_color_sensor()

    def apply_params():
        sensor.set_option(rs.option.enable_auto_exposure, 0)
        sensor.set_option(rs.option.enable_auto_white_balance, 0)
        sensor.set_option(rs.option.exposure, clamp(sensor, rs.option.exposure, exposure))
        sensor.set_option(rs.option.white_balance, clamp(sensor, rs.option.white_balance, wb))
        sensor.set_option(rs.option.gain, clamp(sensor, rs.option.gain, gain))
        # 与 RealSenseCamera.open() 保持一致：给了才写，没给就保持相机默认
        if brightness is not None:
            sensor.set_option(rs.option.brightness, clamp(sensor, rs.option.brightness, brightness))
        if contrast is not None:
            sensor.set_option(rs.option.contrast, clamp(sensor, rs.option.contrast, contrast))

    apply_params()
    print("[capture_check] 按键: +/- 曝光  g/G 增益  w/W 白平衡  s 保存  q/ESC 退出")

    for _ in range(WARMUP_FRAMES):
        pipeline.wait_for_frames()

    win = "capture_check"
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    frame_idx = 0
    stats = (0.0, 0.0, 0.0)
    try:
        while True:
            frame = np.asanyarray(
                pipeline.wait_for_frames().get_color_frame().get_data()).copy()
            frame_idx += 1
            if frame_idx % 15 == 1:
                v = cv.cvtColor(frame, cv.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
                stats = (float(v.mean()),
                         float((v >= 250).mean() * 100),
                         float((v <= 30).mean() * 100))

            vis = frame.copy()
            cv.rectangle(vis, (0, 0), (width, 46), (0, 0, 0), -1)
            bar = (f"exposure={exposure:.0f} gain={gain:.0f} wb={wb:.0f} | "
                   f"Vmean={stats[0]:.0f} over={stats[1]:.1f}% dark={stats[2]:.1f}%")
            cv.putText(vis, bar, (12, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv.putText(vis, "+/-:exp g/G:gain w/W:wb s:save q:quit",
                       (12, height - 12), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv.imshow(win, vis)

            key = cv.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            changed = False
            if key in (ord("+"), ord("=")):
                exposure = clamp(sensor, rs.option.exposure, exposure + STEP_EXPOSURE)
                changed = True
            elif key == ord("-"):
                exposure = clamp(sensor, rs.option.exposure, exposure - STEP_EXPOSURE)
                changed = True
            elif key == ord("g"):
                gain = clamp(sensor, rs.option.gain, gain + STEP_GAIN)
                changed = True
            elif key == ord("G"):
                gain = clamp(sensor, rs.option.gain, gain - STEP_GAIN)
                changed = True
            elif key == ord("w"):
                wb = clamp(sensor, rs.option.white_balance, wb + STEP_WB)
                changed = True
            elif key == ord("W"):
                wb = clamp(sensor, rs.option.white_balance, wb - STEP_WB)
                changed = True
            elif key == ord("s"):
                os.makedirs(OUT_DIR, exist_ok=True)
                stamp = datetime.datetime.now().strftime("%H%M%S")
                out = os.path.join(
                    OUT_DIR, f"exp{exposure:.0f}_gain{gain:.0f}_wb{wb:.0f}_{stamp}.png")
                cv.imwrite(out, frame)
                print(f"[capture_check] 已保存: {out}")
                print("                 写回 config.py CAMERA_PARAMS: "
                      f"exposure={exposure}, gain={gain}, "
                      f"brightness={p.get('brightness', 0.0)}, "
                      f"contrast={p.get('contrast', 50.0)}, white_balance={wb}")
            if changed:
                apply_params()
                frame_idx = 0  # 参数已变，下一帧重新统计亮度
    finally:
        cv.destroyAllWindows()
        pipeline.stop()


if __name__ == "__main__":
    main()
