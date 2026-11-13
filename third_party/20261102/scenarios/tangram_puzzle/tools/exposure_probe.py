"""曝光探针：把"设置 → 读回"的每一步都打出来，用来判断写入到底有没有生效。

    python -B -m scenarios.tangram_puzzle.tools.exposure_probe
    python -B -m scenarios.tangram_puzzle.tools.exposure_probe --value 446

输出会逐步显示：

1. 设备链路、exposure 选项的 min/max/step
2. 打开前的值（通常等于上一次程序留下的值，例如 625）
3. 关自动曝光后的值
4. 写入目标值后**立即**读回
5. 抓 3 帧后读回（关键：有些传感器要出帧才提交）
6. 再写一次后读回
7. 结论

**只读之外只做一件事：改曝光**（不改白平衡/增益，不动机器人）。
"""

import argparse

import pyrealsense2 as rs


def main(argv=None):
    parser = argparse.ArgumentParser(description="曝光写入探针（会临时改曝光）")
    parser.add_argument("--value", type=float, default=None,
                        help="要测试的曝光值；默认取 config.CAMERA_PARAMS['exposure']")
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args(argv)

    value = args.value
    if value is None:
        from scenarios.tangram_puzzle.config import CAMERA_PARAMS
        value = float(CAMERA_PARAMS.get("exposure") or 400)
        print("config 里的曝光值 = %.1f" % value)

    devices = rs.context().query_devices()
    print("设备数 %d" % len(devices))
    if not len(devices):
        print("没有枚举到相机：检查 USB 线/口，或是否被其它程序占用")
        return 1
    device = devices[0]
    print("链路 %s" % device.get_info(rs.camera_info.usb_type_descriptor))

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, args.fps)
    profile = pipeline.start(config)
    try:
        sensor = profile.get_device().first_color_sensor()
        rng = sensor.get_option_range(rs.option.exposure)
        print("exposure 选项范围 %.0f—%.0f，步长 %.0f" % (rng.min, rng.max, rng.step))
        print("① 打开后：auto=%s exposure=%.0f"
              % (sensor.get_option(rs.option.enable_auto_exposure),
                 sensor.get_option(rs.option.exposure)))

        sensor.set_option(rs.option.enable_auto_exposure, 0)
        print("② 关自动曝光后：auto=%s exposure=%.0f"
              % (sensor.get_option(rs.option.enable_auto_exposure),
                 sensor.get_option(rs.option.exposure)))

        sensor.set_option(rs.option.exposure, float(value))
        print("③ 写入 %.0f 后立即读：%.0f" % (value, sensor.get_option(rs.option.exposure)))

        for _ in range(3):
            pipeline.wait_for_frames()
        print("④ 抓 3 帧后读：%.0f" % sensor.get_option(rs.option.exposure))

        sensor.set_option(rs.option.exposure, float(value))
        after = float(sensor.get_option(rs.option.exposure))
        print("⑤ 再写一次后读：%.0f" % after)

        if abs(after - value) <= 1.0:
            print("\n结论：能写进去。若程序里仍显示旧值，多半是 config 没保存或跑的是另一份代码。")
        else:
            print("\n结论：**写不进去**。写入 %.0f，实际 %.0f。" % (value, after))
            print("常见原因：")
            print("  · 这台相机被另一个程序（RealSense Viewer / 其它 python）同时占着，它在反复覆盖")
            print("  · 该分辨率/帧率下曝光被驱动限幅（试着把 fps 从 15 改回 30，或降分辨率）")
            print("  · 传感器需要先出帧才提交（本探针第 ④ ⑤ 步已试过）")
        return 0
    finally:
        pipeline.stop()


if __name__ == "__main__":
    raise SystemExit(main())
