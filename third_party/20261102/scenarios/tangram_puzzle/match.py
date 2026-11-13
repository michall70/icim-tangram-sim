"""一条命令跑完整套：目标图进 → 计划/执行出。

比赛当天只有两步（其余全部提前做完）：

    # ① 拿到目标图后，先花 20 秒看它能不能用（不动机器人）
    python -B -m scenarios.tangram_puzzle.match --target "F:\tangram_03.png"

    # ② 确认没问题就开跑（会连机器人、自己拍照、逐块抓放）
    python -B -m scenarios.tangram_puzzle.match --target "F:\tangram_03.png" --execute

赛前准备（开赛前一次性做完，不占比赛时间）：

    python -B -m scenarios.tangram_puzzle.match --prep

它会把"相机 / 曝光 / 原料框 / 锚点 / 目标图解析 / 七块检测"逐项体检，
每项给 PASS / WARN / FAIL，并告诉你差哪一步。

## 为什么能压缩

- **和"目标图"无关的都能提前做**：相机曝光、原料框 ROI、托盘锚点、吸盘偏心；
- 比赛当天只剩"换一张目标图"这一件新事，所以就是一条命令。

设计中刻意不做的事：不在比赛当天重标定、不重画 ROI、不重新标偏心。
那些都应在 ``--prep`` 阶段完成并写进 config。
"""

import argparse
import os
import sys
import time

import cv2 as cv
import numpy as np

from scenarios.tangram_puzzle.config import (CAMERA_MATRIX, CAMERA_PARAMS, DETECT_PARAMS,
                                             DIST_COEFFS, HOMOGRAPHY,
                                             TARGET_AREA_QUAD_MM,
                                             TARGET_IMAGE_PATH, TARGET_SOURCE)

CONFIG_HINTS = {
    "touches ROI boundary": "块压到原料框边了：把块往里挪 10mm 以上，或确认 raw_roi 是否该重画",
    "multiple substantial candidates": "某块被高光劈成两块：降曝光（over% < 1%）或换摆放角度",
    "poor polygon fit": "某块掩模残缺：同样先降曝光/柔光",
    "检测未通过": "先看 health_check 的逐色掩模，确认是缺块还是切边",
    "放不进成品区": "换朝向 yaw=90 试一次，或换更小的目标图",
    "IKPreflight": "有抓取点超出臂展：把黑板朝机器人推 8—10cm，或挪最远那两块",
    "Couldn't resolve requests": "相机链路问题：确认 usb 是 2.1/3.x，且没被 Viewer 占用",
    "No device connected": "相机没枚举到：重插、换口、关掉占用程序",
}


def _hint(message):
    for key, hint in CONFIG_HINTS.items():
        if key.lower() in str(message).lower():
            return hint
    return None


def prep(argv=None):
    """赛前一次性体检：不动机器人；只开相机看一眼、解析一次目标图。"""
    print("=" * 66)
    print("赛前体检（一次性，不占比赛时间）")
    print("=" * 66)
    results = []

    def record(name, ok, detail):
        results.append((("PASS" if ok is True else ("WARN" if ok is None else "FAIL")), name, detail))

    # 1) 目标图解析（用 config 里那张当样本）
    try:
        from scenarios.tangram_puzzle.planning import target_image as ti
        placements, summary = ti.load_target(TARGET_IMAGE_PATH)
        kinds = summary.get("kinds")
        total = float(summary.get("total_area", 0))
        ok = len(placements) == 7 and abs(total - 1.0) < 1e-3
        record("目标图解析", ok, "%d 块，类别 %s，面积和 %.6f（样本：%s）"
               % (len(placements), kinds, total, os.path.basename(TARGET_IMAGE_PATH)))
    except Exception as exc:
        record("目标图解析", False, str(exc))

    # 2) 成品区（四角优先级最高）
    try:
        from scenarios.tangram_puzzle.planning import target_image as ti
        area_quad = ti.quad_from_config(TARGET_AREA_QUAD_MM)
    except Exception:
        area_quad = None
    if area_quad is not None:
        center = ti.quad_center(area_quad)
        sides = [float(np.linalg.norm(area_quad[(i + 1) % 4] - area_quad[i])) for i in range(4)]
        record("成品区四角", True,
               "中心 (%.1f, %.1f) mm，边长 %.0f/%.0f/%.0f/%.0f mm，朝向 %.1f°"
               % (center[0], center[1], sides[0], sides[1], sides[2], sides[3],
                  ti.quad_yaw_deg(area_quad)))
    else:
        record("成品区四角", None,
               "未填 TARGET_AREA_QUAD_MM，退回 template_roi 像素框；成品区不在画面时会失效")

    # 3) 相机 + 曝光
    try:
        from common.realsense_camera import RealSenseCamera
        from common.calibration import CameraMapper
        camera = RealSenseCamera(**CAMERA_PARAMS)
        camera.open()
        try:
            frame = camera.grab_color(warmup=20)
        finally:
            camera.close()
        mapper = CameraMapper(CAMERA_MATRIX, DIST_COEFFS, HOMOGRAPHY)
        image = mapper.undistort(frame)
        x, y, w, h = DETECT_PARAMS["raw_roi"]
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        patch = hsv[y:y + h, x:x + w, 2]
        vmean = float(patch.mean())
        over = float((patch >= 250).mean() * 100)
        ok = (over < 1.0) and (50 <= vmean <= 110)
        record("相机与曝光", ok if ok else None,
               "链路 %s，V均值 %.0f，过曝 %.2f%%（目标：V 50—110、过曝 <1%%）"
               % (camera.usb or "?", vmean, over))
        detect_ok = None
        try:
            from scenarios.tangram_puzzle.tangram_detector import TangramDetector
            pieces = TangramDetector(DETECT_PARAMS).detect_raw_material(image)
            detect_ok = (len(pieces) == 7) if len(pieces) else None
            record("原料七块", detect_ok,
                   "检出 %d/7%s" % (len(pieces), "" if len(pieces) == 7 else "（空板时正常）"))
        except Exception as exc:
            record("原料七块", None, str(exc))
    except Exception as exc:
        record("相机与曝光", False, str(exc))
        hint = _hint(exc)
        if hint:
            print("  → %s" % hint)

    print()
    for state, name, detail in results:
        mark = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[state]
        print("%-7s %-10s %s" % (mark, name, detail))
    fails = [r for r in results if r[0] == "FAIL"]
    warns = [r for r in results if r[0] == "WARN"]
    print()
    if fails:
        print("体检未通过 %d 项，先修好再比赛（详见上面 FAIL 行）" % len(fails))
        return 1
    if warns:
        print("体检通过，但有 %d 项提醒（不影响开跑）" % len(warns))
        return 0
    print("体检全部通过，比赛当天只需一条命令：")
    print("  python -B -m scenarios.tangram_puzzle.match --target <目标图> --execute")
    return 0


def run_match(target, frame, execute, yaw):
    """目标图 → 计划/执行。

    - 离线（无 --execute）：`--frame` 是"要分析的原料帧"；
    - 实机（--execute/--go）：
        · 给了 `--frame` → **方案 A**：复用这张帧（必须是在拍照位拍的），不现场抓帧；
        · 没给          → **方案 B**：开相机现场抓一帧。
    """
    import scenarios.tangram_puzzle.main as main_mod

    if yaw is not None:
        main_mod.TARGET_YAW_DEG = float(yaw)
    started = time.time()
    if execute and frame:
        print("[match] 方案 A：复用已有帧 %s（不现场抓帧）" % os.path.basename(frame))
    elif execute:
        print("[match] 方案 B：现场抓帧")
    try:
        if execute:
            main_mod.run(execute=True, target_source="image", target_image_path=target,
                         use_frame=frame)
        else:
            main_mod.run(raw_image_path=frame, execute=False,
                         target_source="image", target_image_path=target)
    except Exception as exc:
        print("\n" + "=" * 66)
        print("失败：%s" % exc)
        hint = _hint(exc)
        if hint:
            print("建议：%s" % hint)
        print("用时 %.1f 秒" % (time.time() - started))
        return 1
    print("\n" + "=" * 66)
    print("完成，用时 %.1f 秒" % (time.time() - started))
    if not execute:
        print("这是离线结果；确认无误后加 --execute 实机运行。")
    return 0


def find_target(search_dirs=("F:\\", "F:\\images", "F:\\image"), exts=(".png", ".jpg", ".jpeg", ".bmp")):
    """在 F 盘找图片，按文件名排序返回 (默认选中项, 全部候选)。

    按**文件名**排序（比赛常常是 tangram_01…10），而不是按修改时间——
    U 盘里可能一次给了十几张，必须让人挑，不能替人选。
    """
    candidates = []
    for root in search_dirs:
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if os.path.isfile(path) and name.lower().endswith(exts):
                candidates.append(path)
    candidates.sort(key=lambda p: os.path.basename(p).lower())
    return (candidates[0] if candidates else None), candidates


def choose_target(candidates, ask=input):
    """列出全部候选，让人输编号；回车=第 1 个。返回选中的路径。"""
    print("U 盘里找到 %d 张图片：" % len(candidates))
    for index, path in enumerate(candidates, 1):
        try:
            stamp = time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(path)))
        except OSError:
            stamp = "-"
        print("  [%2d] %-28s %s" % (index, os.path.basename(path), stamp))
    while True:
        try:
            answer = str(ask("输入编号后回车（直接回车 = 1）：")).strip()
        except EOFError:                     # 非交互环境：退回第一张
            print("\n（没有交互输入，默认选第 1 张）")
            return candidates[0]
        if not answer:
            return candidates[0]
        if answer.isdigit() and 1 <= int(answer) <= len(candidates):
            return candidates[int(answer) - 1]
        print("  请输入 1—%d 之间的编号" % len(candidates))


def main(argv=None):
    parser = argparse.ArgumentParser(description="一条命令跑完七巧板（目标图 → 计划/执行）")
    parser.add_argument("--target", help="直接指定目标图；不给则列出 U 盘图片让你选")
    parser.add_argument("--frame",
                        help="原料帧：离线时是要分析的图；实机执行时=复用这张帧（方案 A）")
    parser.add_argument("--execute", action="store_true", help="实机执行（会动机器人）")
    parser.add_argument("--go", action="store_true",
                        help="列出 U 盘图片 → 选一张 → 实机执行（比赛当天用这个）")
    parser.add_argument("--yaw", type=float, default=None, help="强制图案朝向（度）")
    parser.add_argument("--prep", action="store_true", help="赛前一次性体检")
    args = parser.parse_args(argv)

    if args.prep:
        return prep()

    target = args.target
    if not target:
        default, candidates = find_target()
        if not candidates:
            print("没在 F 盘找到图片（.png/.jpg/.bmp）。")
            print("确认 U 盘插好，或直接指定：--target \"F:\\文件名.png\"")
            return 1
        if len(candidates) == 1:
            target = candidates[0]
            print("U 盘里只有一张图片，直接用：%s" % os.path.basename(target))
        else:
            target = choose_target(candidates)
            print("已选择：%s" % os.path.basename(target))

    if not os.path.isfile(target):
        print("目标图不存在：%s" % target)
        return 1
    execute = args.execute or args.go
    if execute and args.frame and not os.path.isfile(args.frame):
        print("--frame 指定的帧不存在：%s" % args.frame)
        return 1
    if TARGET_SOURCE != "image":
        print("提示：config.TARGET_SOURCE=%r，本命令强制走目标图通道" % TARGET_SOURCE)
    return run_match(target, args.frame, execute, args.yaw)


if __name__ == "__main__":
    raise SystemExit(main())
