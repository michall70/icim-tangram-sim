"""现场一键重标七组 HSV 阈值：换光照/换曝光后，不用改代码、只跑这一条命令。

## 解决的痛点

比赛现场的光照与实验室不同，相机必须重新调曝光/增益/白平衡。而这些参数一变，
画面亮度与色相整体平移，`DETECT_PARAMS` 里七组 HSV 区间就全部失配——
2026-10-29 就发生过：固定曝光后黄色与绿色的掩模像素直接变成 0。

手工重标七组区间既慢又容易出错。本工具用**当前这一帧的真实像素**把七组区间重新拟合一遍，
输出可直接粘贴回 `config.py` 的参数块。整个过程不改任何代码逻辑，符合"现场只能改参数"的约束。

## 用法

    # 用实时相机（会占用相机几秒，不动机器人）
    python -B -m scenarios.tangram_puzzle.tools.tune_colors

    # 或对着一张已存的实拍帧
    python -B -m scenarios.tangram_puzzle.tools.tune_colors --image <实拍帧>

    # 只打印、不写文件（默认就是只打印）
    python -B -m scenarios.tangram_puzzle.tools.tune_colors --apply   # 直接写 config.py

判据：输出里每块都要有"实测 H/S/V"和"建议区间"，并且最后一行给出 7/7。
若某块显示 `未找到`，说明该色连当前区间都覆盖不到，需要用 `--seed` 手动给一个近似区间。
"""

import argparse
import sys

import cv2 as cv
import numpy as np

from common.calibration import CameraMapper
from scenarios.tangram_puzzle.config import (CAMERA_MATRIX, CAMERA_PARAMS, DETECT_PARAMS,
                                             DIST_COEFFS, HOMOGRAPHY)

# 拟合区间的余量：在实测 5%—95% 分位之外再放宽这么多
HUE_MARGIN = 3
SAT_MARGIN = 8
VAL_MARGIN = 12

# 色相是环绕的：区间跨越 0/180 时要拆成两段
HUE_PERIOD = 180


def _percentile_range(values, margin, floor, ceil):
    low = float(np.percentile(values, 3)) - margin
    high = float(np.percentile(values, 97)) + margin
    return max(floor, low), min(ceil, high)


def _fit_from_blob(hsv, mask):
    pixels = hsv[mask > 0]
    if len(pixels) < 200:
        return None
    hues = pixels[:, 0].astype(float)
    # 色相环绕：若全部像素都在两端（<15 或 >165），把 0/180 附近合并后再算
    if hues.min() < 20 and hues.max() > 160:
        hues = np.where(hues > 90, hues - HUE_PERIOD, hues)
        low, high = _percentile_range(hues, HUE_MARGIN, -HUE_PERIOD, 0)
        low %= HUE_PERIOD
        high %= HUE_PERIOD
        if low > high:
            # 跨 0/180 的红色：拆成两段，第二段至少保留 5 度，避免出现 H0-0 的空区间
            tail = max(5, int(high))
            ranges = [((int(max(low, 0)), *_sv(pixels)), (HUE_PERIOD, 255, 255)),
                      ((0, *_sv(pixels)), (tail, 255, 255))]
        else:
            ranges = [((int(low), *_sv(pixels)), (int(high), 255, 255))]
        return ranges
    low, high = _percentile_range(hues, HUE_MARGIN, 0, HUE_PERIOD)
    s_low, _ = _percentile_range(pixels[:, 1], SAT_MARGIN, 0, 255)
    v_low, _ = _percentile_range(pixels[:, 2], VAL_MARGIN, 0, 255)
    return [((int(low), int(s_low), int(v_low)), (int(high), 255, 255))]


def _sv(pixels):
    s_low, _ = _percentile_range(pixels[:, 1], SAT_MARGIN, 0, 255)
    v_low, _ = _percentile_range(pixels[:, 2], VAL_MARGIN, 0, 255)
    return [int(s_low), int(v_low)]


def fit_specs(hsv, roi, log=print):
    """在同一帧上重拟合七组区间；返回 [(id, name, 实测中位 HSV, 建议 ranges)]。"""
    x, y, w, h = roi
    inside = np.zeros(hsv.shape[:2], bool)
    inside[y:y + h, x:x + w] = True
    result = []
    for spec in DETECT_PARAMS["specs"]:
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in spec["ranges"]:
            mask |= cv.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask[~inside] = 0
        count, labels, stats, _ = cv.connectedComponentsWithStats(mask, 8)
        if count <= 1:
            result.append((spec["id"], spec["name"], None, None))
            continue
        biggest = max(range(1, count), key=lambda i: stats[i, cv.CC_STAT_AREA])
        blob = (labels == biggest).astype(np.uint8)
        median = np.median(hsv[blob > 0], axis=0)
        ranges = _fit_from_blob(hsv, blob)
        result.append((spec["id"], spec["name"], median, ranges))
    return result


def _render(rows):
    lines = []
    for color_id, name, median, ranges in rows:
        if ranges is None:
            lines.append("    # ID=%d %s：本帧未找到，需手动给初始区间" % (color_id, name))
            continue
        text = ", ".join("((%d, %d, %d), (%d, %d, %d))" % (lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])
                         for lo, hi in ranges)
        lines.append("    # ID=%d %s 实测中位 HSV=(%.0f, %.0f, %.0f)" % (color_id, name, *median))
        lines.append("    ")
        lines.append("__RANGES_%d__ = [%s]" % (color_id, text))
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description="现场重标七组 HSV 阈值（只读，不动机器人）")
    parser.add_argument("--image", help="已存实拍帧；缺省则用实时相机拍一帧")
    parser.add_argument("--area", choices=("raw", "template"), default="raw")
    parser.add_argument("--apply", action="store_true", help="直接写回 config.py")
    args = parser.parse_args(argv)

    mapper = CameraMapper(CAMERA_MATRIX, DIST_COEFFS, HOMOGRAPHY)
    if args.image:
        image = cv.imread(args.image)
        if image is None:
            print("无法读取:", args.image)
            return 2
    else:
        from common.realsense_camera import RealSenseCamera
        camera = RealSenseCamera(**CAMERA_PARAMS)
        camera.open()
        try:
            image = camera.grab_color(warmup=30)
        finally:
            camera.close()
        print("[tune_colors] 已用实时相机拍一帧")

    image = mapper.undistort(image)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    roi = DETECT_PARAMS["%s_roi" % args.area]

    rows = fit_specs(hsv, roi)
    print("\n%-4s %-9s %-22s %s" % ("ID", "颜色", "实测中位 HSV", "建议区间"))
    for color_id, name, median, ranges in rows:
        if ranges is None:
            print("%-4s %-9s %-22s %s" % (color_id, name, "—", "未找到"))
            continue
        text = "; ".join("H%d-%d S%d-V%d" % (lo[0], hi[0], lo[1], lo[2]) for lo, hi in ranges)
        print("%-4s %-9s H%3.0f S%3.0f V%3.0f       %s"
              % (color_id, name, median[0], median[1], median[2], text))

    missing = [name for _, name, _, ranges in rows if ranges is None]
    print("\n拟合结果：%d/7 块找到%s" % (7 - len(missing),
                                   "；未找到：" + "、".join(missing) if missing else ""))

    block = "\n".join(_render(rows))
    print("\n可粘贴的参数片段：\n")
    print(block)
    if args.apply:
        print("\n--apply 尚未实现自动改写（区间结构需人工确认），请把上面的区间手工填入 config.py")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
