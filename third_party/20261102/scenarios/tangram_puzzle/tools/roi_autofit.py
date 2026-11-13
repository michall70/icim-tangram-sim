"""从一张原料区实拍图自动估算 `raw_roi` 与 `template_roi`。

    python -B -m scenarios.tangram_puzzle.tools.roi_autofit --image <实拍帧>
    python -B -m scenarios.tangram_puzzle.tools.roi_autofit --image <帧> --out <目录>

## 它怎么估

1. **找暗缝**：两块板之间有一条更暗的缝。在图像中部取列亮度剖面，最暗的那一列就是分界。
2. **右侧托盘**：分界以右，按局部阈值取暗区最大连通域 → 外接矩形。托盘按实测 **190mm** 校核。
3. **左侧原料板**：用七块彩色件的包围盒做锚，按实测 **245mm** 与"右侧托盘给的 px/mm"推出板的大小，
   再夹到"分界以左 + 画面有效区"内。
4. **自检**：宽高比接近 1、尺寸与 190/245mm 的偏差 <12%，否则明确报"不可信，请人工确认"。

## 它不做什么

不改 config、不连相机、不连机器人；只打印可直接粘贴的代码行，并存一张叠加图供肉眼核对。
估不准时以 `tools/roi_frame.py --image ...` 人工读数为准。
"""

import argparse
import os

import cv2 as cv
import numpy as np

from scenarios.tangram_puzzle.config import CAMERA_MATRIX, DIST_COEFFS, HOMOGRAPHY
from common.calibration import CameraMapper

RAW_MM = 245.0        # 原料区实测边长
TRAY_MM = 190.0       # 成品区实测边长
SIZE_TOLERANCE = 0.12  # 由 190/245mm 推出的像素尺寸允许偏差


def _find_crease(gray):
    """两块板之间最暗的那一列。"""
    height, width = gray.shape
    band = gray[int(0.2 * height):int(0.9 * height), :].astype(np.float32)
    profile = band.mean(axis=0)
    lo, hi = int(0.45 * width), int(0.75 * width)
    return lo + int(np.argmin(profile[lo:hi]))


def _dark_bbox(gray, x0, x1, threshold_scale=0.75):
    """区间内暗区的最大连通域外接矩形。阈值取区间亮度的 Otsu 结果做缩放。"""
    sub = gray[:, x0:x1]
    valid = sub > 10
    if valid.sum() < 1000:
        return None
    otsu, _ = cv.threshold(sub[valid], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    thr = float(min(otsu * threshold_scale, 140))
    dark = ((sub < thr) & valid).astype(np.uint8)
    dark = cv.morphologyEx(dark, cv.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    dark = cv.morphologyEx(dark, cv.MORPH_OPEN, np.ones((9, 9), np.uint8))
    count, labels, stats, _ = cv.connectedComponentsWithStats(dark, 8)
    if count <= 1:
        return None
    index = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))
    x, y, w, h, area = stats[index]
    return int(x + x0), int(y), int(w), int(h), int(area), thr


def _colored_bbox(image, x0, x1):
    """七块料的联合包围盒。只保留足够大、足够饱和的色块，避免暗背景噪声把框撑满。"""
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, np.array([0, 90, 90], np.uint8), np.array([180, 255, 255], np.uint8))
    mask[:, :x0] = 0
    mask[:, x1:] = 0
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((7, 7), np.uint8))
    count, labels, stats, _ = cv.connectedComponentsWithStats(mask, 8)
    boxes = [stats[i, :4] for i in range(1, count) if stats[i, cv.CC_STAT_AREA] > 2000]
    if not boxes:
        return None
    xs0 = min(int(b[0]) for b in boxes)
    ys0 = min(int(b[1]) for b in boxes)
    xs1 = max(int(b[0] + b[2]) for b in boxes)
    ys1 = max(int(b[1] + b[3]) for b in boxes)
    return xs0, ys0, xs1, ys1


def _best_dark_window(gray, side, right_limit, contain=None):
    """在 [0,right_limit) 内找边长为 side 的窗口，使暗像素占比最大。

    `contain=(x0,y0,x1,y1)`：窗口必须把这些像素包进去（用七块料的包围盒约束，
    保证框一定盖住所有料，不会飘到旁边的黑背景上）。
    """
    height, width = gray.shape
    dark = (gray < 120).astype(np.float32)
    integral = cv.integral(dark)[1:, 1:]

    def fraction(x, y):
        x2, y2 = x + side, y + side
        total = integral[y2 - 1, x2 - 1] - integral[y - 1, x2 - 1] \
            - integral[y2 - 1, x - 1] + integral[y - 1, x - 1]
        return float(total) / float(side * side)

    x_lo, x_hi = 0, min(right_limit - side, width - side)
    y_lo, y_hi = 0, height - side
    if contain is not None:
        cx0, cy0, cx1, cy1 = contain
        x_lo = max(x_lo, cx1 - side + 1)
        x_hi = min(x_hi, cx0)
        y_lo = max(y_lo, cy1 - side + 1)
        y_hi = min(y_hi, cy0)
    if x_hi < x_lo or y_hi < y_lo:
        return None

    best = None
    for y in range(y_lo, y_hi + 1, 4):
        for x in range(x_lo, x_hi + 1, 4):
            score = fraction(x, y)
            if best is None or score > best[0] + 1e-6:
                best = (score, x, y)
    if best is None:
        return None
    return int(best[1]), int(best[2]), int(side), int(side), float(best[0])


def _snap_edges(gray, box, search=45):
    """把矩形四条边吸附到附近的真实边缘（梯度最大处）。

    暗区连通域容易被阴影/反光带偏（实测托盘上沿被算成 121，真实约 175），
    吸附一次能把它拉回真正的板边。
    """
    height, width = gray.shape
    blur = cv.GaussianBlur(gray.astype(np.float32), (7, 7), 0)
    dx = np.abs(np.diff(blur, axis=1))
    dy = np.abs(np.diff(blur, axis=0))
    x, y, w, h = box
    rows = slice(max(0, y + 10), min(height, y + h - 10))
    cols = slice(max(0, x + 10), min(width, x + w - 10))

    def best_in(axis, start, stop, fixed_slice, limit):
        lo, hi = max(0, start - search), min(limit - 1, start + search)
        if hi <= lo:
            return start
        scores = [(float(axis[fixed_slice, pos].mean()), pos) for pos in range(lo, hi)]
        return max(scores)[1]

    left = best_in(dx, x, x, rows, width)
    right = best_in(dx, x + w - 1, x, rows, width)
    top = best_in(dy, y, y, cols, height)
    bottom = best_in(dy, y + h - 1, y, cols, height)
    if right - left < 20 or bottom - top < 20:
        return box
    return int(left), int(top), int(right - left + 1), int(bottom - top + 1)


def estimate(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    height, width = gray.shape
    crease = _find_crease(gray)

    tray = _dark_bbox(gray, crease + 4, width)
    report = {"crease": crease}
    if tray is None:
        raise SystemExit("右侧没有找到托盘暗区；请换一张板外背景更亮的图，或用 roi_frame 人工读数")

    raw_tray = (tray[0], tray[1], tray[2], tray[3])
    snapped = _snap_edges(gray, raw_tray)

    def squareness(box):
        # 托盘近似正方形：用 |宽-高|/宽 衡量；吸附后若更不像正方形就退回原值
        return abs(box[2] - box[3]) / max(box[2], 1)

    chosen = snapped if squareness(snapped) <= squareness(raw_tray) else raw_tray
    tray = (chosen[0], chosen[1], chosen[2], chosen[3], tray[4], tray[5])
    report.update({"tray": tray, "tray_unsnapped": raw_tray,
                   "tray_source": "snapped" if chosen is snapped else "unsnapped"})

    tx, ty, tw, th, _, _ = tray
    px_per_mm = tw / TRAY_MM
    expected = TRAY_MM * px_per_mm
    size_error = abs(th - expected) / expected
    aspect = tw / max(th, 1)
    report["tray_px_per_mm"] = px_per_mm
    report["tray_size_error"] = size_error

    colored = _colored_bbox(image, 0, crease - 2)
    if colored is None:
        raise SystemExit("左侧没有找到彩色件，无法定位原料板；请确认图里有七块料")
    span = max(colored[2] - colored[0], colored[3] - colored[1])
    best = None
    # 右侧托盘给的 px/mm 在左侧会有几个百分点偏差（透视），所以在 ±15% 内搜一遍边长
    for factor in (0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.15):
        side = int(round(RAW_MM * px_per_mm * factor))
        if side < span + 6:
            continue
        found = _best_dark_window(gray, side, crease - 2, contain=colored)
        if found is None:
            continue
        if best is None or found[4] > best[4] + 1e-6:
            best = found
    if best is None:
        raise SystemExit("左侧找不到能盖住全部料块的 245mm 窗口；请用 roi_frame 人工读数")
    rx, ry, rw, rh, fill = best
    report.update({"raw": (rx, ry, rw, rh), "raw_fill": fill, "colored_bbox": colored,
                   "aspect": aspect, "tray_ok": size_error < SIZE_TOLERANCE and 0.8 < aspect < 1.25})
    return report


def draw(image, report, path):
    canvas = image.copy()
    rx, ry, rw, rh = report["raw"]
    tx, ty, tw, th, _, _ = report["tray"]
    cv.rectangle(canvas, (rx, ry), (rx + rw - 1, ry + rh - 1), (60, 200, 60), 2)
    cv.putText(canvas, "raw (245mm)", (rx + 6, ry + 22), cv.FONT_HERSHEY_SIMPLEX, 0.6, (60, 200, 60), 2)
    cv.rectangle(canvas, (tx, ty), (tx + tw - 1, ty + th - 1), (0, 140, 255), 2)
    cv.putText(canvas, "template (190mm)", (tx + 6, ty + 22), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
    cv.line(canvas, (report["crease"], 0), (report["crease"], image.shape[0] - 1), (255, 0, 255), 1)
    cv.imwrite(path, canvas)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="从一张实拍图估算两个 ROI（只读，不改 config）")
    parser.add_argument("--image", required=True, help="原料区实拍帧")
    parser.add_argument("--out", help="叠加图输出目录，默认 results/roi_autofit")
    parser.add_argument("--already-undistorted", action="store_true", help="输入图已去畸变")
    args = parser.parse_args(argv)

    image = cv.imread(args.image, cv.IMREAD_COLOR)
    if image is None:
        raise SystemExit("无法读取图片: %s" % args.image)
    if not args.already_undistorted:
        image = CameraMapper(CAMERA_MATRIX, DIST_COEFFS, HOMOGRAPHY).undistort(image)

    report = estimate(image)
    rx, ry, rw, rh = report["raw"]
    tx, ty, tw, th, _, _ = report["tray"]

    print("暗缝列 x = %d" % report["crease"])
    print("右侧托盘 bbox = (%d, %d, %d, %d)  →  %.1f px/mm，高度与 190mm 偏差 %.1f%%"
          % (tx, ty, tw, th, report["tray_px_per_mm"], 100 * report["tray_size_error"]))
    print("左侧原料 bbox = (%d, %d, %d, %d)  （按 245mm 与块质心推定）" % (rx, ry, rw, rh))
    print("自检：托盘尺寸/形状 %s" % ("通过" if report["tray_ok"] else "不通过 [警告]"))
    print()
    print("可粘贴到 config.py 的 DETECT_PARAMS：")
    print("    raw_roi=(%d, %d, %d, %d)," % (rx, ry, rw, rh))
    print("    template_roi=(%d, %d, %d, %d)," % (tx, ty, tw, th))

    out_dir = args.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                       "results", "roi_autofit")
    os.makedirs(out_dir, exist_ok=True)
    path = draw(image, report, os.path.join(out_dir, "roi_autofit.png"))
    print("\n叠加图: %s（先看一眼两个框贴不贴边，再决定写不写）" % path)
    return 0 if report["tray_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
